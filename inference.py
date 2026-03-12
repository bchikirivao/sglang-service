import logging
import re
import time

import sglang as sgl
from transformers import AutoTokenizer

import config

logger = logging.getLogger("sglang-service")


# Llama 3 special tokens that signal end-of-turn.
LLAMA_STOP_SEQS = [
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
]


def _extract_json(raw: str) -> str:
    """
    Extract the first complete JSON object from raw model output.

    Walks the string with a brace counter and returns only the outermost
    {...} block, discarding everything after (special tokens, commentary).
    """
    start = raw.find("{")
    if start == -1:
        return raw
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return raw


def _repair_json(raw: str) -> str:
    """
    1. Extract just the JSON object (discard anything after the closing }).
    2. Collapse digit,digit sequences caused by small models mis-generating
       numbers e.g. 1,0,0 → 100.
    """
    raw = _extract_json(raw)
    repaired = re.sub(r"(\d),(\d)", lambda m: m.group(1) + m.group(2), raw)
    while repaired != raw:
        raw = repaired
        repaired = re.sub(r"(\d),(\d)", lambda m: m.group(1) + m.group(2), raw)
    return repaired


class InferenceEngine:
    """
    Wrapper around SGLang's embedded Engine.

    The Engine is loaded once at startup. All inference goes through
    async_generate() which feeds SGLang's continuous batching scheduler
    — concurrent NATS requests are batched automatically.

    The tokenizer is loaded separately to apply the chat template before
    passing the formatted prompt string to the Engine.
    """

    def __init__(self) -> None:
        logger.info(
            f"Loading model {config.MODEL_ID} "
            f"(dtype={config.DTYPE}, quant={config.QUANTIZATION or 'none'}) ..."
        )

        engine_kwargs = dict(
            model_path=config.MODEL_ID,
            dtype=config.DTYPE,
            mem_fraction_static=config.MEM_FRACTION_STATIC,
            log_level="error",
        )

        if config.QUANTIZATION:
            engine_kwargs["quantization"] = config.QUANTIZATION

        if config.MAX_TOTAL_TOKENS > 0:
            engine_kwargs["max_total_tokens"] = config.MAX_TOTAL_TOKENS

        if config.DISABLE_RADIX_CACHE:
            engine_kwargs["disable_radix_cache"] = True
            logger.info("RadixAttention prefix cache disabled")

        if config.SPECULATIVE_ALGORITHM:
            engine_kwargs["speculative_algorithm"] = config.SPECULATIVE_ALGORITHM
            engine_kwargs["speculative_num_draft_tokens"] = config.SPECULATIVE_NUM_DRAFT_TOKENS
            engine_kwargs["speculative_num_steps"] = config.SPECULATIVE_NUM_STEPS
            if config.SPECULATIVE_DRAFT_MODEL:
                engine_kwargs["speculative_draft_model_path"] = config.SPECULATIVE_DRAFT_MODEL
                logger.info(
                    f"Speculative decoding: {config.SPECULATIVE_ALGORITHM} "
                    f"draft={config.SPECULATIVE_DRAFT_MODEL} "
                    f"steps={config.SPECULATIVE_NUM_STEPS} "
                    f"tokens={config.SPECULATIVE_NUM_DRAFT_TOKENS}"
                )
            else:
                logger.info(
                    f"Speculative decoding: {config.SPECULATIVE_ALGORITHM} "
                    f"steps={config.SPECULATIVE_NUM_STEPS} "
                    f"tokens={config.SPECULATIVE_NUM_DRAFT_TOKENS}"
                )

        self._engine    = sgl.Engine(**engine_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        logger.info("Model loaded and ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        messages: list[dict],
        system_prompt: str | None,
    ) -> str:
        """Apply the model's chat template to produce a raw prompt string."""
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        return self._tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    # Public async entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[dict],
        system_prompt: str | None,
        grammar: str | None,
        max_tokens: int,
        temperature: float,
        created_at: int | None,
        max_age_ms: int,
    ) -> dict:
        """
        Run inference, returning a result dict.

        TTL check: if created_at (Unix ms) is supplied and the request is
        older than max_age_ms the call is short-circuited.

        grammar (json_schema string): passed directly to SGLang's
        sampling_params["json_schema"]. Unlike mistral.rs, SGLang's
        constrained decoding does not cause number-splitting bugs.

        cached_tokens in the response reflects RadixAttention cache hits —
        a non-zero value means the system prompt prefix was served from
        cache rather than recomputed.
        """

        # -- TTL check ---------------------------------------------------
        if created_at is not None:
            age_ms = (time.time() * 1000) - created_at
            if age_ms > max_age_ms:
                logger.warning(
                    f"Dropping stale request "
                    f"(age={age_ms:.0f}ms > limit={max_age_ms}ms)"
                )
                return {
                    "content":     None,
                    "usage":       None,
                    "performance": None,
                    "error":       f"Request expired after {age_ms:.0f}ms",
                }

        # -- Build prompt from messages + system prompt ------------------
        prompt = self._build_prompt(messages, system_prompt)

        # -- Sampling parameters -----------------------------------------
        sampling_params: dict = {
            "max_new_tokens": max_tokens,
            "temperature":    temperature,
            "stop":           LLAMA_STOP_SEQS,
        }
        if grammar:
            sampling_params["json_schema"] = grammar

        # -- Async inference (feeds continuous batching scheduler) -------
        t0 = time.time()
        result = await self._engine.async_generate(prompt, sampling_params)
        elapsed_ms = (time.time() - t0) * 1000

        meta             = result["meta_info"]
        content          = _repair_json(result["text"])
        prompt_tokens    = meta["prompt_tokens"]
        compl_tokens     = meta["completion_tokens"]
        cached_tokens    = meta.get("cached_tokens", 0)
        tok_per_sec      = round(compl_tokens / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0.0

        return {
            "content": content,
            "usage": {
                "prompt_tokens":    prompt_tokens,
                "completion_tokens": compl_tokens,
                "total_tokens":     prompt_tokens + compl_tokens,
                "cached_tokens":    cached_tokens,
            },
            "performance": {
                "tok_per_sec": tok_per_sec,
                "total_ms":    round(elapsed_ms, 1),
            },
            "error": None,
        }

    def shutdown(self) -> None:
        """Cleanly terminate SGLang subprocesses."""
        logger.info("Shutting down SGLang engine ...")
        self._engine.shutdown()

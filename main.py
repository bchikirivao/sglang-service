"""
sglang-service
==============
A NATS-based LLM inference service backed by SGLang's embedded Engine.

Identical request/response schema to mistralrs-service for direct comparison.

Subjects
--------
Each .txt file in prompts/ registers one subject:
    infer.ecommerce  →  prompts/ecommerce.txt system prompt
    infer.default    →  prompts/default.txt system prompt

Health
------
    infer.health     →  {"status": "ok", "model": "...", "subjects": [...]}

Request schema (JSON)
---------------------
{
    "messages":    [{"role": "user", "content": "..."}],  # required
    "request_id":  "uuid",                                 # optional, echoed back
    "created_at":  1700000000000,                          # optional Unix ms, enables TTL
    "max_tokens":  256,                                    # optional
    "temperature": 0.0                                     # optional
}

Response schema (JSON)
----------------------
{
    "request_id":  "uuid",
    "content":     "...",
    "usage":       {"prompt_tokens": 10, "completion_tokens": 50,
                    "total_tokens": 60, "cached_tokens": 45},
    "performance": {"tok_per_sec": 120.0, "total_ms": 430.0},
    "error":       null
}

Note: cached_tokens > 0 means RadixAttention served part of the prompt
from the prefix cache (system prompt reuse across requests).
"""

import asyncio
import json
import logging
import signal
import sys

from nats.client import connect

import config
from inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sglang-service] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sglang-service")


# ---------------------------------------------------------------------------
# Per-subject inference loop
# ---------------------------------------------------------------------------

async def infer_loop(
    sub,
    subject: str,
    system_prompt: str | None,
    grammar: str | None,
    engine: InferenceEngine,
    nc,
) -> None:
    logger.info(
        f"Listening on {subject!r} "
        f"(prompt={'set' if system_prompt else 'none'}, "
        f"grammar={'json_schema' if grammar else 'none'})"
    )
    async for msg in sub:
        request_id = "n/a"
        result: dict = {}

        try:
            data        = json.loads(msg.data.decode())
            request_id  = data.get("request_id", "n/a")
            messages    = data.get("messages", [])
            max_tokens  = data.get("max_tokens",  config.DEFAULT_MAX_TOKENS)
            temperature = data.get("temperature", config.DEFAULT_TEMPERATURE)
            created_at  = data.get("created_at")

            if not messages:
                raise ValueError("'messages' field is required and must not be empty")

            logger.info(
                f"[{request_id}] {subject} "
                f"messages={len(messages)} "
                f"max_tokens={max_tokens} "
                f"temperature={temperature}"
            )

            result = await engine.run(
                messages=messages,
                system_prompt=system_prompt,
                grammar=grammar,
                max_tokens=max_tokens,
                temperature=temperature,
                created_at=created_at,
                max_age_ms=config.MAX_REQUEST_AGE_MS,
            )

            if result["error"] is None:
                p = result["performance"]
                u = result["usage"]
                logger.info(
                    f"[{request_id}] ok "
                    f"{p['total_ms']:.0f}ms "
                    f"{p['tok_per_sec']} tok/s "
                    f"{u['completion_tokens']} tokens "
                    f"(cached={u['cached_tokens']})"
                )
            else:
                logger.warning(f"[{request_id}] dropped: {result['error']}")

        except Exception as exc:
            logger.error(f"[{request_id}] error: {exc}")
            result = {
                "content":     None,
                "usage":       None,
                "performance": None,
                "error":       str(exc),
            }

        if msg.reply:
            result["request_id"] = request_id
            await nc.publish(msg.reply, json.dumps(result).encode())


# ---------------------------------------------------------------------------
# Health loop
# ---------------------------------------------------------------------------

async def health_loop(sub, subjects: list[str], model_id: str, nc) -> None:
    logger.info(f"Listening on {config.HEALTH_SUBJECT!r}")
    response_bytes = json.dumps({
        "status":   "ok",
        "model":    model_id,
        "subjects": subjects,
    }).encode()

    async for msg in sub:
        if msg.reply:
            await nc.publish(msg.reply, response_bytes)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run() -> None:
    logger.info("Starting sglang-service")
    logger.info(f"  NATS:        {config.NATS_URL}")
    logger.info(f"  Model:       {config.MODEL_ID}")
    logger.info(f"  dtype:       {config.DTYPE}")
    logger.info(f"  quant:       {config.QUANTIZATION or 'none'}")
    logger.info(f"  radix cache: {'disabled' if config.DISABLE_RADIX_CACHE else 'enabled'}")
    logger.info(f"  Subjects:    {list(config.SUBJECTS.keys())}")
    logger.info(f"  Request TTL: {config.MAX_REQUEST_AGE_MS}ms")

    # Load the model — this is the slow step
    engine = InferenceEngine()

    async with await connect(config.NATS_URL) as nc:
        logger.info(f"Connected to NATS at {config.NATS_URL}")

        # Subscribe to all infer subjects (queue group for load balancing)
        infer_subs = []
        for subject in config.SUBJECTS:
            sub = await nc.subscribe(
                subject,
                queue=config.QUEUE_GROUP,
                max_pending_messages=1000,
                max_pending_bytes=64 * 1024 * 1024,
            )
            infer_subs.append(sub)

        # Health check (no queue group — every instance responds)
        health_sub = await nc.subscribe(config.HEALTH_SUBJECT)

        # Signal handling
        loop = asyncio.get_running_loop()

        def _handle_shutdown() -> None:
            logger.info("Shutdown signal received — draining subscriptions ...")
            for s in infer_subs:
                asyncio.create_task(s.drain())
            asyncio.create_task(health_sub.drain())
            engine.shutdown()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_shutdown)

        logger.info("Service ready.")

        try:
            async with asyncio.TaskGroup() as tg:
                for sub, (subject, system_prompt) in zip(infer_subs, config.SUBJECTS.items()):
                    tg.create_task(
                        infer_loop(
                            sub, subject, system_prompt,
                            config.SCHEMAS.get(subject), engine, nc,
                        )
                    )
                tg.create_task(
                    health_loop(health_sub, list(config.SUBJECTS.keys()), config.MODEL_ID, nc)
                )
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.error(f"Task failed: {exc}")
            engine.shutdown()
            sys.exit(1)

    logger.info("Service stopped.")


if __name__ == "__main__":
    asyncio.run(run())

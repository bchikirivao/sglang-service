import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# NATS
# ---------------------------------------------------------------------------
NATS_URL       = os.getenv("NATS_URL",       "nats://localhost:4222")
QUEUE_GROUP    = os.getenv("QUEUE_GROUP",    "sglang-service")
HEALTH_SUBJECT = os.getenv("HEALTH_SUBJECT", "infer.health")

# ---------------------------------------------------------------------------
# Model
#
# HF_TOKEN is read directly from the environment by huggingface_hub / SGLang.
# Set it here or export it before starting the service.
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

# ---------------------------------------------------------------------------
# SGLang Engine options
#
# DTYPE: "auto" lets SGLang pick the best dtype for the hardware (BF16 on
#        Ampere+). Set "float16" for older GPUs.
#
# QUANTIZATION: "" (disabled) | "int4" | "fp8" | "awq" | "gptq"
#   int4  - closest to mistral.rs ISQ Q4K for apples-to-apples comparison
#   fp8   - faster on Hopper+ (H100), good on Ampere with Flash Attention
#   ""    - full BF16, highest quality, uses more VRAM
#
# MEM_FRACTION_STATIC: fraction of GPU memory reserved for the KV cache.
#   SGLang defaults to 0.88. Increase if you have spare VRAM, decrease if
#   the model OOMs at startup.
#
# MAX_TOTAL_TOKENS: total token budget across all sequences in flight.
#   0 = let SGLang decide based on available VRAM.
#
# DISABLE_RADIX_CACHE: set true to disable RadixAttention prefix caching.
#   Useful for benchmarking to isolate raw inference speed.
# ---------------------------------------------------------------------------
DTYPE                = os.getenv("DTYPE",                "auto")
QUANTIZATION         = os.getenv("QUANTIZATION",         "")
MEM_FRACTION_STATIC  = float(os.getenv("MEM_FRACTION_STATIC", "0.88"))
MAX_TOTAL_TOKENS     = int(os.getenv("MAX_TOTAL_TOKENS",   "0"))
DISABLE_RADIX_CACHE  = os.getenv("DISABLE_RADIX_CACHE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Speculative decoding
#
# SPECULATIVE_ALGORITHM: "" (disabled) | "NGRAM" | "STANDALONE" | "EAGLE"
#
#   NGRAM       — no extra model needed. Uses n-gram pattern matching on the
#                 prompt/context to predict tokens. Great for repetitive
#                 structured outputs like JSON. Zero extra VRAM.
#
#   STANDALONE  — small draft model generates candidates, target verifies.
#                 Set SPECULATIVE_DRAFT_MODEL to a smaller model in the same
#                 family (e.g. Llama-3.2-1B-Instruct for a 3B target).
#
#   EAGLE       — lightweight head trained on target hidden states. Highest
#                 acceptance rate but requires a specific EAGLE-trained draft.
#
# SPECULATIVE_NUM_DRAFT_TOKENS: candidate tokens generated per step (default 4).
# SPECULATIVE_NUM_STEPS: draft steps before verification (default 3).
# ---------------------------------------------------------------------------
SPECULATIVE_ALGORITHM        = os.getenv("SPECULATIVE_ALGORITHM",        "")
SPECULATIVE_DRAFT_MODEL      = os.getenv("SPECULATIVE_DRAFT_MODEL",      "")
SPECULATIVE_NUM_DRAFT_TOKENS = int(os.getenv("SPECULATIVE_NUM_DRAFT_TOKENS", "4"))
SPECULATIVE_NUM_STEPS        = int(os.getenv("SPECULATIVE_NUM_STEPS",        "3"))

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_TOKENS  = int(  os.getenv("DEFAULT_MAX_TOKENS",  "256"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# Request TTL
#
# Requests older than MAX_REQUEST_AGE_MS milliseconds are dropped before
# inference runs. Same semantics as mistralrs-service.
# ---------------------------------------------------------------------------
MAX_REQUEST_AGE_MS = int(os.getenv("MAX_REQUEST_AGE_MS", "5000"))

# ---------------------------------------------------------------------------
# Prompts → Subjects
#
# Every .txt file in PROMPTS_DIR becomes one NATS subject:
#   prompts/ecommerce.txt  →  infer.ecommerce
#   prompts/default.txt    →  infer.default
#
# The file contents become the system prompt for that subject.
# To add a new persona: drop a .txt file in the directory and restart.
# ---------------------------------------------------------------------------
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "./prompts"))


def _load_subjects() -> dict[str, str | None]:
    subjects: dict[str, str | None] = {}
    if PROMPTS_DIR.exists():
        for path in sorted(PROMPTS_DIR.glob("*.txt")):
            subject = f"infer.{path.stem}"
            subjects[subject] = path.read_text().strip() or None
    if not subjects:
        subjects["infer.default"] = None
    return subjects


def _load_schemas() -> dict[str, str | None]:
    """
    For each .txt prompt file, look for a companion .schema.json file.
    If found, load it as a JSON string for grammar-constrained generation.
    SGLang uses json_schema in sampling_params — works correctly unlike
    mistral.rs grammar constraints which caused number-splitting bugs.
    Convention: prompts/ecommerce.txt + prompts/ecommerce.schema.json
    """
    schemas: dict[str, str | None] = {}
    if PROMPTS_DIR.exists():
        for path in sorted(PROMPTS_DIR.glob("*.schema.json")):
            subject = f"infer.{path.stem.replace('.schema', '')}"
            schemas[subject] = path.read_text().strip()
    return schemas


SUBJECTS: dict[str, str | None] = _load_subjects()
SCHEMAS:  dict[str, str | None] = _load_schemas()

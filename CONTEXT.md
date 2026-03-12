# sglang-service — Project Context

This document is for a Claude Code session on the Ubuntu machine.
Read this first to understand the project, current state, and what needs to be done.

---

## What This Project Is

A **NATS-based LLM inference service** backed by SGLang's embedded Engine.

The goal is to benchmark SGLang against a similar service built on **mistralrs**, comparing:
- Throughput (tok/s)
- Latency (ms per request)
- Effects of quantization (fp8, awq, etc.)
- Effects of speculative decoding (NGRAM, STANDALONE)
- RadixAttention prefix cache benefits (cached_tokens)

Both services expose an identical JSON request/response schema over NATS so they can be swapped without changing the client.

---

## GitHub Repo

```
https://github.com/bchikirivao/sglang-service
```

The service code lives here. Pull latest before starting work:

```bash
cd ~/sglangstuff/sglang-service
git pull
```

---

## Ubuntu Directory Layout

```
~/sglangstuff/
├── sglang-service/          # This repo — the NATS service
│   ├── main.py              # Entry point, NATS subscribe/publish loops
│   ├── inference.py         # InferenceEngine wrapper around sgl.Engine
│   ├── config.py            # All config via env vars / .env file
│   ├── .env.example         # Template — copy to .env and fill in values
│   ├── prompts/             # .txt files → NATS subjects + system prompts
│   │   └── *.schema.json    # Optional JSON schema for grammar-constrained generation
│   └── requirements.txt
│
├── sglang/                  # SGLang source clone (read-only reference)
│   └── python/sglang/srt/   # Core runtime source
│
└── sglang-venv/             # Python venv with SGLang installed
    └── lib/python3.13/site-packages/sglang/
```

The venv is activated with:
```bash
source ~/sglangstuff/sglang-venv/bin/activate
```

The service is started with:
```bash
cd ~/sglangstuff/sglang-service
source ~/sglangstuff/sglang-venv/bin/activate
python main.py
```

---

## How the Service Works

- `config.py` reads all settings from environment variables (via `.env`)
- Every `.txt` file in `prompts/` registers one NATS subject: `prompts/ecommerce.txt` → `infer.ecommerce`
- The file contents become the system prompt for that subject
- Optional companion `prompts/ecommerce.schema.json` enables JSON grammar-constrained generation
- `main.py` subscribes to all subjects with a queue group (load balancing across instances)
- `inference.py` wraps `sgl.Engine` — concurrent NATS requests are batched automatically by SGLang's continuous batching scheduler
- Health check: publish to `infer.health`, get back `{"status": "ok", "model": "...", "subjects": [...]}`

---

## Request / Response Schema

**Request** (JSON, publish to e.g. `infer.default`):
```json
{
    "messages":    [{"role": "user", "content": "..."}],
    "request_id":  "uuid",
    "created_at":  1700000000000,
    "max_tokens":  256,
    "temperature": 0.0
}
```

**Response**:
```json
{
    "request_id":  "uuid",
    "content":     "...",
    "usage":       {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60, "cached_tokens": 45},
    "performance": {"tok_per_sec": 120.0, "total_ms": 430.0},
    "error":       null
}
```

`cached_tokens > 0` means RadixAttention served part of the prompt from the prefix cache.

---

## Key Config Options (.env)

```bash
# Model
MODEL_ID=meta-llama/Llama-3.2-3B-Instruct   # or Qwen, etc.
HF_TOKEN=your_token_here

# SGLang engine
DTYPE=auto                    # auto = BF16 on Ampere+
QUANTIZATION=                 # empty=BF16, fp8, awq, awq_marlin, gptq, gptq_marlin, qoq
MEM_FRACTION_STATIC=0.88
MAX_TOTAL_TOKENS=0            # 0 = auto

# Speculative decoding
SPECULATIVE_ALGORITHM=        # empty=off, NGRAM, STANDALONE, EAGLE
SPECULATIVE_DRAFT_MODEL=      # only for STANDALONE (e.g. Llama-3.2-1B-Instruct)
SPECULATIVE_NUM_DRAFT_TOKENS=4
SPECULATIVE_NUM_STEPS=3

# TTL
MAX_REQUEST_AGE_MS=5000
```

---

## Performance Numbers So Far

Tested on this machine with various configs:

| Config | Latency | Throughput |
|--------|---------|------------|
| Llama 3.2-3B BF16 | ~1050ms | ~low |
| Llama 3.2-3B fp8 | ~633ms | ~higher |
| Qwen 0.8B BF16 | ~368ms | ~217 tok/s |
| mistralrs (comparison baseline) | faster than SGLang single-request |

Note: SGLang's real advantage over mistralrs shows under **concurrent load** — its continuous batching scheduler amortises compute across simultaneous requests. Single-request benchmarks favour mistralrs.

---

## Current Bug — NGRAM Speculative Decoding Crash

### Status: Fix identified, not yet applied

When `SPECULATIVE_ALGORITHM=NGRAM` is set, the service crashes on first inference:

```
AttributeError: 'NgramVerifyInput' object has no attribute 'topk'
```

### Root Cause

File:
```
~/sglangstuff/sglang-venv/lib/python3.13/site-packages/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
```

The code has EAGLE-specific logic that checks `spec_info.topk > 1`. This works for EAGLE (which has `topk`) but crashes for NGRAM because `NgramVerifyInput` has no `topk` attribute.

There are **4 occurrences** in the file. Search for `topk` to find them all.

### The Fix

For lines like:
```python
if forward_mode.is_target_verify() and spec_info.topk > 1:
```
Change to:
```python
if forward_mode.is_target_verify() and getattr(spec_info, "topk", 1) > 1:
```

For lines like:
```python
if forward_batch.spec_info.topk > 1:
```
Change to:
```python
if getattr(forward_batch.spec_info, "topk", 1) > 1:
```

**Why this works:** `getattr(obj, "topk", 1)` returns `1` if the attribute doesn't exist. `1 > 1` is `False`, so the EAGLE tree-attention block is skipped for NGRAM — which is correct, NGRAM uses flat drafts not trees.

### How to verify the fix

After editing, check no bare `.topk` remain:
```bash
grep -n "topk" ~/sglangstuff/sglang-venv/lib/python3.13/site-packages/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
```

All remaining lines should use `getattr(...)`.

---

## Next Steps (Priority Order)

1. **Apply the NGRAM fix** above and restart with `SPECULATIVE_ALGORITHM=NGRAM`
2. **Benchmark NGRAM** — record latency and tok/s, compare to BF16 baseline
3. **Test STANDALONE speculative decoding** with a 1B draft model:
   ```
   SPECULATIVE_ALGORITHM=STANDALONE
   SPECULATIVE_DRAFT_MODEL=meta-llama/Llama-3.2-1B-Instruct
   ```
4. **Concurrent load benchmark** — run multiple simultaneous requests to show SGLang's batching advantage over mistralrs
5. **Document findings** in a comparison table

---

## Notes on SGLang Quantization

SGLang cannot quantize at load time (unlike mistralrs ISQ). You must use a pre-quantized model:

| QUANTIZATION value | Model needed |
|---|---|
| `fp8` | standard BF16 model works (SGLang applies fp8 internally) |
| `awq` | pre-quantized AWQ model e.g. `hugging-quants/Meta-Llama-3.2-3B-Instruct-AWQ-INT4` |
| `awq_marlin` | same AWQ model, faster kernels on Ampere |
| `gptq` | pre-quantized GPTQ model |

---

## Notes on SGLang vs mistralrs Architecture

- **mistralrs**: single-process, synchronous batching, supports ISQ (quantize any model at load time)
- **SGLang**: multi-process (tokenizer/scheduler/detokenizer workers), continuous batching, RadixAttention prefix cache, requires pre-quantized models
- Both services use identical NATS request/response schema — they are drop-in replaceable from the client's perspective

# Eden.cpp — Local Model Benchmarks (measured 2026-08-05)

Real numbers from our production rig: 2x RTX 5060 Ti 16GB (split GPU
offload), eden-server, generation-only throughput.

## Methodology

- **Isolated servers**: each model served by its OWN eden-server instance
  (`-ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0`).
  NO router-mode autoload — on-demand loading contaminates tok/s by
  2-16x (measured: the 26B read 6.0 tok/s in router mode vs 96.2
  isolated).
- **Warm-up**: each server warmed with a short call before measuring.
- **Generation-only**: tok/s = tokens produced AFTER the first token,
  divided by time after TTFT. Load time is excluded.
- **Thinking models** (qwen35 family): capped with
  `thinking_budget_tokens` so they return visible content instead of
  burning the whole budget on <think> (param per server-common.cpp:1178).
- **Notes format**: mini model cards — `class · context · modality`
  (modality from model configs: image/video = vision-capable,
  omni = text+audio+vision+video).

---

## Gemma Family

| Model | Size | tok/s | TTFT (s) | Notes (class · ctx · modality) |
|---|---|---|---|---|
| gemma4-e2b-native Q4_K_M | 3.3 GB | 119 | 1.17 | 2B dense · 131K · omni (text/audio/vision/video) |
| gemma4-e2b-text | 8.9 GB | 64 | 1.98 | 2B dense · 131K · omni (f16-class — native Q4 wins 2x at 1/3 size) |
| gemma4-E4B Q4_K_M | 5.3 GB | 92 | 1.93 | 4B dense · 32K+ · text |
| gemma-4-12B-uncensored | 8.5 GB | 45 | 3.82 | 12B dense · 32K+ · text |
| gemma-4-26B-A4B Q4_K_M | 16.8 GB | 96 | 1.99 | 26B MoE (A4B) · 128K · omni |
| gemma-4-31B-A4B | 18.7 GB | 16 | 9.30 | 31B MoE (A4B) · 128K · omni |

**Gemma 4 E2B** (2B-class omni) — engine-level notes (deployment role is
an Eden OE choice, not an engine claim):
- Omni: text + audio + vision + video tokens (config: audio_token_id
  258881, image_token_id 258880, video_token_id 258884; 280 vision soft
  tokens/image)
- Sliding-window attention (512) + 20 shared KV layers — tiny KV cache,
  long sessions
- 131K max context; 3.3 GB Q4 fits any modern GPU

---

## Qwen Family

| Model | Size | tok/s | TTFT (s) | Notes (class · ctx · modality) |
|---|---|---|---|---|
| qwen35-2b-q4 | 1.2 GB | 0 | — | 2B dense · 256K · image/video — needs anti-loop penalties (repeat/dry); useless with defaults |
| qwen35-4b-q4 | 2.7 GB | 125 | 1.17 | 4B dense · 256K · image/video |
| qwen35-9b-q4 | 5.6 GB | 68 | 2.23 | 9B dense · 256K · image/video |
| qwen36-27b-q4 | 16.5 GB | 23 | 7.91 | 27B dense · 256K · image/video |
| **qwen36-35b-A3B Q4_K_M** | **20.2 GB** | **173** | 0.97 | **35B MoE (A3B) · 256K · image/video** — fastest in the ladder |

**Qwen3.6-35B-A3B** — the MoE crown jewel: 40 layers, 256 experts / 8 active
(~3B active per token). Conversion fixed via
[PR #12](https://github.com/Project-Glacie/eden.cpp/pull/12)
(Ranger's MTP-exclusion fix, closes
[issue #10](https://github.com/Project-Glacie/eden.cpp/issues/10)) —
E2E verified: converted from vanilla → quantized Q4_K_M → served →
benchmarked **173 tok/s**, the fastest model measured. Note: requires
`thinking_budget_tokens` cap (thinking model); known teardown double-free
at exit ([issue #13](https://github.com/Project-Glacie/eden.cpp/issues/13)),
does not affect serving.

**Qwen3.6-35B-A3B** — measured pending conversion fix (see
[issue #10](https://github.com/Project-Glacie/eden.cpp/issues/10)):
40-layer MoE (256 experts / 8 active), 256K ctx, image/video modality.
Vanilla source complete; GGUF conversion drops `blk.40.attn_norm.weight`.

---

## Operational Findings

1. **Router mode is NOT production-serving viable.** `--models-dir`
   autoload poisons throughput 2-16x. Use it for discovery; serve
   production models as isolated instances.
2. **The 2B cannot answer without anti-loop sampling** — repeat_penalty
   1.0 / dry 0.0 defaults = guaranteed repetition past ~60 tokens.
   Preset: `--repeat-penalty 1.15 --dry-multiplier 1.2 --presence-penalty
   0.3 --frequency-penalty 0.3`.
3. **Thinking models return EMPTY content without `thinking_budget_tokens`**
   — they spend the whole budget reasoning. Cap it per-model.

---

## Bench Scripts

- `model_shootout.py` — router-mode sweep (fast, contaminated — use for
  model discovery only)
- `bench_models.py` — isolated per-model generation-only benchmark
  (the numbers above)

---

*Scorecard goal: expand by family as we test more — Llama, DeepSeek,
Mistral, Phi, etc. Same methodology, same mini-card format, so the
families are directly comparable at the same sizes.*

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

## The Speed Ladder

| Model | tok/s | TTFT (s) | Size | Notes (class · ctx · modality) |
|---|---|---|---|---|
| qwen35-2b-q4 | 0 | — | 1.2 GB | 2B dense · 256K · image/video — needs anti-loop penalties (repeat/dry); useless with defaults |
| qwen35-4b-q4 | 125 | 1.17 | 2.7 GB | 4B dense · 256K · image/video |
| qwen35-9b-q4 | 68 | 2.23 | 5.6 GB | 9B dense · 256K · image/video |
| gemma-4-12b | 45 | 3.82 | 8.5 GB | 12B dense · 32K+ · text |
| qwen36-27b-q4 | 23 | 7.91 | 16.5 GB | 27B dense · 256K · image/video |
| gemma-4-26b (A4B) | 96 | 1.99 | 16.8 GB | 26B MoE (A4B) · 128K · omni |
| gemma-4-31b | 16 | 9.30 | 18.7 GB | 31B MoE (A4B) · 128K · omni |

## Small-Model Comparison (2B-4B class)

| Model | tok/s | TTFT (s) | Size | Notes (class · ctx · modality) |
|---|---|---|---|---|
| gemma4-e2b-native Q4_K_M | 119 | 1.17 | 3.3 GB | 2B dense · 131K · omni (text/audio/vision/video) |
| qwen35-4b-q4 | 102 | 1.18 | 2.7 GB | 4B dense · 256K · image/video |
| gemma4-E4B Q4_K_M | 92 | 1.93 | 5.3 GB | 4B dense · 32K+ · text |
| gemma4-e2b-text | 64 | 1.98 | 8.9 GB | 2B dense · 131K · omni (f16-class — native Q4 wins 2x at 1/3 size) |

# The E2B Model (2B-class omni)

Engine-level notes on the 2B-class omni model (deployment role — executor
vs brain — is an Eden OE choice, not an engine claim):
- **Omni**: text + audio + vision + video tokens (config: audio_token_id
  258881, image_token_id 258880, video_token_id 258884; 280 vision soft
  tokens/image)
- 2B class, 3.3 GB Q4 — fits any modern GPU
- Sliding-window attention (512) + 20 shared KV layers — tiny KV cache,
  long sessions
- 131K max context
- **119 tok/s** measured — fastest small model in the ladder

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

## Bench Scripts

- `model_shootout.py` — router-mode sweep (fast, contaminated — use for
  model discovery only)
- `bench_models.py` — isolated per-model generation-only benchmark
  (the numbers above)

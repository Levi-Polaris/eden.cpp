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

| Model | tok/s | TTFT (s) | Size | Notes |
|---|---|---|---|---|
| qwen35-2b-q4 | 0 | — | 1.2 GB | needs anti-loop penalties (repeat/dry); useless with defaults |
| **qwen35-4b-q4** | **125** | 1.17 | 2.7 GB | fastest complete answers; executor sweet spot |
| qwen35-9b-q4 | 68 | 2.23 | 5.6 GB | strong mid |
| gemma-4-12b | 45 | 3.82 | 8.5 GB | solid |
| qwen36-27b-q4 | 23 | 7.91 | 16.5 GB | big brain, slower |
| **gemma-4-26b (A4B)** | **96** | 1.99 | 16.8 GB | production primary |
| gemma-4-31b | 16 | 9.30 | 18.7 GB | biggest, slowest |

## Ship Candidates (the small ones that matter)

| Model | tok/s | TTFT (s) | Size | Notes |
|---|---|---|---|---|
| **gemma4-e2b-native Q4_K_M** | **119** | 1.17 | **3.3 GB** | 🏆 ship pick — omni, sliding-window, 131K ctx |
| qwen35-4b-q4 | 102 | 1.18 | 2.7 GB | most complete answers (1032 chars) |
| gemma4-E4B Q4_K_M | 92 | 1.93 | 5.3 GB | solid, bigger |
| gemma4-e2b-text | 64 | 1.98 | 8.9 GB | f16-class bloat — native Q4 wins 2x at 1/3 size |

## The Ship Model: Gemma 4 E2B (native Q4_K_M)

The recommended default brain for Eden OE public runtimes:
- **Omni**: text + audio + vision + video tokens (config: audio_token_id
  258881, image_token_id 258880, video_token_id 258884; 280 vision soft
  tokens/image)
- 2B class, 3.3 GB Q4 — fits any modern GPU
- Sliding-window attention (512) + 20 shared KV layers — tiny KV cache,
  long sessions
- 131K max context
- **119 tok/s** measured — as fast as the 26B production brain at 1/5
  the size

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

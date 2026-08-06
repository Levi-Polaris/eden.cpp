# eden.cpp — Sovereign Inference Engine

A frozen fork of llama.cpp, hardened for sovereignty: native Blackwell FP4 (NVFP4), Vulkan + CUDA backends, embedded chat UI with persistence, and architecture support for **Gemma 3/4, Qwen 3/3.5 (+MTP/MoE), DeepSeek 2, and 180+ architectures**.

**MIT licensed. One binary. Zero cloud.**

---

## Quick Start

```bash
# Download a base model (Gemma 4 26B-A4B — our production model, ~103 tok/s on 2x 5060 Ti)
wget https://huggingface.co/Huihui/Gemma-4-26B-A4B-abliterated-GGUF/resolve/main/Huihui-gemma-4-26B-A4B-abliterated-Q4_K.gguf

# Or our distilled LoRA (Qwen3.5-4B base)
wget https://huggingface.co/FrostiSteele/eden-4b-distilled-lora/resolve/main/eden-4b-distilled-lora.gguf

# Build and run
git clone https://github.com/Project-Glacie/eden.cpp
cd eden.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DEDEN_BUILD_SERVER=ON
make -j$(nproc) eden-server
./bin/eden-server -m /path/to/model.gguf --port 9094

# Open http://localhost:9094/eden-chat.html
```

Fits 4GB+ VRAM GPUs (Qwen3.5-4B Q4_K_M: 2.9GB base + 85MB LoRA ≈ 3GB total).

---

## Supported Architectures

The engine runs any GGUF model for **180+ architectures** (full llama.cpp set plus our additions). Notable:

| Family | Architectures |
|---|---|
| **Gemma** | gemma, gemma2, gemma3, gemma3n, **gemma4** |
| **Qwen** | qwen, qwen2, qwen2moe, qwen2vl, qwen3, qwen3moe, qwen3next, qwen3vl, qwen3vlmoe, **qwen35, qwen35moe, qwen35_mtp, qwen35moe_mtp** |
| **DeepSeek** | deepseek, deepseek2, deepseek2-ocr |
| **Llama** | eden (llama), eden4 (llama4), deci, falcon, grok, gpt2, gptj, gptneox, mpt, baichuan, starcoder, refact, bloom, stablelm |
| **BERT family** | bert, modern-bert, nomic-bert (+moe), neo-bert, jina-bert-v2/v3, eurobert |
| **Others** | phi2, phi3, phimoe, plamo/2/3, codeshell, orion, internlm2, minicpm/3, and 130+ more |

**Gemma 4** (gemma4) is our production architecture — the engine serves it at 103 tok/s on 2x RTX 5060 Ti.

---

## Download Models

| Model | Size | Source |
|---|---|---|
| Gemma 4 26B-A4B Q4_K_M (production) | 16.8 GB | [Huihui](https://huggingface.co/Huihui) |
| Base (Qwen3.5-4B Q4_K_M) | 2.9 GB | [Bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF) |
| Eden Distilled LoRA | 85 MB | [FrostiSteele](https://huggingface.co/FrostiSteele/eden-4b-distilled-lora) |

**NVFP4** = Blackwell (RTX 5060+, tool available). **Q4_K_M** = all GPUs + CPU (recommended).

---

## Build

### CUDA (Blackwell — RTX 5060 series)

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DEDEN_BUILD_SERVER=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
make -j$(nproc) eden-server
```

Requires CUDA 12.8+. Native Blackwell FP4 (`BLACKWELL_NATIVE_FP4 = 1` confirmed in silicon).

> **Compiler pitfall:** CMake may resolve `/usr/bin/nvcc` (system CUDA 12.0) instead of the full toolkit. The 12.0 nvcc does NOT support `compute_120a`. Always pass `-DCMAKE_CUDA_COMPILER=` explicitly. See [BUILD.md](BUILD.md).

### Vulkan (AMD, Intel, older NVIDIA — any GPU)

```bash
mkdir build-vulkan && cd build-vulkan
cmake .. -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DEDEN_BUILD_SERVER=ON
make -j$(nproc) eden-server
```

### CPU-only

```bash
mkdir build-cpu && cd build-cpu
cmake .. -DGGML_CUDA=OFF -DEDEN_BUILD_SERVER=ON
make -j$(nproc) eden-server
```

---

## Tools

| Tool | What It Does |
|---|---|
| `eden-server` | OpenAI-compatible API on :9094, embedded chat UI, multi-model serving |
| `eden-quantize` | Quantize GGUF models to NVFP4, Q4_K_M, Q8_0, f16, Q2_K–Q6_K, TBQ3_0, TBQ4_0 (EdenQuant-Turbo) |
| `eden-perplexity` | Benchmark PPL on wikitext. Compare quantization formats |
| `eden-imatrix` | Importance matrix calibration for optimal quantization |
| `eden-kv-cache` | KV cache management (including ISWA incremental sliding-window partitioning) |

### Quantize a model

```bash
./bin/eden-quantize model-f16.gguf model-NVFP4.gguf NVFP4
```

Supported formats: `NVFP4`, `Q4_K_M`, `Q8_0`, `Q2_K`, `Q3_K_M`, `Q4_K_S`, `Q5_K_M`, `Q6_K`, `f16`, `TBQ3_0`, `TBQ4_0`

### Benchmark PPL

```bash
./bin/eden-perplexity -m model.gguf -f wikitext-test.txt -ngl 99 -c 512
```

---

## API

OpenAI-compatible endpoint at `http://localhost:9094/v1/chat/completions`:

```bash
curl http://localhost:9094/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

Also: `/v1/models`, `/health`, `/eden-chat.html`, `/index.html` (WebUI).

---

## What We Added vs. Upstream

- **NVFP4 quantization tool** — Quantize to NVFP4 format (4.85 BPW). Confirmed: `BLACKWELL_NATIVE_FP4 = 1` on RTX 5060 series. Tool available, production models ship Q4_K_M.
- **Gemma 4 architecture** — Full Gemma 3/3N/4 support including the A4B MoE variants we run in production.
- **QWEN35 architecture** — Qwen3.5 + MTP layer support + hybrid attention + MoE variants.
- **DeepSeek 2** — deepseek, deepseek2, deepseek2-ocr support.
- **EdenQuant-Turbo (TBQ)** — TBQ3_0, TBQ4_0 quantization formats from Indras-Mirror (MIT).
- **KV cache work** — `eden-kv-cache` + ISWA (incremental sliding-window attention) partitioning; KV q8_0 halves VRAM.
- **Multimodal** — mmproj/CLIP projector support for vision models.
- **Vulkan backend** — SPIR-V header fixes for Ubuntu, cross-GPU compatibility.
- **Embedded chat UI** — `eden-chat.html`, zero dependencies, dark theme, conversation persistence.
- **Full eval pipeline** — `eden-perplexity` + `eden-imatrix`, benchmarked across formats.
- **Frozen fork** — We don't track upstream. We cherry-pick what we need.

---

## Performance

**Generation throughput — measured 2026-08-05 on our production rig (2x RTX 5060 Ti 16GB, split GPU offload). Isolated servers, warm-up, generation-only (excludes load time). Full data + methodology: [BENCHMARKS.md](docs/BENCHMARKS.md).**

| Model | Size | Generation tok/s | TTFT | Notes |
|---|---|---|---|---|
| **gemma4-e2b-native Q4_K_M** | **3.3 GB** | **~119** | 1.2s | 2B-class omni (text/audio/vision/video), 131K ctx |
| qwen35-4b-q4 | 2.7 GB | ~125 | 1.2s | fastest of the ladder |
| qwen35-9b-q4 | 5.6 GB | ~68 | 2.2s | strong mid |
| gemma-4-12B-uncensored | 8.5 GB | ~45 | 3.8s | solid |
| qwen36-27b-q4 | 16.5 GB | ~23 | 7.9s | big brain |
| **gemma-4-26B-A4B Q4_K_M** | **16.8 GB** | **~96** | 2.0s | dense-class MoE (A4B) |
| gemma-4-31B-A4B | 18.7 GB | ~16 | 9.3s | biggest |

**The model data:** all numbers are engine benchmarks — what eden.cpp does with each GGUF, isolated and generation-only. Which models a runtime *deploys* (brain vs executor vs embedder) is an OE-level choice, not an engine claim.

**Methodology:** isolated per-model eden-server instances (`-ngl 99 --flash-attn on --cache-type-k/v q8_0`), warmed, then generation-only tok/s (tokens after TTFT ÷ time after TTFT). Router-mode `--models-dir` autoload contaminates throughput 2-16x — it's for discovery, not serving. Thinking models capped via `thinking_budget_tokens`. See [BENCHMARKS.md](docs/BENCHMARKS.md) for the full numbers, the 2B anti-loop sampler findings, and the operational notes.

> Note: earlier README revisions listed throughput in the thousands of tok/s — those were prefill (prompt-processing), not generation. Generation is 20-125 tok/s on this hardware. Benchmarks without a stated methodology should be treated as unverified.

**Methodology:** live `/v1/chat/completions` calls on the production server, wall-clock including model thinking time, tokens counted from API `usage.completion_tokens`. These are end-to-end generation speeds a user actually experiences.

> Note: earlier README revisions listed throughput in the thousands of tok/s. Those figures are **prompt-processing (prefill) throughput**, not generation — prefill processes the full input batch in parallel and legitimately reaches 5-15k tok/s on Blackwell, but it is NOT the speed at which the model writes tokens. Generation (the number that matters for interactive use) is 20-100 tok/s on this hardware. The table above shows generation. Benchmarks without a stated methodology should be treated as unverified.

---

## License

MIT. See [LICENSE](LICENSE) for full text.

- Our contributions: Copyright (c) 2026 Project Glacie LLC
- Upstream: Copyright (c) 2023-2026 The ggml authors
- TurboQuant: Copyright (c) 2024 Indras-Mirror (MIT)
- Base models: Qwen3.5 family under Apache 2.0
- Vendored deps retain original MIT/Apache licenses

See [NOTICE](NOTICE.md) for full attributions.

---

## Community

- [GitHub](https://github.com/Project-Glacie/eden.cpp)
- [HuggingFace](https://huggingface.co/FrostiSteele)
- Builds in public. Weekly. No investors. No cloud dependency.

---

**The best engineering makes the impossible feel obvious.**

— Haven Steele, COO, Project Glacie

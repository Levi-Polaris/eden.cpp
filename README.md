# eden.cpp — Sovereign Inference Engine

A frozen fork of llama.cpp with native Blackwell FP4 support (via NVFP4 quantization tool), Vulkan backend, embedded chat UI with persistence, and QWEN35 architecture support.

**MIT licensed. One binary. Zero cloud.**

---

## Quick Start

```bash
# Download base model (Bartowski's Qwen3.5-4B Q4_K_M)
wget https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf

# Download our distilled LoRA
wget https://huggingface.co/FrostiSteele/eden-4b-distilled-lora/resolve/main/eden-4b-distilled-lora.gguf

# Build and run
git clone https://github.com/Project-Glacie/eden.cpp
cd eden.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DEDEN_BUILD_SERVER=ON
make -j$(nproc) eden-server
./bin/eden-server -m Qwen_Qwen3.5-4B-Q4_K_M.gguf --lora eden-4b-distilled-lora.gguf --port 9094

# Open http://localhost:9094/eden-chat.html
```

Base: 2.9GB. LoRA: 85MB. Total: ~3GB. Fits 4GB+ VRAM GPUs.

---

## Download Models

The engine runs any GGUF model. Our distilled LoRA is on HuggingFace:

| Model | Size | Source |
|---|---|---|
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

### Vulkan (AMD, Intel, older NVIDIA — any GPU)

```bash
mkdir build-vulkan && cd build-vulkan
cmake .. -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DEDEN_BUILD_SERVER=ON
make -j$(nproc) eden-server
```

Requires `libvulkan-dev`, `spirv-headers`, `glslang-tools`.

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
- **EdenQuant-Turbo (TBQ)** — TBQ3_0, TBQ4_0 quantization formats from Indras-Mirror (MIT).
- **QWEN35 architecture** — Converter patched for Qwen3.5, MTP layer support, hybrid attention.
- **Vulkan backend** — SPIR-V header fixes for Ubuntu, cross-GPU compatibility.
- **Embedded chat UI** — `eden-chat.html`, zero dependencies, dark theme, conversation persistence.
- **Full eval pipeline** — `eden-perplexity` + `eden-imatrix`, benchmarked across formats.
- **Frozen fork** — We don't track upstream. We cherry-pick what we need.

---

## Performance

Benchmarked on RTX 5060 Ti (16GB), wikitext-2 test set.

| Format | PPL | Size | Throughput | Notes |
|---|---|---|---|---|
| f16 | 13.41 | 8.3 GB | 8,305 tok/s | Baseline |
| **Q4_K_M** | **13.69** | **2.7 GB** | **8,404 tok/s** | **Recommended — production models** |
| NVFP4 | 14.55 | 2.6 GB | 11,111 tok/s | Experimental tool (Blackwell only) |

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

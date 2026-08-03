# NOTICE — Attributions

eden.cpp is built on open-source work. The following components are incorporated.

---

## Upstream Engine

**llama.cpp** — MIT License
Copyright (c) 2023-2026 The ggml authors
https://github.com/ggerganov/llama.cpp

**ggml** — MIT License
Tensor library for machine learning
Copyright (c) 2023-2026 The ggml authors
https://github.com/ggerganov/ggml

**cpp-httplib** — MIT License
A C++ header-only HTTP/HTTPS server and client library
Copyright (c) 2020-2024 yhirose

**TurboQuant (TBQ) quantization formats** — MIT License
Copyright (c) 2024 Indras-Mirror
https://github.com/Indras-Mirror/llama.cpp-mtp
EdenQuant-Turbo (TBQ3_0, TBQ4_0) are renamed variants of TurboQuant.

---

## Eden Contributions

All modifications, additions, and new files are:
Copyright (c) 2026 Project Glacie LLC
MIT License

Key additions:
- NVFP4 quantization format and Blackwell FP4 kernels
- QWEN35 architecture support in GGUF converter
- eden-chat.html embedded chat interface
- Vulkan backend SPIR-V compatibility fixes
- Distillation pipeline scripts
- Perplexity and imatrix evaluation tools
- Multi-model fleet architecture

---

## Base Models

Our distilled models are derived from:

**Qwen3.5 family** — Apache 2.0 License
Copyright (c) 2025 Alibaba Cloud
https://huggingface.co/Qwen

Our LoRA weights, merged models, and quantized formats are our own work.
The base architecture and tokenizer are provided under Apache 2.0.

**OpenOrca dataset** — MIT License
https://huggingface.co/datasets/Open-Orca/OpenOrca
Used for distillation training (20,000 samples).

---

## Build Dependencies

| Component | License | Purpose |
|---|---|---|
| CUDA Toolkit 12.8 | NVIDIA EULA | CUDA backend (optional) |
| Vulkan SDK | Apache 2.0 | Vulkan backend (optional) |
| SPIR-V Headers | MIT / Khronos | Vulkan shader compilation |
| shaderc / glslang | Apache 2.0 | Vulkan shader compilation |

These are build-time dependencies only. They are not distributed with eden.cpp.

---

## Training Dependencies

| Component | License | Purpose |
|---|---|---|
| PyTorch | BSD | Model loading and training |
| Transformers | Apache 2.0 | HuggingFace model loading |
| PEFT | Apache 2.0 | LoRA training and merging |
| BitsAndBytes | MIT | 4-bit model quantization for teachers |
| Datasets | Apache 2.0 | Training data loading |
| huggingface_hub | Apache 2.0 | Model upload/download |

These are used in the distillation pipeline only. They are not distributed with eden.cpp.

---

No code from these training libraries is included in the eden.cpp source tree.
Training scripts are provided separately in our tooling repository.

---

To the upstream authors: thank you. Sovereignty is contagious.

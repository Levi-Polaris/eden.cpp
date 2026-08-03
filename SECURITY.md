# Security Policy

## Reporting a Vulnerability

Do NOT open a public issue. Email: **frosti@projectglacie.tech**

We'll respond within 48 hours and work with you on a fix.

## Our Security Posture

eden.cpp runs locally. There is no cloud component. No telemetry. No data leaves your machine.
This eliminates entire classes of attack — no API key theft, no cloud credential exfiltration,
no remote code execution via network.

## What We Protect Against

- Malicious GGUF files (validate model structure before loading)
- Prompt injection (mitigated by our model routing architecture)
- Resource exhaustion (configurable context limits and batch sizes)

## What We Cannot Protect Against

- Malicious code running on the same machine with user privileges
- Physical access to the hardware
- Upstream vulnerabilities in CUDA, Vulkan, or system libraries

## Dependencies

We vendor or copy only what's needed. Dependencies are documented in [NOTICE.md](NOTICE.md).

Build-time dependencies (CUDA Toolkit, Vulkan SDK) are NOT distributed with eden.cpp.
Training-time dependencies (PyTorch, Transformers) are NOT included in the binary.

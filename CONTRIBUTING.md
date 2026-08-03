# Contributing to eden.cpp

We welcome contributions that align with our thesis: sovereign, local-first AI.

## Our Fork Policy

eden.cpp is a **frozen fork** of llama.cpp. We do not track upstream. We cherry-pick the
features and fixes that matter for our use case. PRs that add meaningful capability
(NVFP4 improvements, new quantization formats, multi-model routing) are welcome.
PRs that merge upstream llama.cpp changes without clear benefit will be rejected.

## Before Contributing

1. Read our [README](README.md) to understand what eden.cpp is
2. Check existing issues and PRs
3. For large changes, open an issue first to discuss

## Pull Request Process

1. Fork the repo and create a feature branch
2. Test your changes across relevant backends (CUDA, Vulkan, CPU)
3. Update documentation if needed
4. Submit PR against `main`
5. All PRs require review before merge

## Code Style

- Follow the existing code patterns in the repo
- C++17, no exceptions in performance-critical paths
- Keep it simple. We prefer readable over clever.

## License

All contributions are MIT licensed. See [LICENSE](LICENSE).

# eden.cpp — Build Guide & Troubleshooting

## Build (verified 2026-08-05)

The repo was NOT buildable as shipped on main — the root `CMakeLists.txt`
was a byte-identical copy of `src/CMakeLists.txt`, missing the parent
build entry entirely. Fixed in PR #2. This doc records the correct,
verified build path.

### CPU-only build

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=OFF -DEDEN_BUILD_SERVER=ON
make -j$(nproc) eden-server
```

### CUDA (Blackwell — RTX 5060 series)

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DEDEN_BUILD_SERVER=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
make -j$(nproc) eden-server
```

> **Compiler pitfall (2026-08-05):** CMake may resolve `/usr/bin/nvcc`
> (system CUDA 12.0) instead of the full toolkit at
> `/usr/local/cuda-12.8/bin/nvcc`. The 12.0 nvcc does NOT support
> `compute_120a` (Blackwell) and fails with
> `nvcc fatal: Unsupported gpu architecture 'compute_120a'`.
> Always pass `-DCMAKE_CUDA_COMPILER=` explicitly and confirm with
> `nvcc --list-gpu-arch` that 120 is present.

### Vulkan (AMD, Intel, older NVIDIA)

```bash
mkdir build-vulkan && cd build-vulkan
cmake .. -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DEDEN_BUILD_SERVER=ON
make -j$(nproc) eden-server
```

## What the root CMakeLists.txt does

- `project("eden.cpp")` — parent project entry (was missing; the file
  shipped as a copy of src/CMakeLists.txt)
- Defines `EDEN_INSTALL_VERSION "1.1.0"` — consumed by
  `set_target_properties(... VERSION ...)` in src/ and common/
- `add_subdirectory(ggml)`, `common`, `src`, `vendor/cpp-httplib`, `tools`
- `include(cmake/license.cmake)` + `license_generate(eden-common)` —
  bakes LICENSE into the eden-common library (arg.cpp references
  `LICENSES[]`)

## Broken-file forensics (what was wrong)

| Symptom | Root cause | Fixed by |
|---|---|---|
| `set_target_properties` error (src:46, common:121) | `EDEN_INSTALL_VERSION` never defined (broken root) | root defines it |
| `Unknown command "license_add_file"` | `cmake/license.cmake` never included | root includes it |
| `undefined reference to LICENSES` (link) | `license_generate()` never called | root calls `license_generate(eden-common)` |
| `-lcpp-httplib` not found (link) | vendored lib never added as subdir | `add_subdirectory(vendor/cpp-httplib)` |
| Missing `tests/`/`pocs/` dirs | broken root referenced them | replaced root doesn't |
| `compute_120a` unsupported | CMake picked system CUDA 12.0 nvcc | explicit `-DCMAKE_CUDA_COMPILER` |

## Verification

- CPU build: PASS (configure → build → run → `/health` OK, models listed)
- Smoke test: eden-server served qwen35-4b on :9192, embedded UI assets
  (bundle.css/js/chat.html) generated
- CUDA build: verified on RTX 5060 Ti with CUDA 12.8 (compute_100/120)

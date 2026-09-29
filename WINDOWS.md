# Native Windows build — RTX 5070 Ti (text-only)

This branch (`windows-port`) adds a minimal native Windows (MSVC) build of the RTX 5080 runtime so the
canonical Qwen3.8-27B artifact runs on Windows 11 without WSL. It was developed and measured on a
single **RTX 5070 Ti 16 GB** (GB203, `sm_120a`), the same chip family and VRAM size as the RTX 5080.

## Built on

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — the original NInfer engine.
- [ruwwww/ninfer-5060ti](https://github.com/ruwwww/ninfer-5060ti) — the 16 GB consumer-Blackwell fork this line grew from.
- [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080) — the Qwen3.8-27B true-128K runtime and
  artifact this branch ports. All model, kernel and scheduling work is theirs; this branch only adds the Windows layer.

Model artifact (unchanged): [ninfer-5080/Qwen3.8-27B-RTX5080](https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080),
`qwen3_8_27b.ninfer`, 16,461,267,456 bytes, SHA-256 `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21`.

## What changed (commit "Native Windows (MSVC) build for text inference")

Additive and guarded; Linux builds are unchanged. No CUDA kernel math was changed.

| Area | Change |
|---|---|
| CMake | `NINFER_ENABLE_MEDIA` option (default `ON`); `OFF` drops FFmpeg/libcurl and builds media stubs (text-only). MSVC flags: `/Zc:preprocessor` (required by CCCL), `/Zc:__cplusplus`, `/utf-8`, `/EHsc`, `/bigobj`, defines `NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS UTF8PROC_STATIC`. Copies `cudart64_*.dll` next to the executables. |
| Artifact reader | Windows `MappedFile`: `CreateFileMappingW`/`MapViewOfFile` for metadata and a `FILE_FLAG_NO_BUFFERING` handle with `OVERLAPPED` offsets for the 4096-byte aligned direct reads. |
| NVFP4 TMA kernels | MSVC cannot pass the 128-byte-aligned `CUtensorMap` descriptors by value (C2719); on MSVC these launchers are stubs that throw. The Q3/Q4/Q5 artifact never uses NVFP4. |
| Runtime | Explicit move constructors for `SequencePlan`/`RequestBasePlan`/`RequestPlan` (MSVC does not emit defaulted explicit specializations → LNK2019). |
| Small POSIX shims | `_isatty`, `localtime_s`, `_getpid`; device `memcpy` instead of `__builtin_memcpy` under MSVC. |
| Tests | `constexpr std::sqrt` → `const`, `_aligned_malloc`, `_getpid`. |

Repository hygiene: with Git for Windows' default `core.autocrlf=true`, text fixtures are checked out with CRLF and
the 118K qualification fixture's SHA-256 no longer matches. Clone with `git -c core.autocrlf=false clone ...`.

## Build

Requirements: Visual Studio 2022 Build Tools (MSVC 14.44 used), CUDA Toolkit ≥ 13.1 (13.2 used; it can live
side by side with an older default toolkit), the CMake and Ninja bundled with Visual Studio.

From an x64 Native Tools command prompt with the CUDA 13.x `bin` directory on `PATH`:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_ENABLE_MEDIA=OFF ^
  "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe"
ninja -C build
```

Add `-DBUILD_TESTING=ON` for the tests and `-DNINFER_BUILD_BENCHMARKS=ON` for the operator benchmarks.

## Run (OpenAI/Anthropic-compatible server)

```bat
build\apps\ninfer-serve.exe qwen3_8_27b.ninfer --host 127.0.0.1 --port 8080 --model-id llama ^
  --max-context 131072 --kv-capacity 131072 --prefill-chunk 896 --kv-dtype q4 --embedding-host ^
  --spec mtp --draft-tokens 3 --max-concurrency 1 --default-thinking-budget 2048 --preserve-thinking
```

- `--preserve-thinking` makes multi-turn agent sessions append-only when the client echoes `reasoning_content`
  (opencode and Pi both do), so later turns reuse the cached prefix instead of re-prefilling the history.
- Windows gives each process a VRAM budget below physical VRAM. With the display on the same GPU, 131072 fit over
  Remote Desktop but only ~64K at the local console; lower `--max-context`/`--kv-capacity` to what starts.
  The server refuses to start rather than overcommitting, but other apps growing later can push memory into
  system RAM (NVIDIA "CUDA Sysmem Fallback Policy"), which shows up as a silent slowdown.

## Measured (RTX 5070 Ti, 128K/128K Q4 KV, MTP-3, CUDA Graph ON, greedy)

| Prompt | Prefill tok/s | Decode tok/s |
|---|---|---|
| 37 tokens | — | 117 |
| 6.7K tokens of source code | 1,889 | 102 |
| 43.5K tokens of source code | 1,567 | 85 (98 sustained over 2,048 tokens) |
| 118K `workflow-118k-v1` fixture | 1,137 | 103 |

The 118K canonical fixture reproduces the upstream MTP statistics exactly (44.74%, 2.31 tok/round).
Test suite: all failures are explained (media/NVFP4 disabled paths, or upstream test drift); all 9 model-backed
decision tests pass.

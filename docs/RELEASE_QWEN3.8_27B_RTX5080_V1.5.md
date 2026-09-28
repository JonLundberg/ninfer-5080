# Qwen3.8-27B RTX 5080 128K + Vision v1.5

## Release summary

v1.5 moves the validated RTX 5080 production runtime to the CUDA 13.4 / R615 baseline and retunes the Q3/A8 large-prefill SwiGLU schedule for Blackwell/CC 12.0.

The model artifact is unchanged:

```text
MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

Qualification platform:

```text
GPU=NVIDIA GeForce RTX 5080 16 GB
CUDA=13.4.92
DRIVER=615.71.09
MAX_CONTEXT=131072
KV_CAPACITY=131072
KV_DTYPE=q4-group64
MTP=3
CUDA_GRAPH=ON
EMBEDDING_HOST=ON
VISION_PROFILE=2048
```

## Canonical benchmark contract

From this release onward, `ninfer_bench` is the source of truth for whole-model performance.

```text
fixture=bench/fixtures/workflow-118k-v1/ninfer_bench_118001.ids
tokens=118001
corpus_sha256=5b08da2c7b7ea5cafad2fab5699dccbcbce86040d8a37219b8c21f094d1d1eb7
test=pp118001+tg2048
warmup=1
measured_repetitions=2
max_context=131072
kv_capacity=131072
kv_dtype=q4-group64
mtp_draft_tokens=3
cuda_graph=on
embedding_host=on
```

The benchmark itself disables model-default stops so all measured runs include the exact requested sustained decode length. Q4 benchmark support is part of the retained `ninfer_bench` interface.

## Final old-vs-new performance gate

The old production configuration and Schedule A were compared on the exact same immutable token-ID corpus and benchmark contract.

| Metric | Old production | v1.5 | Change |
|---|---:|---:|---:|
| Prefill chunk | 896 | **1792** | — |
| Prefill, 2-run mean | 1,315.712 tok/s | **1,374.383 tok/s** | **+4.459%** |
| Sustained decode, 2-run mean | 109.666 tok/s | **112.215 tok/s** | **+2.324%** |
| Decode fallbacks | 0 | **0** | unchanged |
| Workspace | 115.999 MiB | **231.998 MiB** | +115.999 MiB |
| Planned slack | ~843.17 MiB | **~727.23 MiB** | -115.94 MiB |

Result:

```text
PERFORMANCE_GATE=PASS
RELEASE_DECISION=PUBLISH_SCHEDULE_A
```

## Blackwell Q3/A8 retune

Previous large-prefill schedule:

```cpp
Q3Int8SwiGluSchedule<64, 256, 16, 128, 3, 1>
```

v1.5 schedule:

```cpp
Q3Int8SwiGluSchedule<64, 128, 16, 64, 3, 1>
```

CUDA 13.4 changed code generation for the previous Full=false large-prefill path and introduced local-memory spill traffic. Structural qualification showed the old CUDA 13.4 kernel at 255 registers with stack/local spill traffic, while Schedule A compiled at approximately 204–206 registers with zero stack/local allocation and no LDL/STL spill instructions.

Decode and small-T kernels are intentionally unchanged by this schedule retune.

## Recommended production profile

```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b.ninfer \
  --host 0.0.0.0 \
  --port 8080 \
  --model-id local-model \
  --max-context 131072 \
  --kv-capacity 131072 \
  --prefill-chunk 1792 \
  --kv-dtype q4 \
  --spec mtp \
  --draft-tokens 3 \
  --max-concurrency 1 \
  --max-pending-requests 16 \
  --pending-timeout-ms 180000 \
  --embedding-host \
  --vision \
  --vision-max-tokens 2048 \
  --default-thinking-budget 2048 \
  --prefix-checkpoint-policy rolling-tool
```

CUDA Graph remains enabled by default.

## Benchmark policy

Whole-model release performance claims use the canonical `ninfer_bench` contract above. Additional context-size sweeps, kernel microbenchmarks, Nsight profiling and serving measurements are diagnostic and should be run only to answer a specific engineering question; they do not replace the canonical benchmark.

## Validation status

| Validation | Result |
|---|---|
| Canonical immutable token-ID corpus | PASS |
| Q4 `ninfer_bench` support | PASS |
| CUDA 13.4.92 build | PASS |
| NVIDIA 615.71.09 driver baseline | PASS |
| 131,072 context / Q4 KV | PASS |
| MTP-3 / CUDA Graph | PASS |
| Old-vs-new canonical performance gate | PASS |
| No decode fallback regression | PASS |
| Memory envelope at chunk 1792 | PASS |
| Final production serving smoke | pending deployment |


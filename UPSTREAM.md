# Upstream

Code this extension copies from elsewhere, and where it came from. Copies are
adapted here, not shared; changes to them are this repository's own.

| files | origin | notes |
| --- | --- | --- |
| `kernels/rocm/mmq/` | `ds41-gufo-extension` `eb8ece9`, itself from llama.cpp `5c0e946` (`ggml/src/ggml-cuda/`, MIT) | quantized matvec and MMQ kernels; symbols renamed `hy4_*`; see [VENDOR.md](kernels/rocm/mmq/VENDOR.md) |
| `kernels/rocm/hy4_f16_mm.hip.cpp` | `ds41-gufo-extension` `eb8ece9` (`kernels/rocm/ds41_dense_f16.hip.cpp`) | the dense FP16 matrix-core kernel and its Q6_K/Q8_0 formats; Q4_K, row guards and the routed kernel are this repository's |
| `kernels/rocm/hy4_gemv.hip.cpp`, `kernels/rocm/hy4_gemv_tasks.hpp` | `ds41-gufo-extension` `237ba69` (`kernels/rocm/ds41_gemv.hip.cpp`, `ds41_gemv_tasks.hpp`) | the FP32-activation decode GEMVs for Q8_0, Q4_K and Q6_K; launches take a stream and the caller's split scratch, Q5_K and the unused task decoders are dropped; `expert_gemv_kernel` is the lane kernel over routed slots (new here) |
| `runtime/tokenizer.{hpp,cpp}` | `ds41-gufo-extension` `eb8ece9` (`runtime/tokenizer.cpp`) | the DeepSeek3 pretokenizer, which Hy4 declares |
| `runtime/hy4_vision_gemm.{hpp,cpp}` | Gufo `0a473ebf` (`src/models/embedding_gemma2/hip/blas.cpp`) | the hipBLASLt candidate timing; tuned once per shape at load and checked per row count, with a bias epilogue and accumulation (E61) |
| `image_cache.{hpp,cpp}` | Gufo `0a473ebf` (`src/families/qwen/vision/image_cache.cpp`) | the prepared-image cache; keeps the pixel digest too (E62) |
| `image_resize.{hpp,cpp}` | `gemma4-gufo-extension` `52af868` (`vision/resize.cpp`), from Gufo `83b764b4` (`src/core/image.cpp`) | antialiased bicubic RGB8 resize |
| `FusedAttention` in `kernels/rocm/hy4_vision.hip.cpp` | `gemma4-gufo-extension` `52af868` (`vision/encoder.hip`) | scale folded into Q, FP32 output |
| `ContentSpan` and the request-text spans in `chat_template.cpp`, the span check in `Tokenizer::EncodeWithSpecials` | Gufo `0a473ebf` (`src/models/deepseek_v4_flash/chat_template.cpp`: `ContentSpan`, `MarkContent`, `SplitControlTokens`) | a control-token spelling that overlaps request text stays text; Hy4 records the spans through a writer and matches every special token, not a fixed list |
| `stable_prefix_bytes` in `RenderChat` and its use in `PreparePrompt` and `PrepareImagePrompt` | Gufo `0a473ebf` (`src/families/qwen/text/chat_template.cpp`, `src/families/qwen/vision/prompt.cpp`) | the renderer reports what a following user turn keeps and the prompt is tokenized once; Hy4 cuts after a special token and encodes the shorter side |

The model's reference implementations, which nothing here copies:

- llama.cpp PR #32517 (`cf70454a6`): the text model on the same GGUF, the
  correctness oracle ([docs/QUALITY.md](docs/QUALITY.md)).
- Transformers `hy_v4_vl`: the vision tower and merger, transcribed into a
  numpy reference for the vision checks.
- Tencent's DFly2 release: the drafter's configuration and weights.

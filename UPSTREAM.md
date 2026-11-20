# Upstream

Code this extension copies from elsewhere, and where it came from. Copies are
adapted here, not shared; changes to them are this repository's own.

| files | origin | notes |
| --- | --- | --- |
| `kernels/rocm/mmq/` | `ds41-gufo-extension` `eb8ece9`, itself from llama.cpp `5c0e946` (`ggml/src/ggml-cuda/`, MIT) | quantized matvec and MMQ kernels; symbols renamed `hy4_*`; see [VENDOR.md](kernels/rocm/mmq/VENDOR.md) |
| `kernels/rocm/hy4_f16_mm.hip.cpp` | `ds41-gufo-extension` `eb8ece9` (`kernels/rocm/ds41_dense_f16.hip.cpp`) | the dense FP16 matrix-core kernel and its Q6_K/Q8_0 formats; Q4_K, row guards and the routed kernel are this repository's |
| `runtime/tokenizer.{hpp,cpp}` | `ds41-gufo-extension` `eb8ece9` (`runtime/tokenizer.cpp`) | the DeepSeek3 pretokenizer, which Hy4 declares |
| `image_resize.{hpp,cpp}` | `gemma4-gufo-extension` `52af868` (`vision/resize.cpp`), from Gufo `83b764b4` (`src/core/image.cpp`) | antialiased bicubic RGB8 resize |
| `FusedAttention` in `kernels/rocm/hy4_vision.hip.cpp` | `gemma4-gufo-extension` `52af868` (`vision/encoder.hip`) | scale folded into Q, FP32 output |

The model's reference implementations, which nothing here copies:

- llama.cpp PR #32517 (`cf70454a6`): the text model on the same GGUF, the
  correctness oracle ([docs/QUALITY.md](docs/QUALITY.md)).
- Transformers `hy_v4_vl`: the vision tower and merger, transcribed into a
  numpy reference for the vision checks.
- Tencent's DFly2 release: the drafter's configuration and weights.

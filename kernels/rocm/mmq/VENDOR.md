# HIP quantized matrix kernels

These model-private kernels derive from llama.cpp commit
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT, The ggml authors),
originally under `ggml/src/ggml-cuda/`.

Gufo builds this directory with HIP for AMD `gfx1151`. Device headers use
`.hip.hpp`; translation units use `.hip.cpp`. The adapter calls HIP/hipBLAS
directly and uses `hy4_ggml_hip_*` symbols to remain separate from Qwen's
matrix backend. There is no CUDA build or CUDA-to-HIP API alias layer.

`hy4_mmq.h` exposes the raw-pointer C ABI. `hy4_ggml_stubs.h` supplies the
small ggml type and allocation interface needed by the templates. Quantization
layouts, tables and active AMD arithmetic remain from the imported kernels.
Unused NVIDIA/MUSA preprocessing paths, ggml graph support and full ggml
tensor entry points are removed; the V4.1 decoder owns its HIP graph
execution.

This copy comes from `ds41-gufo-extension` at `eb8ece9`
([UPSTREAM.md](../../../UPSTREAM.md)). Hy4 keeps only what its engine calls:
`hy4_mmq_dense_typed` (mmq) and the `hy4_mmq_moe_vec*_typed` matvecs (mmvq),
for Q8_0, Q4_K and Q6_K, the only quantized types in Hy4's GGUFs. DS4.1's
other entry points, its routed-expert tiling, the `mmid` expert grouping, the
fused DS4 prefill path and its D2R stubs, and the tiles for the V4.1 types are
removed. The vendored kernel headers stay as imported. When importing upstream changes,
retain the HIP implementation and check the result with the KL gate in
[QUALITY.md](../../../docs/QUALITY.md).

Local additions: `hy4_mmvq_moe_shared` (end of `mmvq.hip.cpp`) runs a
multi-token pass's routed experts in one launch with the slots ordered by
expert; each slot repeats `mul_mat_vec_q<type, 1>` or
`mul_mat_vec_q_moe<type, 2>` exactly, so results equal one-token launches
bitwise (`kernels_test` checks it).

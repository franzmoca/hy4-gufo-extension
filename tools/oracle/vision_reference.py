#!/usr/bin/env python3
"""Reference rows of Hy4 Flash's vision tower and merger for one image.

Runs Tencent's Transformers implementation (huggingface/transformers PR #50903:
HYV4VLImageProcessor and HYV4VLVisionModel, which HYV4Flash builds its vision
tower from) in float64 on the CPU, with the weights of the mmproj GGUF. The
difference from `hy4_vision_probe` is then the engine's arithmetic alone.

It also writes the processor's resized RGB8 image as a binary PPM, which
`hy4_vision_probe` encodes, so both start from the same pixels; comparing that
image with the engine's own resize checks the preprocessing separately.

Environment: the one of reference.py (torch, transformers at PR #50903, gguf-py
from llama.cpp PR #32517), plus Pillow. CONFIG_DIR holds tencent/Hy4-flash's
config.json and processor_config.json.

  vision_reference.py MMPROJ.gguf CONFIG_DIR IMAGE --rows OUT.f32 --ppm OUT.ppm
  vision_reference.py MMPROJ.gguf CONFIG_DIR IMAGE --ppm OUT.ppm --pixels-only

The float64 tower holds a [16][patches][patches] score matrix per layer, which
limits it to images of a few thousand patches; --pixels-only skips it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig
from transformers.models.hy_v4_vl.image_processing_pil_hy_v4_vl import HYV4VLImageProcessorPil
from transformers.models.hy_v4_vl.modeling_hy_v4_vl import HYV4VLVisionModel


def gguf_tensors(path: str) -> dict[str, np.ndarray]:
    """The mmproj tensors in float64, in numpy (row-major) shape."""
    out = {}
    for t in gguf.GGUFReader(path).tensors:
        a = np.asarray(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise SystemExit(f"{t.name}: unexpected type {t.tensor_type.name}")
        out[t.name] = a.reshape([int(d) for d in reversed(t.shape)]).astype(np.float64)
    return out


def state_dict(g: dict[str, np.ndarray], layers: int) -> dict[str, torch.Tensor]:
    """The GGUF's tensors under HYV4VLVisionModel's names."""
    sd = {
        "embeddings.patch_embedding.weight": g["v.patch_embd.weight"],
        "embeddings.patch_embedding.bias": g["v.patch_embd.bias"],
        # The checkpoint's first row is an unused class-token position, which
        # the GGUF drops; interpolate_pos_encoding reads rows 1 onward.
        "embeddings.position_embedding.weight": np.concatenate(
            [np.zeros((1, g["v.position_embd.weight"].shape[1])), g["v.position_embd.weight"]]
        ),
        "patch_merger.before_rms.weight": g["mm.pre_norm.weight"],
        "patch_merger.proj_conv.weight": g["mm.0.weight"],
        "patch_merger.proj_conv.bias": g["mm.0.bias"],
        "patch_merger.proj_out.weight": g["mm.2.weight"],
        "patch_merger.proj_out.bias": g["mm.2.bias"],
        "patch_merger.image_newline": g["v.image_newline"],
    }
    names = {
        "ln1": "layer_norm1", "ln2": "layer_norm2",
        "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj", "attn_out": "self_attn.o_proj",
        "ffn_up": "mlp.fc1", "ffn_down": "mlp.fc2",
    }
    for i in range(layers):
        for src, dst in names.items():
            for kind in ("weight", "bias"):
                sd[f"layers.{i}.{dst}.{kind}"] = g[f"v.blk.{i}.{src}.{kind}"]
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mmproj")
    ap.add_argument("config_dir")
    ap.add_argument("image")
    ap.add_argument("--rows")
    ap.add_argument("--ppm", required=True)
    ap.add_argument("--pixels-only", action="store_true")
    args = ap.parse_args()
    if not args.pixels_only and not args.rows:
        ap.error("--rows is required unless --pixels-only")

    config = AutoConfig.from_pretrained(args.config_dir).vision_config
    config._attn_implementation = "eager"
    processor_config = json.loads(Path(args.config_dir, "processor_config.json").read_text())
    processor = HYV4VLImageProcessorPil(**processor_config["image_processor"])

    image = Image.open(args.image).convert("RGB")
    batch = processor(images=[image], return_tensors="pt")
    pixels = batch["pixel_values"].to(torch.float64)
    grid = batch["image_grid_thw"]
    _, gh, gw = (int(v) for v in grid[0])
    p = config.patch_size

    # The processed image back in RGB8: patches are row-major, each (c, y, x).
    mean = np.array(processor_config["image_processor"]["image_mean"])
    std = np.array(processor_config["image_processor"]["image_std"])
    planes = pixels.numpy().reshape(gh, gw, 3, p, p).transpose(2, 0, 3, 1, 4).reshape(3, gh * p, gw * p)
    rgb = (planes.transpose(1, 2, 0) * std + mean) / processor_config["image_processor"]["rescale_factor"]
    rgb8 = np.rint(rgb)
    if np.abs(rgb - rgb8).max() > 1e-3:
        raise SystemExit("processed pixels are not whole RGB8 values")
    with open(args.ppm, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (gw * p, gh * p))
        f.write(rgb8.astype(np.uint8).tobytes())

    if args.pixels_only:
        print(f"{image.width}x{image.height} -> {gw * p}x{gh * p}")
        return
    model = HYV4VLVisionModel(config).to(torch.float64).eval()
    missing, unexpected = model.load_state_dict(
        state_dict(gguf_tensors(args.mmproj), config.num_hidden_layers), strict=True
    )
    with torch.no_grad():
        rows = model(pixel_values=pixels, grid_thw=grid).pooler_output[0]
    rows.to(torch.float32).numpy().tofile(args.rows)
    print(f"{image.width}x{image.height} -> {gw * p}x{gh * p}: {rows.shape[0]} rows")


if __name__ == "__main__":
    main()

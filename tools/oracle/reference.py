#!/usr/bin/env python3
"""Exact-arithmetic reference over the shipped GGUF weights.

Runs Tencent's Transformers implementation of Hy4 Flash (huggingface/transformers
PR #50903, linked from the tencent/Hy4-flash model card) in FP32 on the CPU, with
every weight dequantized from the GGUF. The difference from this engine is then
the engine's arithmetic alone, with quantization taken out. The model is built on
the meta device and run one decoder layer at a time: each layer's weights are
loaded, used for the whole chunk and freed, routed experts are dequantized only
when routed to, and Engram rows are read from the memory-mapped FP8 table.

It writes a llama-perplexity --kl-divergence-base file over the same tokens as an
existing one (the llama.cpp oracle's), so `hy4_eval_probe --kl` and `--kl-decode`
score the engine's prompt and decode paths against it.

Environment (Python 3.12):
  torch (CPU), numpy,
  transformers at b865850d05326cf20e306b31f14b0027de802da3 (PR #50903),
  gguf-py from llama.cpp cf70454a6 (PR #32517, tools/oracle/llama-ref.nix),
  the tencent/Hy4-flash config.json in CONFIG_DIR.

  reference.py MODEL.gguf CONFIG_DIR --base BASE.kld --out OUT.kld [--chunks N]
  reference.py MODEL.gguf CONFIG_DIR --tokens 1,2,3 --logits OUT.f32

The GGUF stores some tensors fused or reshaped by llama.cpp's converter
(conversion/hy_v4_flash.py); load_layer undoes each of those exactly.
"""
from __future__ import annotations

import argparse
import glob
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gguf
from transformers import AutoConfig
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M


class Weights:
    """Tensors and metadata of a (split) GGUF, dequantized to FP32 on request."""

    def __init__(self, first_shard: str):
        pattern = first_shard.replace("-00001-of-", "-*-of-")
        self.tensors = {}
        self.fields = {}
        for path in sorted(glob.glob(pattern)) or [first_shard]:
            reader = gguf.GGUFReader(path)
            self.tensors.update({t.name: t for t in reader.tensors})
            self.fields.update(reader.fields)

    def has(self, name: str) -> bool:
        return name in self.tensors

    def meta(self, key: str):
        return self.fields[key].contents()

    def get(self, name: str, index=None) -> torch.Tensor:
        t = self.tensors[name]
        data = np.asarray(t.data if index is None else t.data[index])
        if t.tensor_type in (gguf.GGMLQuantizationType.F32,):
            values = data.astype(np.float32)
        else:
            values = gguf.quants.dequantize(data, t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))

    def rows(self, name: str, rows: np.ndarray) -> torch.Tensor:
        """Dequantized rows of a 2-D tensor (ne1 = row count)."""
        t = self.tensors[name]
        values = gguf.quants.dequantize(np.asarray(t.data[rows]), t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


class LazyExperts(nn.Module):
    """HYV4FlashExperts with each routed expert dequantized when it is hit."""

    def __init__(self, w: Weights, il: int, swiglu_limit: float):
        super().__init__()
        self.w, self.il = w, il
        self.swiglu_limit = swiglu_limit
        self.num_experts = w.tensors[f"blk.{il}.ffn_gate_exps.weight"].data.shape[0]

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        return M.HYV4FlashExperts._apply_gate(self, gate_up)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        b = f"blk.{self.il}."
        for expert_idx in hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            gate_up = torch.cat([self.w.get(b + "ffn_gate_exps.weight", e), self.w.get(b + "ffn_up_exps.weight", e)])
            down = self.w.get(b + "ffn_down_exps.weight", e)
            top_k_pos, token_idx = torch.where(mask[e])
            current = self._apply_gate(F.linear(hidden_states[token_idx], gate_up))
            current = F.linear(current, down) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class LazyEngramTable(nn.Module):
    """HYV4FlashEngramEmbedding over the GGUF's FP8 table: one layer's slice, rows read on use."""

    def __init__(self, w: Weights, first_row: int, head_vocab_sizes: list[int]):
        super().__init__()
        self.table = w.tensors["per_layer_token_embd.weight"].data  # [rows, head_dim] uint8 (E4M3)
        self.first_row = first_row
        self.offsets = torch.tensor([0] + head_vocab_sizes[:-1]).cumsum(0)
        self.e4m3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        indices = hash_ids.clamp_min(0) + self.offsets + self.first_row
        flat = indices.reshape(-1).numpy()
        order = np.argsort(flat, kind="stable")
        raw = np.empty((flat.size, self.table.shape[1]), np.uint8)
        raw[order] = self.table[flat[order]]
        rows = self.e4m3[torch.from_numpy(raw).long()]
        embeddings = rows.view(*hash_ids.shape, -1)
        return embeddings.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)


def set_param(module: nn.Module, name: str, value: torch.Tensor):
    *path, leaf = name.split(".")
    for p in path:
        module = getattr(module, p)
    old = getattr(module, leaf)
    if tuple(old.shape) != tuple(value.shape):
        raise ValueError(f"{name}: GGUF gives {tuple(value.shape)}, the model wants {tuple(old.shape)}")
    if isinstance(old, nn.Parameter):
        setattr(module, leaf, nn.Parameter(value.contiguous(), requires_grad=False))
    else:
        setattr(module, leaf, value.contiguous())


def load_layer(layer: M.HYV4FlashDecoderLayer, w: Weights, il: int, cfg, engram_rows: dict[int, int]):
    b = f"blk.{il}."
    g = lambda n: w.get(b + n)  # noqa: E731
    a = layer.self_attn
    p = {
        "self_attn.sinks": g("attn_sinks.weight"),
        "self_attn.q_a_proj.weight": g("attn_q_a.weight"),
        "self_attn.q_a_layernorm.weight": g("attn_q_a_norm.weight"),
        "self_attn.q_b_proj.weight": g("attn_q_b.weight"),
        "self_attn.kv_a_proj_with_mqa.weight": g("attn_kv_a_mqa.weight"),
        "self_attn.kv_a_layernorm.weight": g("attn_kv_a_norm.weight"),
        "self_attn.gate_proj.weight": g("attn_gate.weight"),
        "self_attn.o_proj.weight": g("attn_output.weight"),
        "self_attn.post_proj_norm.weight": g("post_attention_norm.weight"),
        "input_layernorm.weight": g("attn_norm.weight"),
        "post_attention_layernorm.weight": g("ffn_norm.weight"),
    }
    # kv_b_proj was split per head: k_b transposed to [kv_lora, nope], v_b as is.
    k_b = g("attn_k_b.weight").transpose(1, 2)
    v_b = g("attn_v_b.weight")
    p["self_attn.kv_b_proj.weight"] = torch.cat([k_b, v_b], dim=1).reshape(-1, k_b.shape[-1])
    for n in "attn_hc", "ffn_hc":
        src = "hc_attn_" if n == "attn_hc" else "hc_ffn_"
        for leaf in "fn", "base", "scale":
            p[f"{n}.{leaf}"] = g(src + leaf + ".weight")
    if getattr(a, "compressed_indexer", None) is not None:
        ci = "self_attn.compressed_indexer."
        p.update({
            ci + "intra_group_ape": g("indexer_compressor_ape.weight"),
            ci + "query_projection.weight": g("indexer.attn_q_b.weight"),
            ci + "head_weight_projection.weight": g("indexer.proj.weight"),
            ci + "token_key_projection.weight": g("indexer.attn_k.weight"),
            ci + "token_key_norm.weight": g("indexer.k_norm.weight"),
            ci + "key_value_projection.weight": g("indexer_compressor_value.weight"),
            ci + "key_gate_projection.weight": g("indexer_compressor_gate.weight"),
            ci + "key_norm.weight": g("indexer_compressor_norm.weight"),
        })
    if isinstance(layer.mlp, M.HYV4FlashMoE):
        p.update({
            "mlp.gate.weight": g("ffn_gate_inp.weight"),
            "mlp.gate.e_score_correction_bias": g("exp_probs_b.bias"),
            "mlp.shared_experts.gate_proj.weight": g("ffn_gate_shexp.weight"),
            "mlp.shared_experts.up_proj.weight": g("ffn_up_shexp.weight"),
            "mlp.shared_experts.down_proj.weight": g("ffn_down_shexp.weight"),
            "mlp.pre_expert_proj.weight": g("ffn_routed_down.weight"),
            "mlp.post_expert_proj.weight": g("ffn_routed_up.weight"),
            "mlp.combine_output_norm.weight": g("post_ffw_norm.weight"),
        })
    else:
        p.update({
            "mlp.gate_proj.weight": g("ffn_gate.weight"),
            "mlp.up_proj.weight": g("ffn_up.weight"),
            "mlp.down_proj.weight": g("ffn_down.weight"),
            "post_mlp_layernorm.weight": g("post_ffw_norm.weight"),
        })
    e = layer.engram_layer
    if e is not None:
        hc = cfg.hc_mult
        kv = g("engram_kv.weight")  # key_projs.0..hc-1 then value_proj, stacked by rows
        kv_b = g("engram_kv_b.weight")
        rows = kv.shape[0] // (hc + 1)
        for c in range(hc):
            p[f"engram_layer.key_projs.{c}.weight"] = kv[c * rows:(c + 1) * rows]
            p[f"engram_layer.key_projs.{c}.bias"] = kv_b[c]
        p["engram_layer.value_proj.weight"] = kv[hc * rows:]
        p["engram_layer.value_proj.bias"] = kv_b[hc]
        # engram_gate folds norm_key * norm_query; the gate only uses their product.
        gate = g("engram_gate.weight")
        for c in range(hc):
            p[f"engram_layer.norm_key.{c}.weight"] = gate[c]
            p[f"engram_layer.norm_query.{c}.weight"] = torch.ones_like(gate[c])
        norms = g("engram_lookup_norm.weight")
        for h in range(norms.shape[0]):
            p[f"engram_layer.lookup_per_head_norms.{h}.weight"] = norms[h]
        scales = g("engram_hc_scale.weight")
        for c in range(hc):
            p[f"engram_layer.unified_per_hc_scales.{c}"] = scales[c]
        p["engram_layer.unified_pre_gamma"] = g("engram_pre_gamma.weight")
    # Everything but the experts and the Engram table, which stay lazy.
    for name, value in p.items():
        set_param(layer, name, value)
    if getattr(a, "compressed_indexer", None) is not None:
        a.compressed_indexer.rotary_emb = M.HYV4FlashRotaryEmbedding(cfg)
    if isinstance(layer.mlp, M.HYV4FlashMoE):
        layer.mlp.experts = LazyExperts(w, il, cfg.swiglu_limit)
    if e is not None:
        e.multi_head_embedding = LazyEngramTable(w, engram_rows[il], e.config_head_vocab_sizes)
        token_map = np.asarray(w.meta("hy_v4_flash.engram.token_map"), dtype=np.int32)
        e.lookup_table = torch.from_numpy(token_map)
        e.head_vocab_sizes = torch.tensor(e.config_head_vocab_sizes)
        # The GGUF's multipliers, in Engram layer order, n-gram sizes per layer.
        mult = np.asarray(w.meta("hy_v4_flash.ple.layer_multipliers"), dtype=np.int64)
        k = sorted(engram_rows).index(il)
        n = cfg.engram_max_ngram_size
        e.hash_multipliers = torch.from_numpy(mult[k * n:(k + 1) * n].copy())
    leftover = [n for n, t in layer.named_parameters() if t.is_meta] + [n for n, t in layer.named_buffers() if t.is_meta]
    if leftover:
        raise ValueError(f"layer {il}: weights not loaded: {leftover}")


def engram_first_rows(w: Weights, cfg) -> dict[int, int]:
    """First row of each Engram layer's slice in the shared table, by 0-based layer."""
    layers = [int(i) for i in w.meta("hy_v4_flash.ple.layers")]
    out, base = {}, 0
    for il in layers:
        out[il] = base
        base += sum(sum(order) for order in cfg.engram_layer_head_vocab_sizes[str(il + 1)])
    rows = w.tensors["per_layer_token_embd.weight"].data.shape[0]
    if base != rows:
        raise ValueError(f"Engram tables need {base} rows, the GGUF has {rows}")
    return out


@torch.no_grad()
def forward(w: Weights, cfg, tokens: list[int], log=print) -> torch.Tensor:
    """Logits [len(tokens), vocab] of one sequence from position 0."""
    with torch.device("meta"):
        model = M.HYV4FlashTextModel(cfg)
    model.rotary_emb = M.HYV4FlashRotaryEmbedding(config=cfg)
    head = model.hc_head
    for leaf in "hc_fn", "hc_base", "hc_scale":
        set_param(head, leaf, w.get(f"output_{leaf}.weight"))
    set_param(model.norm, "weight", w.get("output_norm.weight"))
    ids = torch.tensor([tokens])
    n = len(tokens)
    x = w.rows("token_embd.weight", np.asarray(tokens)).unsqueeze(0)
    position_ids = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
    text_position_ids = position_ids[0]
    mask_kwargs = dict(config=cfg, inputs_embeds=x, attention_mask=None, cache=None,
                       position_ids=text_position_ids, allow_is_causal_skip=False)
    masks = {"indexed_attention": M.create_causal_mask(**mask_kwargs),
             "sliding_attention": M.create_sliding_window_causal_mask(**mask_kwargs)}
    ropes = {t: model.rotary_emb(x, position_ids=position_ids, layer_type=t) for t in model.rotary_emb.layer_types}
    ngram_tokens = model.build_ngram_tokens(ids, None)
    engram_mask = ids != cfg.engram_pad_id
    engram_rows = engram_first_rows(w, cfg)
    h = x.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    for il, layer in enumerate(model.layers):
        t0 = time.time()
        load_layer(layer, w, il, cfg, engram_rows)
        h = layer(h, attention_mask=masks[layer.attention_type], position_embeddings=ropes[layer.attention_type],
                  position_ids=position_ids, cache=None, use_cache=False, indexer_position_ids=position_ids,
                  ngram_tokens=ngram_tokens, engram_mask=engram_mask)
        model.layers[il] = None
        log(f"layer {il:2d} {layer.attention_type:18s} {time.time() - t0:6.1f} s")
    out = model.norm(head(h))[0]
    logits = torch.empty(n, w.tensors["output.weight"].data.shape[0])
    vocab = logits.shape[1]
    step = 8192
    for v0 in range(0, vocab, step):
        rows = np.arange(v0, min(vocab, v0 + step))
        logits[:, v0:v0 + len(rows)] = out @ w.rows("output.weight", rows).T
    return logits


def write_kld_records(f, logits: torch.Tensor, first: int):
    """llama-perplexity base records for positions first .. n-2, encoded as its log_softmax does: the
    logits from max - 16 up as uint16 steps, after the scale and the minimum log-probability."""
    vocab = logits.shape[1]
    pad = np.zeros(vocab % 2, "<u2")
    for at in range(first, logits.shape[0] - 1):
        row = logits[at].double().numpy()
        top = row.max()
        low = max(row.min(), top - 16)
        lse = np.log(np.exp(row - top).sum())
        scale = (top - low) / 65535
        q = np.where(row > low, np.rint((row - low) / scale), 0).astype("<u2")
        f.write(struct.pack("<ff", scale, low - top - lse))
        f.write(q.tobytes())
        f.write(pad.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("config_dir")
    ap.add_argument("--base", help="llama-perplexity KL base file whose tokens to score")
    ap.add_argument("--out", help="KL base file to write")
    ap.add_argument("--chunks", type=int, default=0, help="score the first N chunks (default: all)")
    ap.add_argument("--tokens", help="comma-separated token ids: write all their logits to --logits")
    ap.add_argument("--logits", help="raw float32 [tokens, vocab] output for --tokens")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = AutoConfig.from_pretrained(args.config_dir).text_config
    cfg._attn_implementation = "eager"
    w = Weights(args.model)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    if args.tokens:
        tokens = [int(t) for t in args.tokens.split(",") if t]
        forward(w, cfg, tokens, log).numpy().astype("<f4").tofile(args.logits)
        return
    with open(args.base, "rb") as f:
        magic, n_ctx, n_vocab, n_chunk = struct.unpack("<8sIii", f.read(20))
        if magic != b"_logits_":
            sys.exit("not a llama-perplexity KL base file")
        toks = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype="<i4").reshape(n_chunk, n_ctx)
    chunks = min(n_chunk, args.chunks or n_chunk)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<8sIii", b"_logits_", n_ctx, n_vocab, chunks))
        f.write(toks[:chunks].tobytes())
        for ch in range(chunks):
            t0 = time.time()
            logits = forward(w, cfg, toks[ch].tolist(), log)
            if logits.shape[1] != n_vocab:
                sys.exit(f"vocab {logits.shape[1]} != base file's {n_vocab}")
            write_kld_records(f, logits, n_ctx // 2)
            log(f"chunk {ch}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

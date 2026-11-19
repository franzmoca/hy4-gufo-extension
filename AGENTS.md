# AGENTS.md

This repository is a Gufo source extension: the Hy4 Flash engine (provider
`hy4flash`, GGUF architecture `hy_v4_flash`), its kernels, tools and
documentation. The production target is Linux x86-64 AMD Strix Halo
(`gfx1151`). [docs/PLAN.md](docs/PLAN.md) holds the plan and the model facts.

## Build

```sh
nix build                    # Gufo with this extension: ./result/bin/gufo
nix flake update gufo        # move the pinned upstream Gufo (flake.lock)
```

The flake builds tracked files only; stage new files before `nix build`.
Development builds and tests run from a Gufo checkout at the pinned revision,
with this repository as an extension directory
(`-DGUFO_EXTENSIONS='*' -DGUFO_EXTENSION_DIRS=/path/to/hy4-gufo-extension`).

## Extension API

Code includes Gufo only through `<gufo/ext/...>`. `extension.json` declares the
minimum API version. When the extension needs something the API lacks, change
upstream Gufo first: one small PR with its `ext-api` change record under
`include/gufo/ext/changes/`, described on Gufo's own terms.

## Reuse from DeepSeek V4.1

Hy4 Flash shares much of its design with DeepSeek V4.1 Flash (hyper-connections,
gated MLA with sinks, Engram). Code taken from `ds41-gufo-extension` is copied
and adapted here, not shared; record its origin in `UPSTREAM.md` when the first
file arrives.

## Correctness

The oracle is llama.cpp PR #32517 on the same GGUF: per-layer hidden states and
logits for fixed prompts. DFly2 and vision have no llama.cpp implementation;
check them against BF16 PyTorch references.

## Formatting

`.clang-format` applies to new and changed C++ lines. Check the files you
touched with `clang-format --dry-run`.

## Development

- Follow Conventional Commits with a single-line message, scope `hy4flash`.
- Committed artifacts contain no usernames, host names or private paths.
- Use `gh` for GitHub operations after checking `gh auth status`.

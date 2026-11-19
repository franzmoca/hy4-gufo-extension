# Hy4 Flash for Gufo

An experimental Gufo source extension for Tencent's
[Hy4 Flash](https://huggingface.co/tencent/Hy4-flash) (`hy_v4_flash`) on
Linux x86-64 / Strix Halo `gfx1151`.

**Status:** in progress. This revision registers provider metadata and the
GGUF probe only; it cannot run the model yet. See [docs/PLAN.md](docs/PLAN.md).

Maintainer: Francesco Moca ([@franzmoca](https://github.com/franzmoca)).

## Model files

The target build is the
[Xylolabs Q4_K_M](https://huggingface.co/Xylolabs/Hy4-flash-GGUF): a 97.8 GB
trunk kept resident, plus the 63.4 GB FP8 Engram shard read from disk. The
DFly2 drafter comes from Tencent's checkpoint
(`model-00148-of-00148.safetensors`).

## Build with Gufo

```sh
nix build                    # Gufo with this extension: ./result/bin/gufo
./result/bin/gufo diagnose --section extensions
```

Without Nix, from a Gufo checkout at the revision pinned in `flake.lock`:

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/hy4-gufo-extension
cmake --build --preset release --parallel 4
```

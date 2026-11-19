# llama.cpp PR #32517 (hy_v4_flash), built the way Gufo builds its llama.cpp
# reference: ROCm 10, gfx1151, HIP UMA.
let
  gufo = builtins.getFlake "github:gufo-org/gufo/0a473ebfca668dbf9e41265b8082239a1935d719";
  lib = gufo.inputs.nixpkgs.lib;
  # Gufo's gfx1151 ROCm 10 set, which gufo.cachix.org and Gufo builds hold;
  # the reference package otherwise takes the generic all-target libraries.
  rocm = (import gufo.inputs.nixpkgs-rocm10 { system = "x86_64-linux"; }).rocmPackages_10.gfx1151;
  base = gufo.packages.x86_64-linux.llama-cpp-reference.override { rocmPackages = rocm; };
  rev = "cf70454a6b9bbd197ced811ac4b4ccc88a244794";
in
base.overrideAttrs (old: {
  pname = "llama-cpp-hy4";
  version = "pr32517";
  src = builtins.fetchGit {
    url = "https://github.com/ggml-org/llama.cpp";
    ref = "refs/pull/32517/head";
    inherit rev;
  };
  # nixpkgs backports llama.cpp #30034, which the PR already contains.
  patches = [ ];
  # Its test phase needs the network and Python modules the sandbox lacks.
  doCheck = false;
  # The PR predates the tools/ui web UI that nixpkgs builds with npm.
  npmDeps = null;
  nativeBuildInputs = builtins.filter
    (p: !(lib.hasInfix "npm-config-hook" (p.name or ""))) old.nativeBuildInputs;
  # Keep Gufo's HIP flags, drop the npm build in front of them.
  preConfigure = lib.last (lib.splitString "popd\n" old.preConfigure);
  cmakeFlags =
    builtins.filter (f: !(lib.hasPrefix "-DLLAMA_BUILD_COMMIT" f)) (old.cmakeFlags or [ ])
    ++ [ "-DLLAMA_BUILD_COMMIT:STRING=${builtins.substring 0 8 rev}" ];
})

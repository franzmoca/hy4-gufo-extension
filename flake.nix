{
  description = "Pinned Gufo source-extension package";
  inputs.gufo.url = "github:gufo-org/gufo/abebd1c08e9c57996f963715695e6c1f1410a7da";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
    # Gufo's server command generator, reading this package's provider
    # metadata (draft options included).
    lib.x86_64-linux.mkServe = args: gufo.lib.mkGufoServe (args // {
      system = "x86_64-linux";
      gufo = self.packages.x86_64-linux.default;
    });
  };
}

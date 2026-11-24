// Tokenizes a file with the Hy4 Flash tokenizer and prints the ids, for
// comparison with llama.cpp's llama-tokenize on the same GGUF.
//
//   hy4_tokenize_probe MODEL.gguf FILE [--specials]
//   hy4_tokenize_probe MODEL.gguf --decode ID,ID,...

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <gufo/ext/gguf_reader.hpp>

#include "runtime/tokenizer.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf FILE [--specials]\n", argv[0]);
    return 2;
  }
  std::string error;
  gufo::core::GgufOpenOptions opts;
  opts.allow_unsized_tensors = true;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error, opts);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto tok = gufo::models::hy4_flash::Tokenizer::Load(
      *reader, &error, /*canonicalize_specials=*/false);
  if (!tok) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  if (std::string(argv[2]) == "--decode" && argc > 3) {
    std::string ids = argv[3];
    for (auto& c : ids) c = c == ',' ? ' ' : c;
    std::istringstream in(ids);
    int id = 0;
    std::string text;
    while (in >> id) text += tok->Decode(id);
    std::printf("%s\n", text.c_str());
    return 0;
  }
  std::ifstream f(argv[2], std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  const bool specials = argc > 3 && std::string(argv[3]) == "--specials";
  const auto ids = specials ? tok->EncodeWithSpecials(text) : tok->Encode(text);
  std::string round;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    std::printf(i ? ",%d" : "%d", ids[i]);
    round += tok->Decode(ids[i]);
  }
  std::printf("\n");
  std::fprintf(stderr, "%zu tokens, round trip %s\n", ids.size(), round == text ? "exact" : "DIFFERS");
  return 0;
}

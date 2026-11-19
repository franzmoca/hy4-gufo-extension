// Evaluates token ids with the Hy4 Flash runtime: prints the top logits after
// the prompt and a greedy continuation, for comparison with the llama.cpp
// reference (docs/PLAN.md, correctness oracle).
//
//   hy4_eval_probe MODEL.gguf TOKENS [--gen N] [--ctx N] [--logits FILE]
//
// TOKENS is a comma-separated id list or @FILE with whitespace-separated ids.
// --logits writes the prompt's last-token logits as raw float32.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/hy4_runtime.hpp"

namespace hy4 = gufo::models::hy4_flash;

namespace {

std::vector<std::int32_t> ParseTokens(const std::string& arg) {
  std::string text = arg;
  if (!arg.empty() && arg[0] == '@') {
    std::ifstream f(arg.substr(1));
    std::stringstream ss;
    ss << f.rdbuf();
    text = ss.str();
  }
  std::replace(text.begin(), text.end(), ',', ' ');
  std::vector<std::int32_t> out;
  std::istringstream in(text);
  long v = 0;
  while (in >> v) out.push_back(static_cast<std::int32_t>(v));
  return out;
}

std::int32_t Argmax(const float* x, std::size_t n) {
  return static_cast<std::int32_t>(std::max_element(x, x + n) - x);
}

double Seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf TOKENS [--gen N] [--ctx N] [--logits FILE]\n", argv[0]);
    return 2;
  }
  const std::string model_path = argv[1];
  const std::vector<std::int32_t> prompt = ParseTokens(argv[2]);
  int gen = 16;
  std::uint32_t ctx = 8192;
  std::string logits_path;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--gen") gen = std::atoi(argv[i + 1]);
    else if (a == "--ctx") ctx = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--logits") logits_path = argv[i + 1];
  }
  if (prompt.empty()) {
    std::fprintf(stderr, "no tokens\n");
    return 2;
  }

  std::string error;
  const auto t0 = std::chrono::steady_clock::now();
  auto model = hy4::Model::Load(model_path, &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "loaded in %.1f s\n", Seconds(t0, t1));
  auto session = model->CreateSession(ctx, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<float> logits;
  const auto p0 = std::chrono::steady_clock::now();
  if (!session->Evaluate(prompt, &logits, false, &error)) {
    std::fprintf(stderr, "prompt failed: %s\n", error.c_str());
    return 1;
  }
  const auto p1 = std::chrono::steady_clock::now();
  const std::size_t vocab = model->config().n_vocab;
  std::printf("prompt %zu tokens in %.3f s (%.1f tok/s)\n", prompt.size(), Seconds(p0, p1),
              prompt.size() / Seconds(p0, p1));
  if (!logits_path.empty()) {
    std::ofstream f(logits_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(vocab * sizeof(float)));
  }
  std::vector<std::int32_t> order(vocab);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                    [&](std::int32_t a, std::int32_t b) { return logits[a] > logits[b]; });
  std::printf("top:");
  for (int i = 0; i < 10; ++i) std::printf(" %d:%.3f", order[i], logits[order[i]]);
  std::printf("\n");

  std::vector<std::int32_t> out;
  std::int32_t next = Argmax(logits.data(), vocab);
  const auto g0 = std::chrono::steady_clock::now();
  for (int i = 0; i < gen; ++i) {
    out.push_back(next);
    const std::int32_t tok[1] = {next};
    if (!session->Evaluate(tok, &logits, false, &error)) {
      std::fprintf(stderr, "decode failed: %s\n", error.c_str());
      return 1;
    }
    next = Argmax(logits.data(), vocab);
  }
  const auto g1 = std::chrono::steady_clock::now();
  std::printf("gen:");
  for (const auto t : out) std::printf(" %d", t);
  std::printf("\n");
  if (gen > 0) std::printf("decode %d tokens in %.3f s (%.2f tok/s)\n", gen, Seconds(g0, g1), gen / Seconds(g0, g1));
  return 0;
}

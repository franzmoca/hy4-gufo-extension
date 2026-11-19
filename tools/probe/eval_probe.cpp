// Evaluates token ids with the Hy4 Flash runtime: prints the top logits after
// the prompt and a greedy continuation, for comparison with the llama.cpp
// reference (docs/PLAN.md, correctness oracle).
//
//   hy4_eval_probe MODEL.gguf TOKENS [--gen N] [--ctx N] [--logits FILE]
//   hy4_eval_probe MODEL.gguf TOKENS --ppl N
//
// TOKENS is a comma-separated id list or @FILE with whitespace-separated ids;
// several prompts joined with '+' each run in a fresh session (after the
// first, only their top probabilities are printed).
// --logits writes the prompt's last-token logits as raw float32. --ppl N
// scores like llama-perplexity -c N: chunks of N tokens in fresh sessions,
// the second half of each chunk scored.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <span>
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

// Softmax probabilities of the ten likeliest tokens.
void PrintTop(const std::vector<float>& logits, std::size_t vocab, std::size_t n_prompt) {
  std::vector<std::int32_t> order(vocab);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                    [&](std::int32_t a, std::int32_t b) { return logits[a] > logits[b]; });
  const float mx = logits[order[0]];
  double den = 0.0;
  for (std::size_t i = 0; i < vocab; ++i) den += std::exp(static_cast<double>(logits[i] - mx));
  std::printf("probs[%zu]:", n_prompt);
  for (int i = 0; i < 10; ++i) std::printf(" %d:%.4f", order[i], std::exp(logits[order[i]] - mx) / den);
  std::printf("\n");
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
  std::vector<std::vector<std::int32_t>> extra;
  std::string first_arg = argv[2];
  for (std::size_t at; (at = first_arg.rfind('+')) != std::string::npos;) {
    extra.insert(extra.begin(), ParseTokens(first_arg.substr(at + 1)));
    first_arg.resize(at);
  }
  const std::vector<std::int32_t> prompt = ParseTokens(first_arg);
  int gen = 16;
  std::uint32_t ctx = 8192;
  std::string logits_path;
  std::uint32_t ppl = 0;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--gen") gen = std::atoi(argv[i + 1]);
    else if (a == "--ctx") ctx = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--logits") logits_path = argv[i + 1];
    else if (a == "--ppl") ppl = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
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
  if (ppl > 0) {
    const std::size_t vocab = model->config().n_vocab;
    double nll = 0.0;
    std::size_t count = 0;
    for (std::size_t c0 = 0; c0 + ppl <= prompt.size(); c0 += ppl) {
      auto s = model->CreateSession(ppl, &error);
      if (!s) {
        std::fprintf(stderr, "session failed: %s\n", error.c_str());
        return 1;
      }
      double chunk_nll = 0.0;
      for (std::size_t i = 0; i < ppl; i += hy4::Model::kMaxChunk) {
        const std::size_t n = std::min<std::size_t>(hy4::Model::kMaxChunk, ppl - i);
        const std::span<const std::int32_t> part(prompt.data() + c0 + i, n);
        const bool need = i + n > ppl / 2;
        if (!s->Evaluate(part, need ? &logits : nullptr, true, &error)) {
          std::fprintf(stderr, "ppl failed: %s\n", error.c_str());
          return 1;
        }
        if (!need) continue;
        for (std::size_t j = 0; j < n; ++j) {
          const std::size_t at = i + j;  // logits predict at + 1
          if (at < ppl / 2 || at + 1 >= ppl) continue;
          const float* row = logits.data() + j * vocab;
          const float mx = *std::max_element(row, row + vocab);
          double den = 0.0;
          for (std::size_t v = 0; v < vocab; ++v) den += std::exp(static_cast<double>(row[v] - mx));
          const double lp = row[prompt[c0 + at + 1]] - mx - std::log(den);
          chunk_nll -= lp;
          nll -= lp;
          ++count;
        }
      }
      std::printf("chunk %zu: ppl %.4f\n", c0 / ppl, std::exp(chunk_nll / (ppl / 2 - 1)));
      std::fflush(stdout);
    }
    std::printf("ppl %.4f over %zu tokens\n", std::exp(nll / static_cast<double>(count)), count);
    return 0;
  }
  for (const auto& p : extra) {
    auto s = model->CreateSession(ctx, &error);
    if (!s || !s->Evaluate(p, &logits, false, &error)) {
      std::fprintf(stderr, "extra prompt failed: %s\n", error.c_str());
      return 1;
    }
    PrintTop(logits, model->config().n_vocab, p.size());
  }
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
  PrintTop(logits, vocab, prompt.size());

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

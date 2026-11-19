// Evaluates token ids with the Hy4 Flash runtime: prints the top logits after
// the prompt and a greedy continuation, for comparison with the llama.cpp
// reference (docs/PLAN.md, correctness oracle).
//
//   hy4_eval_probe MODEL.gguf TOKENS [--gen N] [--ctx N] [--logits FILE]
//   hy4_eval_probe MODEL.gguf TOKENS --ppl N
//   hy4_eval_probe MODEL.gguf - --kl BASE
//
// TOKENS is a comma-separated id list or @FILE with whitespace-separated ids;
// several prompts joined with '+' each run in a fresh session (after the
// first, only their top probabilities are printed).
// --logits writes the prompt's last-token logits as raw float32. --ppl N
// scores like llama-perplexity -c N: chunks of N tokens in fresh sessions,
// the second half of each chunk scored. --kl BASE reads a llama-perplexity
// --kl-divergence-base file (its tokens and log-probabilities) and reports
// the KL divergence from it, the same-top-token rate and the perplexity.

#include <algorithm>
#include <chrono>
#include <cstring>
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
  std::string kl_path;
  std::uint32_t passcheck = 0;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--gen") gen = std::atoi(argv[i + 1]);
    else if (a == "--ctx") ctx = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--logits") logits_path = argv[i + 1];
    else if (a == "--ppl") ppl = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--kl") kl_path = argv[i + 1];
    else if (a == "--passcheck") passcheck = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
  }
  if (prompt.empty() && kl_path.empty()) {
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
  if (passcheck) {
    // Greedy tokens one per pass, then the same tokens `passcheck` per pass on
    // a fresh session: how far a multi-token (verify) pass moves the logits.
    const std::size_t vocab = model->config().n_vocab;
    const int steps = gen > 0 ? gen : 64;
    std::vector<std::int32_t> toks;
    std::vector<float> single;
    if (!session->Evaluate(prompt, &logits, false, &error)) return 1;
    for (int i = 0; i < steps; ++i) {
      const std::int32_t t = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
      toks.push_back(t);
      if (!session->Evaluate(std::span(&toks.back(), 1), &logits, false, &error)) return 1;
      single.insert(single.end(), logits.begin(), logits.end());
    }
    // Row 0 of the first pass, layer by layer, against the one-token step.
    struct Dump {
      std::vector<std::vector<float>> layers;
      std::size_t width;
    };
    const std::size_t width = std::size_t{model->config().hc} * model->config().n_embd;
    auto record = [](void* user, int, const float* h, std::uint32_t) {
      auto* d = static_cast<Dump*>(user);
      d->layers.emplace_back(h, h + d->width);
    };
    Dump one{{}, width}, many{{}, width};
    {
      auto a = model->CreateSession(ctx, &error);
      if (!a || !a->Evaluate(prompt, &logits, false, &error)) return 1;
      a->SetLayerProbe(record, &one);
      if (!a->Evaluate(std::span(toks.data(), 1), &logits, false, &error)) return 1;
      auto b = model->CreateSession(ctx, &error);
      if (!b || !b->Evaluate(prompt, &logits, false, &error)) return 1;
      b->SetLayerProbe(record, &many);
      if (!b->Evaluate(std::span(toks.data(), std::min<std::size_t>(passcheck, toks.size())), &logits, false, &error)) {
        return 1;
      }
      for (std::size_t l = 0; l < one.layers.size() && l < many.layers.size(); ++l) {
        double d = 0.0;
        for (std::size_t i = 0; i < width; ++i) d = std::max(d, double(std::fabs(one.layers[l][i] - many.layers[l][i])));
        std::printf("layer %zu row0 max diff %.3g\n", l, d);
        if (d > 0) break;
      }
    }
    auto again = model->CreateSession(ctx, &error);
    if (!again || !again->Evaluate(prompt, &logits, false, &error)) return 1;
    std::vector<float> multi;
    for (std::size_t i = 0; i < toks.size(); i += passcheck) {
      const std::size_t n = std::min<std::size_t>(passcheck, toks.size() - i);
      if (!again->Evaluate(std::span(toks.data() + i, n), &logits, true, &error)) return 1;
      multi.insert(multi.end(), logits.begin(), logits.end());
    }
    double max_diff = 0.0, sum_diff = 0.0, sum_kl = 0.0;
    int same = 0;
    for (int i = 0; i < steps; ++i) {
      const float* a = single.data() + i * vocab;
      const float* b = multi.data() + i * vocab;
      same += std::max_element(a, a + vocab) - a == std::max_element(b, b + vocab) - b;
      double step_max = 0.0;
      for (std::size_t v = 0; v < vocab; ++v) {
        const double d = std::fabs(double(a[v]) - b[v]);
        step_max = std::max(step_max, d);
        sum_diff += d;
      }
      max_diff = std::max(max_diff, step_max);
      {
        const float ma = *std::max_element(a, a + vocab), mb = *std::max_element(b, b + vocab);
        double za = 0.0, zb = 0.0;
        for (std::size_t v = 0; v < vocab; ++v) {
          za += std::exp(double(a[v]) - ma);
          zb += std::exp(double(b[v]) - mb);
        }
        double kl = 0.0;
        for (std::size_t v = 0; v < vocab; ++v) {
          const double la = a[v] - ma - std::log(za), lb = b[v] - mb - std::log(zb);
          kl += std::exp(la) * (la - lb);
        }
        sum_kl += kl;
      }
      std::printf("%.3f%c", step_max, i % 16 == 15 ? '\n' : ' ');
    }
    std::printf("passcheck %u: %d/%d same top, max |dlogit| %.4f, mean %.6f, mean KL %.5f\n", passcheck, same, steps,
                max_diff, sum_diff / (double(steps) * vocab), sum_kl / steps);
    return 0;
  }
  if (!kl_path.empty()) {
    std::ifstream in(kl_path, std::ios::binary);
    char magic[8] = {};
    std::uint32_t n_ctx = 0;
    std::int32_t n_vocab = 0;
    std::int32_t n_chunk = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&n_ctx), 4);
    in.read(reinterpret_cast<char*>(&n_vocab), 4);
    in.read(reinterpret_cast<char*>(&n_chunk), 4);
    if (!in || std::string(magic, 8) != "_logits_" || n_vocab != static_cast<std::int32_t>(model->config().n_vocab)) {
      std::fprintf(stderr, "bad KL base file\n");
      return 1;
    }
    std::vector<std::int32_t> toks(std::size_t{n_ctx} * n_chunk);
    in.read(reinterpret_cast<char*>(toks.data()), static_cast<std::streamsize>(toks.size() * 4));
    const std::size_t vocab = static_cast<std::size_t>(n_vocab);
    const std::size_t nv = 2 * ((vocab + 1) / 2) + 4;
    std::vector<std::uint16_t> rec(nv);
    double sum_kl = 0.0, sum_kl2 = 0.0, nll = 0.0, nll_base = 0.0;
    std::size_t count = 0, same_top = 0;
    std::vector<double> logp(vocab);
    for (std::int32_t ch = 0; ch < n_chunk; ++ch) {
      auto s = model->CreateSession(n_ctx, &error);
      if (!s) {
        std::fprintf(stderr, "session failed: %s\n", error.c_str());
        return 1;
      }
      const std::int32_t* ct = toks.data() + std::size_t(ch) * n_ctx;
      const std::size_t first = n_ctx / 2;
      double chunk_kl = 0.0;
      std::size_t chunk_n = 0;
      for (std::size_t i = 0; i < n_ctx; i += hy4::Model::kMaxChunk) {
        const std::size_t n = std::min<std::size_t>(hy4::Model::kMaxChunk, n_ctx - i);
        const bool need = i + n > first;
        if (!s->Evaluate(std::span<const std::int32_t>(ct + i, n), need ? &logits : nullptr, true, &error)) {
          std::fprintf(stderr, "kl failed: %s\n", error.c_str());
          return 1;
        }
        if (!need) continue;
        for (std::size_t j = 0; j < n; ++j) {
          const std::size_t at = i + j;
          if (at < first || at + 1 >= n_ctx) continue;
          in.read(reinterpret_cast<char*>(rec.data()), static_cast<std::streamsize>(nv * 2));
          if (!in) {
            std::fprintf(stderr, "KL base file ended early\n");
            return 1;
          }
          float scale = 0.0f, min_lp = 0.0f;
          std::memcpy(&scale, rec.data(), 4);
          std::memcpy(&min_lp, rec.data() + 2, 4);
          const std::uint16_t* q = rec.data() + 4;
          const float* row = logits.data() + j * vocab;
          const float mx = *std::max_element(row, row + vocab);
          double den = 0.0;
          for (std::size_t v = 0; v < vocab; ++v) den += std::exp(static_cast<double>(row[v] - mx));
          const double lse = mx + std::log(den);
          double kl = 0.0;
          std::size_t top_base = 0, top = 0;
          float best_base = -1e30f;
          for (std::size_t v = 0; v < vocab; ++v) {
            const float lb = scale * q[v] + min_lp;
            if (lb > best_base) {
              best_base = lb;
              top_base = v;
            }
            if (row[v] > row[top]) top = v;
            if (lb > -16.0f) kl += std::exp(static_cast<double>(lb)) * (lb - (row[v] - lse));
          }
          const std::int32_t next = ct[at + 1];
          nll += lse - row[next];
          nll_base -= scale * q[next] + min_lp;
          sum_kl += kl;
          sum_kl2 += kl * kl;
          chunk_kl += kl;
          same_top += top == top_base;
          ++count;
          ++chunk_n;
        }
      }
      std::printf("chunk %d: mean KL %.5f\n", ch, chunk_kl / static_cast<double>(chunk_n));
      std::fflush(stdout);
    }
    const double mean = sum_kl / static_cast<double>(count);
    const double err = std::sqrt((sum_kl2 / static_cast<double>(count) - mean * mean) / static_cast<double>(count));
    std::printf("KL %.5f +/- %.5f, same top %.2f%%, PPL %.4f (base %.4f) over %zu tokens\n", mean, err,
                100.0 * static_cast<double>(same_top) / static_cast<double>(count),
                std::exp(nll / static_cast<double>(count)), std::exp(nll_base / static_cast<double>(count)), count);
    return 0;
  }
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
  std::printf("prompt %zu tokens in %.3f s (%.1f tok/s), Engram gather %.3f s\n", prompt.size(), Seconds(p0, p1),
              prompt.size() / Seconds(p0, p1), session->EngramSeconds());
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

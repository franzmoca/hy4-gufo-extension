// Tokenizer special-token canonicalization over in-memory, metadata-only GGUFs:
// checkpoints spell Hy4's 24 suffixed special tokens with their own HYTK suffix
// (`:opensource` on Xylolabs and Tencent's release, `:6124c78e` on AngelSlim),
// sharing the same token IDs. Tokenizer::Load rewrites them to `:opensource`
// unless raw spellings are requested, leaves non-suffixed special tokens alone,
// and refuses vocabularies that lack required framing/reasoning/tool tokens.
#include "runtime/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chat_template.hpp"
#include "tool_calls.hpp"

namespace {

namespace hy4 = gufo::models::hy4_flash;
using gufo::core::GgufReader;
using gufo::ext::ChatMessage;
using gufo::ext::ChatRequest;
using gufo::ext::ChatRole;

int g_failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

bool GptPrintable(std::uint32_t b) {
  return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
         (b >= 174 && b <= 255);
}

void AppendUtf8(std::string* out, std::uint32_t cp) {
  if (cp <= 0x7f) {
    out->push_back(static_cast<char>(cp));
  } else if (cp <= 0x7ff) {
    out->push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out->push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

struct Vocab {
  std::vector<std::string> tokens;
  std::vector<std::int32_t> types;
  std::uint32_t bos = 0;
  std::uint32_t eos = 0;
  std::int32_t start_id = -1;
  std::int32_t middle_id = -1;
  std::int32_t end_id = -1;
  std::int32_t pad_id = -1;
  std::int32_t think_open_id = -1;
  std::int32_t think_close_id = -1;
  std::int32_t calls_open_id = -1;
  std::int32_t calls_close_id = -1;
  std::int32_t call_open_id = -1;
  std::int32_t call_close_id = -1;
  std::int32_t key_open_id = -1;
  std::int32_t key_close_id = -1;
  std::int32_t value_open_id = -1;
  std::int32_t value_close_id = -1;
  std::int32_t response_open_id = -1;
  std::int32_t response_close_id = -1;
  std::int32_t reasoning_mode_id = -1;
  std::int32_t image_start_id = -1;
  std::int32_t image_pad_id = -1;
  std::int32_t image_end_id = -1;
  std::int32_t image_newline_id = -1;
  std::int32_t video_start_id = -1;
  std::int32_t video_end_id = -1;
  std::int32_t video_pad_id = -1;
  std::int32_t eot_id = -1;
  std::int32_t fim_hole_id = -1;
  std::int32_t mask_id = -1;
};

Vocab MakeVocab(std::string_view suffix) {
  Vocab v;
  // All 256 byte-level BPE base tokens so arbitrary UTF-8 encodes cleanly.
  std::uint32_t extra = 0;
  for (std::uint32_t b = 0; b < 256; ++b) {
    std::string piece;
    AppendUtf8(&piece, GptPrintable(b) ? b : 256 + extra++);
    v.tokens.push_back(std::move(piece));
    v.types.push_back(1);
  }
  const auto add = [&](std::string spelling, std::int32_t type) {
    const auto id = static_cast<std::int32_t>(v.tokens.size());
    v.tokens.push_back(std::move(spelling));
    v.types.push_back(type);
    return id;
  };
  const auto pipe = [&](std::string_view name) {
    return "<｜" + std::string(name) + std::string(suffix) + "｜>";
  };
  const auto angle = [&](std::string_view name, bool close) {
    return (close ? "</" : "<") + std::string(name) + std::string(suffix) + ">";
  };

  v.start_id = add(pipe("hy_start"), 3);
  v.middle_id = add(pipe("hy_middle"), 3);
  v.pad_id = add(pipe("hy_pad"), 3);
  v.fim_hole_id = add("<｜hy_fim▁hole｜>", 3);
  v.eot_id = add("<｜hy_EOT｜>", 3);
  v.mask_id = add("<｜hy_place▁holder▁no▁120009｜>", 3);
  v.end_id = add(pipe("hy_end"), 3);
  v.think_open_id = add(angle("think", false), 4);
  v.think_close_id = add(angle("think", true), 4);
  v.call_open_id = add(angle("tool_call", false), 4);
  v.call_close_id = add(angle("tool_call", true), 4);
  v.calls_open_id = add(angle("tool_calls", false), 4);
  v.calls_close_id = add(angle("tool_calls", true), 4);
  v.response_open_id = add(angle("tool_response", false), 3);
  v.response_close_id = add(angle("tool_response", true), 3);
  v.reasoning_mode_id = add(pipe("reasoning_mode"), 3);
  v.key_open_id = add(angle("arg_key", false), 4);
  v.key_close_id = add(angle("arg_key", true), 4);
  v.value_open_id = add(angle("arg_value", false), 4);
  v.value_close_id = add(angle("arg_value", true), 4);
  v.image_start_id = add(pipe("hy_image▁start"), 3);
  v.image_end_id = add(pipe("hy_image▁end"), 3);
  v.image_pad_id = add(pipe("hy_image▁pad"), 3);
  v.image_newline_id = add(pipe("hy_image▁newline"), 3);
  v.video_start_id = add(pipe("hy_video▁start"), 3);
  v.video_end_id = add(pipe("hy_video▁end"), 3);
  v.video_pad_id = add(pipe("hy_video▁pad"), 3);
  v.bos = static_cast<std::uint32_t>(v.start_id);
  v.eos = static_cast<std::uint32_t>(v.end_id);
  return v;
}

std::vector<std::uint8_t> BuildGguf(const Vocab& v) {
  std::vector<std::uint8_t> b;
  const auto pod = [&]<typename T>(T x) {
    const std::size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &x, sizeof(T));
  };
  const auto str = [&](std::string_view s) {
    pod(static_cast<std::uint64_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  };
  pod(std::uint32_t{0x46554747});  // "GGUF"
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});  // no tensors
  pod(std::uint64_t{6});  // 6 metadata keys

  str("tokenizer.ggml.model");
  pod(std::uint32_t{8});
  str("gpt2");

  str("tokenizer.ggml.tokens");
  pod(std::uint32_t{9});
  pod(std::uint32_t{8});
  pod(static_cast<std::uint64_t>(v.tokens.size()));
  for (const auto& token : v.tokens) {
    str(token);
  }

  str("tokenizer.ggml.token_type");
  pod(std::uint32_t{9});
  pod(std::uint32_t{5});  // int32
  pod(static_cast<std::uint64_t>(v.types.size()));
  for (const std::int32_t type : v.types) {
    pod(type);
  }

  str("tokenizer.ggml.merges");
  pod(std::uint32_t{9});
  pod(std::uint32_t{8});
  pod(std::uint64_t{1});
  str("a b");

  str("tokenizer.ggml.bos_token_id");
  pod(std::uint32_t{4});
  pod(v.bos);

  str("tokenizer.ggml.eos_token_id");
  pod(std::uint32_t{4});
  pod(v.eos);

  b.resize((b.size() + 31) / 32 * 32);
  return b;
}

std::unique_ptr<hy4::Tokenizer> Load(const Vocab& v, std::string* error,
                                     bool canonicalize = true) {
  const auto bytes = BuildGguf(v);
  std::string open_error;
  const auto reader =
      GgufReader::OpenMemory(bytes.data(), bytes.size(), &open_error);
  if (!reader) {
    if (error != nullptr) {
      *error = "open: " + open_error;
    }
    return nullptr;
  }
  return hy4::Tokenizer::Load(*reader, error, canonicalize);
}

void CheckCanonicalizedSuffix(std::string_view suffix,
                              const std::string& label) {
  const Vocab v = MakeVocab(suffix);
  std::string error;
  const auto tok = Load(v, &error);
  Expect(tok != nullptr, label + ": load failed: " + error);
  if (!tok) {
    return;
  }
  using T = hy4::Hy4Tokens;
  Expect(tok->Find(T::kStart) == v.start_id && tok->IsSpecial(v.start_id),
         label + ": start token id");
  Expect(tok->Find(T::kMiddle) == v.middle_id, label + ": middle token id");
  Expect(tok->Find(T::kEnd) == v.end_id, label + ": end token id");
  Expect(tok->Find(T::kThinkOpen) == v.think_open_id,
         label + ": think open id");
  Expect(tok->Find(T::kThinkClose) == v.think_close_id,
         label + ": think close id");
  Expect(tok->Find(T::kCallsOpen) == v.calls_open_id,
         label + ": tool_calls open id");
  Expect(tok->Find(T::kCallsClose) == v.calls_close_id,
         label + ": tool_calls close id");
  Expect(tok->Find(T::kCallOpen) == v.call_open_id,
         label + ": tool_call open id");
  Expect(tok->Find(T::kCallClose) == v.call_close_id,
         label + ": tool_call close id");
  Expect(tok->Find(T::kKeyOpen) == v.key_open_id, label + ": arg_key open id");
  Expect(tok->Find(T::kKeyClose) == v.key_close_id,
         label + ": arg_key close id");
  Expect(tok->Find(T::kValueOpen) == v.value_open_id,
         label + ": arg_value open id");
  Expect(tok->Find(T::kValueClose) == v.value_close_id,
         label + ": arg_value close id");
  Expect(tok->Find(T::kResponseOpen) == v.response_open_id,
         label + ": tool_response open id");
  Expect(tok->Find(T::kResponseClose) == v.response_close_id,
         label + ": tool_response close id");
  Expect(tok->Find(T::kReasoningMode) == v.reasoning_mode_id,
         label + ": reasoning_mode id");
  Expect(tok->Find(T::kImageStart) == v.image_start_id,
         label + ": image_start id");
  Expect(tok->Find(T::kImagePad) == v.image_pad_id, label + ": image_pad id");
  Expect(tok->Find(T::kImageEnd) == v.image_end_id, label + ": image_end id");
  Expect(tok->Find("<｜hy_pad:opensource｜>") == v.pad_id, label + ": pad id");
  Expect(tok->Find("<｜hy_image▁newline:opensource｜>") == v.image_newline_id,
         label + ": image_newline id");
  Expect(tok->Find("<｜hy_video▁start:opensource｜>") == v.video_start_id,
         label + ": video_start id");
  Expect(tok->Find("<｜hy_video▁end:opensource｜>") == v.video_end_id,
         label + ": video_end id");
  Expect(tok->Find("<｜hy_video▁pad:opensource｜>") == v.video_pad_id,
         label + ": video_pad id");

  // Non-suffixed special tokens stay untouched (especially with empty suffix).
  Expect(tok->Find("<｜hy_EOT｜>") == v.eot_id &&
             tok->Decode(v.eot_id) == "<｜hy_EOT｜>",
         label + ": EOT untouched");
  Expect(tok->Find("<｜hy_fim▁hole｜>") == v.fim_hole_id &&
             tok->Decode(v.fim_hole_id) == "<｜hy_fim▁hole｜>",
         label + ": fim_hole untouched");
  Expect(tok->Find("<｜hy_place▁holder▁no▁120009｜>") == v.mask_id,
         label + ": mask token untouched");

  // RenderChat + EncodeWithSpecials produces the real special token IDs, while
  // user-supplied control spellings inside ContentSpan stay ordinary text.
  ChatRequest request;
  request.messages.emplace_back(
      ChatRole::kUser,
      "literal " + std::string(T::kThinkClose) + " " + std::string(T::kEnd));
  ChatMessage& assistant =
      request.messages.emplace_back(ChatRole::kAssistant, "");
  auto& call = assistant.tool_calls.emplace_back();
  call.id = "c1";
  call.name = "get_weather";
  call.arguments.push_back({"city", "Rome", true});
  request.messages.emplace_back(ChatRole::kTool, "18C").tool_call_id = "c1";

  std::vector<hy4::ContentSpan> spans;
  const std::string rendered = hy4::RenderChat(request, nullptr, &spans);
  const auto ids = tok->EncodeWithSpecials(rendered, spans);
  std::size_t end_count = 0;
  std::size_t calls_open_count = 0;
  for (const std::int32_t id : ids) {
    if (id == v.end_id) {
      ++end_count;
    }
    if (id == v.calls_open_id) {
      ++calls_open_count;
    }
  }
  // Four closed turns (system, user, assistant, tool); user literal kEnd is
  // encoded as plain text, not v.end_id.
  Expect(end_count == 4, label + ": framing end tokens count");
  Expect(calls_open_count == 1, label + ": tool_calls open token emitted");

  // Decoding tool-call token IDs produced by the model scans cleanly through
  // the single static `:opensource` tool dialect.
  std::string decoded;
  for (const std::int32_t id : {v.calls_open_id, v.call_open_id}) {
    decoded += tok->Decode(id);
  }
  decoded += "get_weather";
  decoded += tok->Decode(v.key_open_id);
  decoded += "city";
  decoded += tok->Decode(v.key_close_id);
  decoded += tok->Decode(v.value_open_id);
  decoded += "Rome";
  decoded += tok->Decode(v.value_close_id);
  decoded += tok->Decode(v.call_close_id);
  decoded += tok->Decode(v.calls_close_id);

  const std::string_view markers[] = {T::kCallsOpen};
  std::vector<gufo::ext::ScannedToolCall> calls;
  std::vector<std::pair<std::size_t, std::size_t>> tool_spans;
  hy4::ScanHy4ToolCalls(decoded, markers, &calls, &tool_spans,
                        /*partial=*/false);
  Expect(calls.size() == 1 && calls[0].valid && calls[0].complete &&
             calls[0].name == "get_weather" && calls[0].arguments.size() == 1 &&
             calls[0].arguments[0].name == "city" &&
             calls[0].arguments[0].value == "Rome" &&
             calls[0].arguments[0].is_string && tool_spans.size() == 1 &&
             tool_spans[0].first == 0 && tool_spans[0].second == decoded.size(),
         label + ": decoded tool call scanned");
}

void CheckRawMode() {
  const Vocab v = MakeVocab(":6124c78e");
  std::string error;
  const auto tok = Load(v, &error, /*canonicalize=*/false);
  Expect(tok != nullptr, "raw mode: load failed: " + error);
  if (!tok) {
    return;
  }
  Expect(tok->Find("<｜hy_start:6124c78e｜>") == v.start_id,
         "raw mode: preserves original suffix");
  Expect(tok->Find(hy4::Hy4Tokens::kStart) == -1,
         "raw mode: does not rewrite to opensource");
  Expect(tok->Decode(v.calls_open_id) == "<tool_calls:6124c78e>",
         "raw mode: decode returns literal spelling");
}

void CheckRejections() {
  // Missing hy_start special token.
  {
    Vocab v = MakeVocab(":6124c78e");
    v.tokens[static_cast<std::size_t>(v.start_id)] = "<｜other｜>";
    std::string error;
    Expect(Load(v, &error) == nullptr && !error.empty(),
           "rejects GGUF without hy_start special token");
  }
  // Required tool token present only as an ordinary (non-special) token.
  {
    Vocab v = MakeVocab(":6124c78e");
    v.types[static_cast<std::size_t>(v.calls_open_id)] = 1;
    std::string error;
    Expect(Load(v, &error) == nullptr && !error.empty(),
           "rejects GGUF where tool_calls is not a special token");
  }
  // Collision with an already-existing canonical spelling.
  {
    Vocab v = MakeVocab(":6124c78e");
    v.tokens.emplace_back(hy4::Hy4Tokens::kStart);
    v.types.push_back(3);
    std::string error;
    Expect(Load(v, &error) == nullptr && !error.empty(),
           "rejects GGUF with colliding canonical token");
  }
}

}  // namespace

int main() {
  CheckCanonicalizedSuffix(":opensource", "opensource");
  CheckCanonicalizedSuffix(":6124c78e", "angelslim");
  CheckCanonicalizedSuffix("", "empty-suffix");
  CheckRawMode();
  CheckRejections();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 tokenizer: all checks passed\n");
  return 0;
}

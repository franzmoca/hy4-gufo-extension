// RenderChat against the GGUF's own Jinja template (tests/fixtures/
// chat_goldens.json, rendered by tools/chat_goldens.py), image offsets, and
// Hy4 tool-call blocks scanned back into the calls that produced them. Every
// check runs twice: once over the Xylolabs template, whose tokens carry the
// `:opensource` suffix, and once over a trunk whose tokens carry another, since
// a prompt built with one checkpoint's spellings is plain text to the other.
#include "chat_template.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "text_contract.hpp"
#include "tool_calls.hpp"

namespace {

namespace hy4 = gufo::models::hy4_flash;
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

ChatRole Role(const std::string& role) {
  if (role == "system")
    return ChatRole::kSystem;
  if (role == "developer")
    return ChatRole::kDeveloper;
  if (role == "assistant")
    return ChatRole::kAssistant;
  if (role == "tool")
    return ChatRole::kTool;
  return ChatRole::kUser;
}

ChatRequest Request(const gufo::json::Value& c) {
  ChatRequest request;
  for (const auto& m : c.find("messages")->items()) {
    ChatMessage& out = request.messages.emplace_back(Role(m.member_str("role")),
                                                     m.member_str("content"));
    out.thought = m.member_str("reasoning");
    out.tool_call_id = m.member_str("tool_call_id");
    if (const auto* images = m.find("images")) {
      for (const auto& offset : images->items()) {
        out.images.push_back(
            {offset.as_size(),
             std::make_shared<const std::vector<std::uint8_t>>()});
      }
    }
    if (const auto* calls = m.find("tool_calls")) {
      for (const auto& call : calls->items()) {
        auto& tool_call = out.tool_calls.emplace_back();
        tool_call.id = call.member_str("id");
        tool_call.name = call.member_str("name");
        for (const auto& a : call.find("arguments")->items()) {
          tool_call.arguments.push_back({a.member_str("name"),
                                         a.member_str("value"),
                                         a.find("is_string")->as_bool()});
        }
      }
    }
  }
  if (const auto* tools = c.find("tools")) {
    for (const auto& tool : tools->items()) {
      auto& out = request.tools.emplace_back();
      out.name = tool.find("function")->member_str("name");
      out.definition_json = tool.tojson(gufo::json::Value::Numbers::kAsParsed);
    }
  }
  if (c.member_str("tool_choice") == "none")
    request.tool_choice = ChatRequest::ToolChoice::kNone;
  const std::string effort = c.member_str("effort", "high");
  if (effort == "no_think")
    request.reasoning.enabled = false;
  if (effort == "max")
    request.reasoning.effort = gufo::ReasoningEffort::kMax;
  return request;
}

// Every Hy4Tokens spelling in `text`, in order, as (offset, spelling).
std::vector<std::pair<std::size_t, std::string_view>> Controls(
    const hy4::Hy4Tokens& tokens, std::string_view text) {
  const std::string_view kAll[] = {
      tokens.start,       tokens.middle,       tokens.end,
      tokens.think_open,  tokens.think_close,  tokens.calls_open,
      tokens.calls_close, tokens.call_open,    tokens.call_close,
      tokens.key_open,    tokens.key_close,    tokens.value_open,
      tokens.value_close, tokens.response_open, tokens.response_close,
      tokens.reasoning_mode, tokens.image_start, tokens.image_pad,
      tokens.image_end};
  std::vector<std::pair<std::size_t, std::string_view>> out;
  for (std::size_t at = 0; at < text.size(); ++at) {
    for (const auto spelling : kAll) {
      if (text.substr(at).starts_with(spelling)) {
        out.emplace_back(at, spelling);
        break;
      }
    }
  }
  return out;
}

bool Overlaps(std::span<const hy4::ContentSpan> spans, std::size_t at,
              std::size_t size) {
  for (const auto& span : spans) {
    if (span.offset < at + size && at < span.offset + span.size)
      return true;
  }
  return false;
}

// The control spellings outside `spans`: what tokenizes as control tokens.
std::vector<std::string_view> Framing(const hy4::Hy4Tokens& tokens,
                                      const std::string& text,
                                      std::span<const hy4::ContentSpan> spans) {
  std::vector<std::string_view> out;
  for (const auto& [at, spelling] : Controls(tokens, text)) {
    if (!Overlaps(spans, at, spelling.size()))
      out.push_back(spelling);
  }
  return out;
}

bool Ordered(std::span<const hy4::ContentSpan> spans, std::size_t size) {
  std::size_t end = 0;
  for (const auto& span : spans) {
    if (span.size == 0 || span.offset < end || span.offset + span.size > size)
      return false;
    end = span.offset + span.size;
  }
  return true;
}

// Where the tokenizer may cut `text`: before and after each control token
// outside `spans`, and at both ends.
std::vector<std::size_t> Cuts(const hy4::Hy4Tokens& tokens,
                              const std::string& text,
                              std::span<const hy4::ContentSpan> spans) {
  std::vector<std::size_t> cuts{0, text.size()};
  for (const auto& [at, spelling] : Controls(tokens, text)) {
    if (!Overlaps(spans, at, spelling.size())) {
      cuts.push_back(at);
      cuts.push_back(at + spelling.size());
    }
  }
  return cuts;
}

// The stable prefix is the last cut at or before the first byte a following
// user turn changes, so its tokens are the ones the prompt and that turn's
// prompt share.
void CheckStablePrefix(const hy4::Hy4Tokens& tokens, const ChatRequest& request,
                       const std::string& name) {
  std::vector<hy4::ContentSpan> spans;
  std::size_t stable = 0;
  const std::string text =
      hy4::RenderChat(tokens, request, nullptr, &spans, &stable);
  ChatRequest next = request;
  next.messages.emplace_back(ChatRole::kUser, "");
  const std::string continued = hy4::RenderChat(tokens, next);
  std::size_t diverge = 0;
  while (diverge < text.size() && diverge < continued.size() &&
         text[diverge] == continued[diverge])
    ++diverge;
  const auto cuts = Cuts(tokens, text, spans);
  std::size_t last = 0;
  for (const std::size_t cut : cuts) {
    if (cut <= diverge)
      last = std::max(last, cut);
  }
  Expect(stable == last, name + ": stable prefix at byte " +
                             std::to_string(stable) + ", want " +
                             std::to_string(last));
  // With images rendered apart, the same prefix less the images before it.
  std::vector<std::size_t> offsets;
  std::size_t bare_stable = 0;
  (void)hy4::RenderChat(tokens, request, &offsets, nullptr, &bare_stable);
  // The prefix follows a turn start or a think tag, never an image.
  std::size_t images_before = 0;
  for (const std::size_t offset : offsets) {
    if (offset < bare_stable)
      ++images_before;
  }
  const std::size_t image_bytes = tokens.image_start.size() +
                                  tokens.image_pad.size() +
                                  tokens.image_end.size();
  Expect(stable - bare_stable == images_before * image_bytes,
         name + ": stable prefix with image offsets");
}

// The first difference, for the failure message.
std::string Diff(const std::string& got, const std::string& want) {
  std::size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i])
    ++i;
  const auto around = [&](const std::string& s) {
    return s.substr(i < 40 ? 0 : i - 40, 100);
  };
  return "at byte " + std::to_string(i) + "\n  got:  " + around(got) +
         "\n  want: " + around(want);
}

void CheckGoldens(const char* path, const hy4::Hy4Tokens& tokens) {
  std::ifstream in(path, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  if (text.empty()) {
    Expect(false, std::string("read ") + path);
    return;
  }
  const auto goldens = gufo::json::parse(text);
  std::size_t count = 0;
  for (const auto& c : goldens.find("cases")->items()) {
    ++count;
    const std::string name = c.member_str("name");
    const ChatRequest request = Request(c);
    const std::string want = c.member_str("rendered");
    const std::string got = hy4::RenderChat(tokens, request);
    Expect(got == want, name + ": " + Diff(got, want));

    // With offsets, images render as nothing and land where the tokens were.
    std::vector<std::size_t> offsets;
    const std::string bare = hy4::RenderChat(tokens, request, &offsets);
    const std::string image =
        tokens.image_start + tokens.image_pad + tokens.image_end;
    std::string rebuilt;
    std::size_t cursor = 0;
    for (const std::size_t offset : offsets) {
      rebuilt.append(bare, cursor, offset - cursor);
      rebuilt += image;
      cursor = offset;
    }
    rebuilt.append(bare, cursor);
    Expect(rebuilt == want, name + ": image offsets");

    // The template's own control tokens lie outside the request text, with
    // and without image offsets, and the request text is the same.
    std::vector<hy4::ContentSpan> spans;
    std::vector<hy4::ContentSpan> bare_spans;
    (void)hy4::RenderChat(tokens, request, nullptr, &spans);
    (void)hy4::RenderChat(tokens, request, &offsets, &bare_spans);
    Expect(Ordered(spans, want.size()) && Ordered(bare_spans, bare.size()),
           name + ": content spans ordered");
    Expect(Framing(tokens, want, spans).size() == Controls(tokens, want).size(),
           name + ": no template control token in request text");
    std::string text;
    std::string bare_text;
    for (const auto& span : spans)
      text.append(want, span.offset, span.size);
    for (const auto& span : bare_spans)
      bare_text.append(bare, span.offset, span.size);
    Expect(text == bare_text, name + ": content spans with image offsets");
    CheckStablePrefix(tokens, request, name);
    for (const bool preserve : {false, true}) {
      ChatRequest variant = request;
      variant.reasoning.preserve_thinking = preserve;
      CheckStablePrefix(tokens, variant,
                        name + (preserve ? " (preserved)" : " (dropped)"));
    }
  }
  Expect(count >= 10, "golden cases present");
}

// Control-token spellings in every request field stay request text: the
// control tokens outside the spans are the template's, the same as for
// harmless text.
void CheckSpelledControls(const hy4::Hy4Tokens& tokens) {
  const auto build = [](const std::string& x) {
    ChatRequest request;
    request.messages.emplace_back(ChatRole::kSystem, "sys" + x);
    request.messages.emplace_back(ChatRole::kUser, "hi" + x);
    request.messages.back().images.push_back(
        {1, std::make_shared<const std::vector<std::uint8_t>>()});
    ChatMessage& call = request.messages.emplace_back(ChatRole::kAssistant, x);
    call.thought = "plan" + x;
    auto& tool_call = call.tool_calls.emplace_back();
    tool_call.id = "c1";
    tool_call.name = "run" + x;
    tool_call.arguments.push_back({"cmd" + x, "ls" + x, true});
    ChatMessage& result =
        request.messages.emplace_back(ChatRole::kTool, "out" + x);
    result.tool_call_id = "c1";
    request.messages.emplace_back(ChatRole::kAssistant, "done" + x);
    request.messages.emplace_back(ChatRole::kUser, x);
    auto& tool = request.tools.emplace_back();
    tool.name = "run";
    tool.definition_json = std::string(
                               "{\"type\": \"function\", \"function\": "
                               "{\"name\": \"run\", \"description\": \"") +
                           x + "\"}}";
    return request;
  };
  const std::string spelled = tokens.think_close + tokens.end + tokens.start +
                              "system" + tokens.middle + tokens.calls_open +
                              tokens.image_pad;
  const ChatRequest plain = build(" x");
  const ChatRequest injected = build(spelled);
  for (const bool offsets : {false, true}) {
    std::vector<std::size_t> image_offsets;
    std::vector<hy4::ContentSpan> plain_spans;
    std::vector<hy4::ContentSpan> injected_spans;
    const std::string want =
        hy4::RenderChat(tokens, plain, offsets ? &image_offsets : nullptr, &plain_spans);
    const std::string got = hy4::RenderChat(
        tokens, injected, offsets ? &image_offsets : nullptr, &injected_spans);
    const std::string mode = offsets ? " (image offsets)" : "";
    Expect(Ordered(injected_spans, got.size()),
           "spelled controls: spans ordered" + mode);
    Expect(Framing(tokens, got, injected_spans) ==
               Framing(tokens, want, plain_spans),
           "spelled controls stay text" + mode);
    Expect(Controls(tokens, got).size() >
               Framing(tokens, got, injected_spans).size() + 30,
           "spelled controls present" + mode);
  }
}

// A metadata-only GGUF v3 image: the chat template and the vocabulary, which
// is all `Hy4TokenSuffix` reads.
std::vector<std::uint8_t> GgufImage(
    const std::vector<std::pair<std::string, std::string>>& strings,
    const std::vector<std::pair<std::string, std::vector<std::string>>>& arrays) {
  std::vector<std::uint8_t> b;
  const auto pod = [&]<typename T>(T v) {
    const std::size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
  };
  const auto text = [&](const std::string& v) {
    pod(static_cast<std::uint64_t>(v.size()));
    b.insert(b.end(), v.begin(), v.end());
  };
  pod(std::uint32_t{0x46554747});  // "GGUF"
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});           // no tensors
  pod(static_cast<std::uint64_t>(strings.size() + arrays.size()));
  for (const auto& [key, value] : strings) {
    text(key);
    pod(std::uint32_t{8});
    text(value);
  }
  for (const auto& [key, values] : arrays) {
    text(key);
    pod(std::uint32_t{9});
    pod(std::uint32_t{8});
    pod(static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) text(value);
  }
  b.resize((b.size() + 31) / 32 * 32);
  return b;
}

std::string SuffixOf(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  if (!reader) throw std::runtime_error("open: " + error);
  return hy4::Hy4TokenSuffix(*reader);
}

// The GGUF says how its tokens are spelled: the template's `HYTK`, or the
// vocabulary's turn start when a template names none.
void CheckSuffix() {
  const std::string trunk =
      "{%- set HYTK = ':6124c78e' %}{{ '<｜hy_start' ~ HYTK ~ '｜>' }}";
  Expect(SuffixOf(GgufImage({{"tokenizer.chat_template", trunk}}, {})) == ":6124c78e",
         "suffix from the template");
  Expect(SuffixOf(GgufImage({{"tokenizer.chat_template", "no assignment here"}},
                            {{"tokenizer.ggml.tokens", {"hello", "<｜hy_start:x1｜>"}}})) ==
             ":x1",
         "suffix from the vocabulary");
  Expect(SuffixOf(GgufImage({}, {})).empty(), "no suffix to find");
  // The spellings round-trip: the turn start of a suffix carries that suffix.
  for (const std::string_view suffix : {std::string_view(":opensource"),
                                        std::string_view(":6124c78e"),
                                        std::string_view("")}) {
    const hy4::Hy4Tokens tokens = hy4::MakeHy4Tokens(suffix);
    Expect(tokens.suffix == suffix, "suffix kept");
    Expect(hy4::Hy4Tokens::SuffixOf(tokens.start) == suffix,
           "round trip " + std::string(suffix));
    Expect(tokens.call_header == tokens.calls_open + tokens.call_open, "call header");
    Expect(tokens.calls_open.size() > 2 && tokens.calls_open.front() == '<' &&
               tokens.calls_open.back() == '>',
           "the block opener is a tag the host can find");
  }
}

// The point of the suffix: one checkpoint's tags are plain text to another, so
// a scan never reads a call block of the other spelling, and a rendered prompt
// carries no tag of a checkpoint it is not serving.
void CheckCrossSuffix() {
  const hy4::Hy4Tokens opensource = hy4::OpensourceHy4Tokens();
  const hy4::Hy4Tokens trunk = hy4::MakeHy4Tokens(":6124c78e");
  ChatRequest request;
  request.messages.emplace_back(ChatRole::kUser, "Weather?");
  ChatMessage& turn = request.messages.emplace_back(ChatRole::kAssistant, "");
  auto& call = turn.tool_calls.emplace_back();
  call.id = "c1";
  call.name = "get_weather";
  call.arguments.push_back({"city", "Rome", true});

  const std::string as_opensource = hy4::RenderChat(opensource, request);
  const std::string as_trunk = hy4::RenderChat(trunk, request);
  Expect(as_opensource != as_trunk, "the two checkpoints render differently");
  Expect(as_opensource.find(":opensource") != std::string::npos &&
             as_trunk.find(":opensource") == std::string::npos,
         "a trunk prompt holds no opensource spelling");

  // Scanned back: each dialect reads its own block and nothing of the other's.
  const std::string_view opensource_markers[] = {opensource.calls_open};
  const std::string_view trunk_markers[] = {trunk.calls_open};
  for (const bool partial : {false, true}) {
    std::vector<gufo::ext::ScannedToolCall> calls;
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    hy4::ScanHy4ToolCalls(as_trunk, opensource_markers, &calls, &spans, partial);
    Expect(calls.empty() && spans.empty(),
           std::string("an opensource dialect misses a trunk block (partial ") +
               (partial ? "on)" : "off)"));
    hy4::ScanHy4ToolCalls(as_opensource, trunk_markers, &calls, &spans, partial);
    Expect(calls.empty() && spans.empty(),
           std::string("a trunk dialect misses an opensource block (partial ") +
               (partial ? "on)" : "off)"));
    hy4::ScanHy4ToolCalls(as_trunk, trunk_markers, &calls, &spans, false);
    Expect(calls.size() == 1 && calls[0].valid && calls[0].complete &&
               calls[0].name == "get_weather" && calls[0].arguments.size() == 1 &&
               calls[0].arguments[0].value == "Rome" && calls[0].arguments[0].is_string,
           "a trunk dialect reads its own block");
  }

  // The dialect of a suffix is its own identity, with the suffix in its tags.
  Expect(hy4::ToolDialectId(opensource) == "hy4flash", "the opensource dialect id");
  Expect(hy4::ToolDialectId(trunk) == "hy4flash-6124c78e", "a trunk dialect id");
  const gufo::ext::ToolDialect dialect = hy4::ToolDialect(trunk);
  Expect(dialect.markers.size() == 1 && dialect.markers[0] == trunk.calls_open,
         "a trunk dialect marks its own block");
  Expect(dialect.reasoning_markers.size() == 1 &&
             dialect.reasoning_markers[0] == trunk.call_header,
         "a trunk dialect ends reasoning at its own call header");
  Expect(dialect.syntax && dialect.syntax->open == trunk.calls_open &&
             dialect.syntax->close == trunk.call_close &&
             dialect.syntax->block_close == trunk.calls_close &&
             dialect.syntax->tagged &&
             dialect.syntax->tagged->parameter_open == trunk.key_open &&
             dialect.syntax->tagged->value_close == trunk.value_close,
         "a trunk dialect constrains its own grammar");
}

void CheckSystemImage() {
  ChatRequest request;
  request.messages.emplace_back(ChatRole::kSystem, "look");
  request.messages.back().images.push_back({0, nullptr});
  bool threw = false;
  try {
    (void)hy4::RenderChat(hy4::OpensourceHy4Tokens(), request);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "system message images are rejected");
}

void CheckEffortNames() {
  gufo::ReasoningOptions r;
  Expect(hy4::ReasoningEffortName(r) == "high", "default effort");
  r.effort = gufo::ReasoningEffort::kLow;
  Expect(hy4::ReasoningEffortName(r) == "high", "low effort maps to high");
  r.effort = gufo::ReasoningEffort::kXHigh;
  Expect(hy4::ReasoningEffortName(r) == "max", "xhigh maps to max");
  r.enabled = false;
  Expect(hy4::ReasoningEffortName(r) == "no_think", "disabled reasoning");
}

void CheckToolScan(const hy4::Hy4Tokens& tokens) {
  const std::string block = tokens.calls_open + tokens.call_open + "search" +
                            tokens.key_open + "q" + tokens.key_close +
                            tokens.value_open + "ünïcode <tags>" +
                            tokens.value_close + tokens.key_open + "limit" +
                            tokens.key_close + tokens.value_open + "5" +
                            tokens.value_close + tokens.key_open + "filters" +
                            tokens.key_close + tokens.value_open +
                            "{\"lang\": [\"en\"]}" + tokens.value_close +
                            tokens.call_close + tokens.call_open + "noop" +
                            tokens.call_close + tokens.calls_close;
  const std::string text = "Let me look." + block;
  const std::string_view markers[] = {tokens.calls_open};
  std::vector<gufo::ext::ScannedToolCall> calls;
  std::vector<std::pair<std::size_t, std::size_t>> spans;
  hy4::ScanHy4ToolCalls(text, markers, &calls, &spans, /*partial=*/false);
  Expect(calls.size() == 2, "scan: two calls");
  Expect(spans.size() == 1 && spans[0].first == 12 &&
             spans[0].second == text.size(),
         "scan: block span");
  if (calls.size() == 2) {
    const auto& c = calls[0];
    Expect(
        c.name == "search" && c.valid && c.complete && c.arguments.size() == 3,
        "scan: first call");
    if (c.arguments.size() == 3) {
      Expect(c.arguments[0].name == "q" &&
                 c.arguments[0].value == "ünïcode <tags>" &&
                 c.arguments[0].is_string,
             "scan: string argument");
      Expect(c.arguments[1].name == "limit" && c.arguments[1].value == "5" &&
                 !c.arguments[1].is_string,
             "scan: number argument");
      Expect(c.arguments[2].name == "filters" && !c.arguments[2].is_string,
             "scan: object argument");
    }
    Expect(calls[1].name == "noop" && calls[1].arguments.empty() &&
               calls[1].complete,
           "scan: call without arguments");
  }

  // A stream cut inside the second argument: one partial call, unfinished.
  const std::string cut = text.substr(0, text.find("5" + tokens.value_close));
  calls.clear();
  spans.clear();
  hy4::ScanHy4ToolCalls(cut, markers, &calls, &spans, /*partial=*/true);
  Expect(
      calls.size() == 1 && !calls[0].complete && calls[0].arguments.size() == 1,
      "scan: partial call");

  calls.clear();
  spans.clear();
  hy4::ScanHy4ToolCalls("No tools here.", markers, &calls, &spans, false);
  Expect(calls.empty() && spans.empty(), "scan: plain text");
}

}  // namespace

int main() {
  const hy4::Hy4Tokens opensource = hy4::OpensourceHy4Tokens();
  const hy4::Hy4Tokens trunk = hy4::MakeHy4Tokens(":6124c78e");
  try {
    CheckGoldens(GUFO_HY4_CHAT_GOLDENS, opensource);
    CheckGoldens(GUFO_HY4_CHAT_GOLDENS_OTHER, trunk);
    CheckSuffix();
    CheckCrossSuffix();
    CheckSystemImage();
    CheckSpelledControls(opensource);
    CheckSpelledControls(trunk);
    CheckEffortNames();
    CheckToolScan(opensource);
    CheckToolScan(trunk);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL exception: %s\n", e.what());
    return 1;
  }
  if (g_failures) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 chat template: all checks passed\n");
  return 0;
}

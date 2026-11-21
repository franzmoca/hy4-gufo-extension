// RenderChat against the GGUF's own Jinja template (tests/fixtures/
// chat_goldens.json, rendered by tools/chat_goldens.py), image offsets, and
// Hy4 tool-call blocks scanned back into the calls that produced them.
#include "chat_template.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    std::string_view text) {
  using T = hy4::Hy4Tokens;
  static constexpr std::string_view kAll[] = {
      T::kStart,      T::kMiddle,       T::kEnd,           T::kThinkOpen,
      T::kThinkClose, T::kCallsOpen,    T::kCallsClose,    T::kCallOpen,
      T::kCallClose,  T::kKeyOpen,      T::kKeyClose,      T::kValueOpen,
      T::kValueClose, T::kResponseOpen, T::kResponseClose, T::kReasoningMode,
      T::kImageStart, T::kImagePad,     T::kImageEnd};
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
std::vector<std::string_view> Framing(const std::string& text,
                                      std::span<const hy4::ContentSpan> spans) {
  std::vector<std::string_view> out;
  for (const auto& [at, spelling] : Controls(text)) {
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

void CheckGoldens() {
  std::ifstream in(GUFO_HY4_CHAT_GOLDENS, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  if (text.empty()) {
    Expect(false, "read " GUFO_HY4_CHAT_GOLDENS);
    return;
  }
  const auto goldens = gufo::json::parse(text);
  std::size_t count = 0;
  for (const auto& c : goldens.find("cases")->items()) {
    ++count;
    const std::string name = c.member_str("name");
    const ChatRequest request = Request(c);
    const std::string want = c.member_str("rendered");
    const std::string got = hy4::RenderChat(request);
    Expect(got == want, name + ": " + Diff(got, want));

    // With offsets, images render as nothing and land where the tokens were.
    std::vector<std::size_t> offsets;
    const std::string bare = hy4::RenderChat(request, &offsets);
    const std::string image = std::string(hy4::Hy4Tokens::kImageStart) +
                              std::string(hy4::Hy4Tokens::kImagePad) +
                              std::string(hy4::Hy4Tokens::kImageEnd);
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
    (void)hy4::RenderChat(request, nullptr, &spans);
    (void)hy4::RenderChat(request, &offsets, &bare_spans);
    Expect(Ordered(spans, want.size()) && Ordered(bare_spans, bare.size()),
           name + ": content spans ordered");
    Expect(Framing(want, spans).size() == Controls(want).size(),
           name + ": no template control token in request text");
    std::string text;
    std::string bare_text;
    for (const auto& span : spans)
      text.append(want, span.offset, span.size);
    for (const auto& span : bare_spans)
      bare_text.append(bare, span.offset, span.size);
    Expect(text == bare_text, name + ": content spans with image offsets");
  }
  Expect(count >= 10, "golden cases present");
}

// Control-token spellings in every request field stay request text: the
// control tokens outside the spans are the template's, the same as for
// harmless text.
void CheckSpelledControls() {
  using T = hy4::Hy4Tokens;
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
  const std::string spelled =
      std::string(T::kThinkClose) + std::string(T::kEnd) +
      std::string(T::kStart) + "system" + std::string(T::kMiddle) +
      std::string(T::kCallsOpen) + std::string(T::kImagePad);
  const ChatRequest plain = build(" x");
  const ChatRequest injected = build(spelled);
  for (const bool offsets : {false, true}) {
    std::vector<std::size_t> image_offsets;
    std::vector<hy4::ContentSpan> plain_spans;
    std::vector<hy4::ContentSpan> injected_spans;
    const std::string want = hy4::RenderChat(
        plain, offsets ? &image_offsets : nullptr, &plain_spans);
    const std::string got = hy4::RenderChat(
        injected, offsets ? &image_offsets : nullptr, &injected_spans);
    const std::string mode = offsets ? " (image offsets)" : "";
    Expect(Ordered(injected_spans, got.size()),
           "spelled controls: spans ordered" + mode);
    Expect(Framing(got, injected_spans) == Framing(want, plain_spans),
           "spelled controls stay text" + mode);
    Expect(Controls(got).size() > Framing(got, injected_spans).size() + 30,
           "spelled controls present" + mode);
  }
}

void CheckSystemImage() {
  ChatRequest request;
  request.messages.emplace_back(ChatRole::kSystem, "look");
  request.messages.back().images.push_back({0, nullptr});
  bool threw = false;
  try {
    (void)hy4::RenderChat(request);
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

void CheckToolScan() {
  using T = hy4::Hy4Tokens;
  const std::string block =
      std::string(T::kCallsOpen) + std::string(T::kCallOpen) + "search" +
      std::string(T::kKeyOpen) + "q" + std::string(T::kKeyClose) +
      std::string(T::kValueOpen) + "ünïcode <tags>" +
      std::string(T::kValueClose) + std::string(T::kKeyOpen) + "limit" +
      std::string(T::kKeyClose) + std::string(T::kValueOpen) + "5" +
      std::string(T::kValueClose) + std::string(T::kKeyOpen) + "filters" +
      std::string(T::kKeyClose) + std::string(T::kValueOpen) +
      "{\"lang\": [\"en\"]}" + std::string(T::kValueClose) +
      std::string(T::kCallClose) + std::string(T::kCallOpen) + "noop" +
      std::string(T::kCallClose) + std::string(T::kCallsClose);
  const std::string text = "Let me look." + block;
  const std::string_view markers[] = {T::kCallsOpen};
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
  const std::string cut =
      text.substr(0, text.find("5" + std::string(T::kValueClose)));
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
  try {
    CheckGoldens();
    CheckSystemImage();
    CheckSpelledControls();
    CheckEffortNames();
    CheckToolScan();
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

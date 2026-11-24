#include "chat_template.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <gufo/ext/json.hpp>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gufo::models::hy4_flash {
namespace {

using ext::ChatMessage;
using ext::ChatRole;

std::string_view RoleName(ChatRole role) {
  switch (role) {
    case ChatRole::kSystem:
    case ChatRole::kDeveloper:
      return "system";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
    case ChatRole::kUser:
      return "user";
  }
  return "user";
}

// The template's tool_to_json: the function object's members except
// `defer_loading` and `strict`, each as `"key": tojson(value)`. Empty for a
// tool with a true `defer_loading`, which the tools prompt leaves out.
std::string ToolJson(const ext::ChatTool& tool) {
  json::Value function;
  if (!tool.definition_json.empty()) {
    json::Value definition = json::parse(tool.definition_json);
    const json::Value* inner = definition.find("function");
    function = inner ? *inner : definition;
  } else {
    function = json::Value::object();
    function["name"] = tool.name;
    if (!tool.description.empty()) function["description"] = tool.description;
    function["parameters"] = json::parse(tool.parameters_json.empty() ? "{}" : tool.parameters_json);
  }
  if (const json::Value* deferred = function.find("defer_loading");
      deferred != nullptr && deferred->as_bool()) {
    return {};
  }
  std::string out = "{";
  bool first = true;
  for (const auto& [key, value] : function.members()) {
    if (key == "defer_loading" || key == "strict") continue;
    if (!first) out += ", ";
    first = false;
    out += json::Value(key).tojson();
    out += ": ";
    out += value.tojson(json::Value::Numbers::kAsParsed);
  }
  out += "}";
  return out;
}

// The rendered prompt and, when asked for, the spans of request text in it.
class Writer {
public:
  explicit Writer(std::vector<ContentSpan>* spans) : spans_(spans) {}

  /// Template framing: its control-token spellings are control tokens.
  void Markup(std::string_view text) { out_ += text; }

  /// Request text: its control-token spellings stay text.
  void Text(std::string_view text) {
    if (text.empty())
      return;
    if (spans_ != nullptr) {
      if (!spans_->empty() &&
          spans_->back().offset + spans_->back().size == out_.size()) {
        spans_->back().size += text.size();
      } else {
        spans_->push_back({out_.size(), text.size()});
      }
    }
    out_ += text;
  }

  std::string& str() { return out_; }

private:
  std::string out_;
  std::vector<ContentSpan>* spans_;
};

void ToolsPrompt(Writer& w, const Hy4Tokens& tokens,
                 const std::vector<ext::ChatTool>& tools) {
  w.Markup(
      "# Tools\n\nYou may call one or more functions to assist with the user "
      "query."
      "\n\nYou are provided with function signatures within <tools></tools> "
      "XML tags:"
      "\n<tools>\n");
  bool first = true;
  for (const auto& tool : tools) {
    const std::string line = ToolJson(tool);
    if (line.empty())
      continue;
    if (!first)
      w.Markup("\n");
    first = false;
    w.Text(line);
  }
  w.Markup("\n</tools>\n\n");
  w.Markup("For function call returns, you should first print ");
  w.Markup(tokens.calls_open);
  w.Markup("\nFor each function call, you should return object like:\n");
  w.Markup(tokens.call_open);
  w.Markup("{function-name}");
  w.Markup(std::string(tokens.key_open) + "{arg-key-1}" +
           std::string(tokens.key_close));
  w.Markup(std::string(tokens.value_open) + "{arg-value-1}" +
           std::string(tokens.value_close));
  w.Markup(std::string(tokens.key_open) + "{arg-key-2}" +
           std::string(tokens.key_close));
  w.Markup(std::string(tokens.value_open) + "{arg-value-2}" +
           std::string(tokens.value_close));
  w.Markup("...");
  w.Markup(tokens.call_close);
  w.Markup("\nAt the end of function call returns, you should print ");
  w.Markup(tokens.calls_close);
}

// The message text with `image` at each image part's offset.
void Content(Writer& w, const ChatMessage& message, std::string_view image) {
  std::vector<std::size_t> offsets;
  for (const auto& part : message.images)
    offsets.push_back(std::min(part.offset, message.content.size()));
  std::ranges::stable_sort(offsets);
  const std::string_view content = message.content;
  std::size_t cursor = 0;
  for (const std::size_t offset : offsets) {
    w.Text(content.substr(cursor, offset - cursor));
    w.Markup(image);
    cursor = offset;
  }
  w.Text(content.substr(cursor));
  w.Text(message.framing_suffix);
}

// A marker no message text contains, for RenderChat's image offsets.
std::string ImageMarker(const ext::ChatRequest& request) {
  static std::atomic<std::uint64_t> counter{std::random_device{}()};
  for (;;) {
    const std::string marker =
        "\x01hy4-image-" + std::to_string(counter++) + "\x01";
    const bool used =
        std::ranges::any_of(request.messages, [&](const ChatMessage& m) {
          return m.content.find(marker) != std::string::npos ||
                 m.framing_suffix.find(marker) != std::string::npos ||
                 m.thought.find(marker) != std::string::npos;
        });
    if (!used)
      return marker;
  }
}

void TurnStart(Writer& w, const Hy4Tokens& tokens, std::string_view role) {
  w.Markup(tokens.start);
  w.Markup(role);
  w.Markup(tokens.middle);
}

// A tool argument value: strings raw, other values as tojson.
std::string ArgumentValue(const ChatMessage::ToolArgument& argument) {
  if (argument.is_string) return argument.value;
  try {
    return json::parse(argument.value).tojson(json::Value::Numbers::kAsParsed);
  } catch (...) {
    return argument.value;
  }
}

// A tag the vocabulary carries between fullwidth bars: `<｜hy_start…｜>`.
std::string PipeToken(std::string_view name, std::string_view suffix) {
  return "<｜" + std::string(name) + std::string(suffix) + "｜>";
}

// A tag between angle brackets, open or closed: `<…>` or `</…>`.
std::string AngleTag(std::string_view name, std::string_view suffix,
                     bool close) {
  return (close ? "</" : "<") + std::string(name) + std::string(suffix) + ">";
}

}  // namespace

Hy4Tokens MakeHy4Tokens(std::string_view suffix) {
  Hy4Tokens tokens;
  tokens.suffix = std::string(suffix);
  tokens.start = PipeToken("hy_start", suffix);
  tokens.middle = PipeToken("hy_middle", suffix);
  tokens.end = PipeToken("hy_end", suffix);
  tokens.think_open = AngleTag("think", suffix, false);
  tokens.think_close = AngleTag("think", suffix, true);
  tokens.calls_open = AngleTag("tool_calls", suffix, false);
  tokens.calls_close = AngleTag("tool_calls", suffix, true);
  tokens.call_open = AngleTag("tool_call", suffix, false);
  tokens.call_close = AngleTag("tool_call", suffix, true);
  tokens.key_open = AngleTag("arg_key", suffix, false);
  tokens.key_close = AngleTag("arg_key", suffix, true);
  tokens.value_open = AngleTag("arg_value", suffix, false);
  tokens.value_close = AngleTag("arg_value", suffix, true);
  tokens.response_open = AngleTag("tool_response", suffix, false);
  tokens.response_close = AngleTag("tool_response", suffix, true);
  tokens.reasoning_mode = PipeToken("reasoning_mode", suffix);
  tokens.image_start = PipeToken("hy_image▁start", suffix);
  tokens.image_pad = PipeToken("hy_image▁pad", suffix);
  tokens.image_end = PipeToken("hy_image▁end", suffix);
  tokens.call_header = tokens.calls_open + tokens.call_open;
  return tokens;
}

const Hy4Tokens& OpensourceHy4Tokens() {
  static const Hy4Tokens tokens = MakeHy4Tokens(":opensource");
  return tokens;
}

std::string Hy4Tokens::SuffixOf(std::string_view start) {
  constexpr std::string_view name = "<｜hy_start";
  constexpr std::string_view tail = "｜>";
  if (!start.starts_with(name) || !start.ends_with(tail) ||
      start.size() < name.size() + tail.size())
    return {};
  return std::string(start.substr(name.size(),
                                  start.size() - name.size() - tail.size()));
}

std::string Hy4TokenSuffix(const core::GgufReader& reader) {
  // The template names the suffix once and builds every tag from it:
  //   {%- set HYTK = ':opensource' %}
  const auto rendered = reader.GetMetadataString("tokenizer.chat_template");
  if (rendered) {
    constexpr std::string_view name = "set HYTK";
    const std::size_t at = rendered->find(name);
    if (at != std::string_view::npos) {
      const std::size_t open = rendered->find_first_of("'\"", at + name.size());
      if (open != std::string_view::npos) {
        const std::size_t close = rendered->find(rendered->at(open), open + 1);
        if (close != std::string_view::npos)
          return std::string(rendered->substr(open + 1, close - open - 1));
      }
    }
  }
  // A template that names none: the vocabulary's own turn start carries it.
  constexpr std::string_view start = "<｜hy_start";
  for (const std::string_view token :
       reader.GetMetadataStringArray("tokenizer.ggml.tokens")) {
    if (token.starts_with(start) && token.ends_with("｜>"))
      return Hy4Tokens::SuffixOf(token);
  }
  return {};
}

std::string_view ReasoningEffortName(const ReasoningOptions& reasoning) {
  if (!reasoning.enabled.value_or(true)) return "no_think";
  const auto effort = reasoning.effort.value_or(ReasoningEffort::kHigh);
  return effort == ReasoningEffort::kMax || effort == ReasoningEffort::kXHigh ? "max" : "high";
}

std::string RenderChat(const Hy4Tokens& tokens, const ext::ChatRequest& request,
                       std::vector<std::size_t>* image_offsets,
                       std::vector<ContentSpan>* content_spans,
                       std::size_t* stable_prefix_bytes) {
  const auto& messages = request.messages;
  const std::string marker = image_offsets ? ImageMarker(request)
                                           : std::string(tokens.image_start) +
                                                 std::string(tokens.image_pad) +
                                                 std::string(tokens.image_end);
  if (content_spans)
    content_spans->clear();
  Writer w(content_spans);
  std::vector<ext::ChatTool> tools;
  if (request.tool_choice != ext::ChatRequest::ToolChoice::kNone) tools = request.tools;
  const std::string_view effort = ReasoningEffortName(request.reasoning);
  const bool no_think = effort == "no_think";
  const bool preserved = request.reasoning.preserve_thinking.value_or(!tools.empty());
  const auto Mode = [&] {
    w.Markup(tokens.reasoning_mode);
    w.Markup("reasoning_effort:");
    w.Markup(effort);
  };

  std::ptrdiff_t last_user = -1;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (messages[i].role == ChatRole::kUser) last_user = static_cast<std::ptrdiff_t>(i);
  }
  // Where the next request's prompt starts to differ: after the first think
  // tag whose reasoning a later user turn drops, else after the generation
  // prompt's turn start. Both follow a special token.
  std::optional<std::size_t> stable;

  const bool leading_system =
      !messages.empty() && (messages[0].role == ChatRole::kSystem || messages[0].role == ChatRole::kDeveloper);
  if (!leading_system) {
    TurnStart(w, tokens, "system");
    if (!tools.empty())
      ToolsPrompt(w, tokens, tools);
    Mode();
    w.Markup(tokens.end);
  }

  // Tool-call ids of the last assistant turn, to order the responses after it.
  std::vector<std::string> call_ids;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    if (m.role == ChatRole::kTool) {
      // Consecutive tool messages form one turn.
      std::size_t end = i;
      while (end < messages.size() && messages[end].role == ChatRole::kTool) ++end;
      std::vector<const ChatMessage*> cluster;
      for (std::size_t j = i; j < end; ++j) cluster.push_back(&messages[j]);
      std::vector<const ChatMessage*> ordered;
      bool matched = !call_ids.empty() && call_ids.size() == cluster.size();
      for (std::size_t k = 0; matched && k < call_ids.size(); ++k) {
        const ChatMessage* hit = nullptr;
        int count = 0;
        for (const ChatMessage* t : cluster) {
          if (t->tool_call_id == call_ids[k]) {
            hit = t;
            ++count;
          }
        }
        if (count == 1) {
          ordered.push_back(hit);
        } else {
          matched = false;
        }
      }
      TurnStart(w, tokens, "tool");
      for (const ChatMessage* t : matched ? ordered : cluster) {
        w.Markup(tokens.response_open);
        Content(w, *t, marker);
        w.Markup(tokens.response_close);
      }
      w.Markup(tokens.end);
      call_ids.clear();
      i = end - 1;
      continue;
    }
    if (m.role == ChatRole::kSystem || m.role == ChatRole::kDeveloper) {
      if (!m.images.empty())
        throw std::invalid_argument(
            "Hy4 Flash system messages cannot contain images");
      TurnStart(w, tokens, "system");
      if (i == 0) {
        if (!tools.empty()) {
          ToolsPrompt(w, tokens, tools);
          if (!m.content.empty() || !m.framing_suffix.empty())
            w.Markup("\n\n");
        }
        Content(w, m, marker);
        Mode();
      } else {
        Content(w, m, marker);
      }
      w.Markup(tokens.end);
    } else if (m.role == ChatRole::kAssistant) {
      TurnStart(w, tokens, "assistant");
      w.Markup(tokens.think_open);
      if (!no_think &&
          (preserved || static_cast<std::ptrdiff_t>(i) > last_user)) {
        if (!preserved && !m.thought.empty() && !stable)
          stable = w.str().size();
        w.Text(m.thought);
      }
      w.Markup(tokens.think_close);
      Content(w, m, marker);
      if (!m.tool_calls.empty()) {
        call_ids.clear();
        bool complete = true;
        for (const auto& call : m.tool_calls) {
          if (call.id.empty()) {
            complete = false;
          } else {
            call_ids.push_back(call.id);
          }
        }
        if (!complete) call_ids.clear();
        w.Markup(tokens.calls_open);
        for (const auto& call : m.tool_calls) {
          w.Markup(tokens.call_open);
          w.Text(call.name);
          for (const auto& argument : call.arguments) {
            w.Markup(tokens.key_open);
            w.Text(argument.name);
            w.Markup(tokens.key_close);
            w.Markup(tokens.value_open);
            w.Text(ArgumentValue(argument));
            w.Markup(tokens.value_close);
          }
          w.Markup(tokens.call_close);
        }
        w.Markup(tokens.calls_close);
      }
      w.Markup(tokens.end);
      if (!m.tool_calls.empty()) continue;  // keep call_ids for the responses
    } else {
      TurnStart(w, tokens, RoleName(m.role));
      Content(w, m, marker);
      w.Markup(tokens.end);
    }
    call_ids.clear();
  }

  if (messages.empty() || messages.back().role != ChatRole::kAssistant) {
    if (!stable)
      stable = w.str().size() + tokens.start.size();
    TurnStart(w, tokens, "assistant");
    w.Markup(tokens.think_open);
    if (no_think)
      w.Markup(tokens.think_close);
  }
  std::string& out = w.str();
  std::size_t stable_bytes = stable.value_or(out.size());
  if (image_offsets) {
    image_offsets->clear();
    std::string clean;
    std::vector<std::size_t> removed;  // marker offsets in `out`
    std::size_t cursor = 0;
    for (std::size_t at = out.find(marker); at != std::string::npos;
         at = out.find(marker, cursor)) {
      clean.append(out, cursor, at - cursor);
      image_offsets->push_back(clean.size());
      removed.push_back(at);
      cursor = at + marker.size();
    }
    clean.append(out, cursor);
    out = std::move(clean);
    // An offset in `out` less the markers that ended at or before it.
    const auto rebase = [&](std::size_t offset) {
      std::size_t gone = 0;
      for (const std::size_t at : removed) {
        if (at + marker.size() > offset)
          break;
        gone += marker.size();
      }
      return offset - gone;
    };
    stable_bytes = rebase(stable_bytes);
    if (content_spans) {
      for (auto& span : *content_spans) {
        const std::size_t begin = rebase(span.offset);
        span.size = rebase(span.offset + span.size) - begin;
        span.offset = begin;
      }
      std::erase_if(*content_spans,
                    [](const ContentSpan& span) { return span.size == 0; });
    }
  }
  if (stable_prefix_bytes)
    *stable_prefix_bytes = stable_bytes;
  return std::move(out);
}

}  // namespace gufo::models::hy4_flash

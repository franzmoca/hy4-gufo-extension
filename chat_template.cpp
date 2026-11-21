#include "chat_template.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <gufo/ext/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gufo::models::hy4_flash {
namespace {

using ext::ChatMessage;
using ext::ChatRole;
using T = Hy4Tokens;

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

void ToolsPrompt(Writer& w, const std::vector<ext::ChatTool>& tools) {
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
  w.Markup(T::kCallsOpen);
  w.Markup("\nFor each function call, you should return object like:\n");
  w.Markup(T::kCallOpen);
  w.Markup("{function-name}");
  w.Markup(std::string(T::kKeyOpen) + "{arg-key-1}" +
           std::string(T::kKeyClose));
  w.Markup(std::string(T::kValueOpen) + "{arg-value-1}" +
           std::string(T::kValueClose));
  w.Markup(std::string(T::kKeyOpen) + "{arg-key-2}" +
           std::string(T::kKeyClose));
  w.Markup(std::string(T::kValueOpen) + "{arg-value-2}" +
           std::string(T::kValueClose));
  w.Markup("...");
  w.Markup(T::kCallClose);
  w.Markup("\nAt the end of function call returns, you should print ");
  w.Markup(T::kCallsClose);
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

void TurnStart(Writer& w, std::string_view role) {
  w.Markup(T::kStart);
  w.Markup(role);
  w.Markup(T::kMiddle);
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

}  // namespace

std::string_view ReasoningEffortName(const ReasoningOptions& reasoning) {
  if (!reasoning.enabled.value_or(true)) return "no_think";
  const auto effort = reasoning.effort.value_or(ReasoningEffort::kHigh);
  return effort == ReasoningEffort::kMax || effort == ReasoningEffort::kXHigh ? "max" : "high";
}

std::string RenderChat(const ext::ChatRequest& request,
                       std::vector<std::size_t>* image_offsets,
                       std::vector<ContentSpan>* content_spans) {
  const auto& messages = request.messages;
  const std::string marker = image_offsets ? ImageMarker(request)
                                           : std::string(T::kImageStart) +
                                                 std::string(T::kImagePad) +
                                                 std::string(T::kImageEnd);
  if (content_spans)
    content_spans->clear();
  Writer w(content_spans);
  std::vector<ext::ChatTool> tools;
  if (request.tool_choice != ext::ChatRequest::ToolChoice::kNone) tools = request.tools;
  const std::string_view effort = ReasoningEffortName(request.reasoning);
  const bool no_think = effort == "no_think";
  const bool preserved = request.reasoning.preserve_thinking.value_or(!tools.empty());
  const auto Mode = [&] {
    w.Markup(T::kReasoningMode);
    w.Markup("reasoning_effort:");
    w.Markup(effort);
  };

  std::ptrdiff_t last_user = -1;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (messages[i].role == ChatRole::kUser) last_user = static_cast<std::ptrdiff_t>(i);
  }

  const bool leading_system =
      !messages.empty() && (messages[0].role == ChatRole::kSystem || messages[0].role == ChatRole::kDeveloper);
  if (!leading_system) {
    TurnStart(w, "system");
    if (!tools.empty())
      ToolsPrompt(w, tools);
    Mode();
    w.Markup(T::kEnd);
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
      TurnStart(w, "tool");
      for (const ChatMessage* t : matched ? ordered : cluster) {
        w.Markup(T::kResponseOpen);
        Content(w, *t, marker);
        w.Markup(T::kResponseClose);
      }
      w.Markup(T::kEnd);
      call_ids.clear();
      i = end - 1;
      continue;
    }
    if (m.role == ChatRole::kSystem || m.role == ChatRole::kDeveloper) {
      if (!m.images.empty())
        throw std::invalid_argument(
            "Hy4 Flash system messages cannot contain images");
      TurnStart(w, "system");
      if (i == 0) {
        if (!tools.empty()) {
          ToolsPrompt(w, tools);
          if (!m.content.empty() || !m.framing_suffix.empty())
            w.Markup("\n\n");
        }
        Content(w, m, marker);
        Mode();
      } else {
        Content(w, m, marker);
      }
      w.Markup(T::kEnd);
    } else if (m.role == ChatRole::kAssistant) {
      TurnStart(w, "assistant");
      w.Markup(T::kThinkOpen);
      if (!no_think &&
          (preserved || static_cast<std::ptrdiff_t>(i) > last_user)) {
        w.Text(m.thought);
      }
      w.Markup(T::kThinkClose);
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
        w.Markup(T::kCallsOpen);
        for (const auto& call : m.tool_calls) {
          w.Markup(T::kCallOpen);
          w.Text(call.name);
          for (const auto& argument : call.arguments) {
            w.Markup(T::kKeyOpen);
            w.Text(argument.name);
            w.Markup(T::kKeyClose);
            w.Markup(T::kValueOpen);
            w.Text(ArgumentValue(argument));
            w.Markup(T::kValueClose);
          }
          w.Markup(T::kCallClose);
        }
        w.Markup(T::kCallsClose);
      }
      w.Markup(T::kEnd);
      if (!m.tool_calls.empty()) continue;  // keep call_ids for the responses
    } else {
      TurnStart(w, RoleName(m.role));
      Content(w, m, marker);
      w.Markup(T::kEnd);
    }
    call_ids.clear();
  }

  if (messages.empty() || messages.back().role != ChatRole::kAssistant) {
    TurnStart(w, "assistant");
    w.Markup(T::kThinkOpen);
    if (no_think)
      w.Markup(T::kThinkClose);
  }
  std::string& out = w.str();
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
    if (content_spans) {
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
      for (auto& span : *content_spans) {
        const std::size_t begin = rebase(span.offset);
        span.size = rebase(span.offset + span.size) - begin;
        span.offset = begin;
      }
      std::erase_if(*content_spans,
                    [](const ContentSpan& span) { return span.size == 0; });
    }
  }
  return std::move(out);
}

}  // namespace gufo::models::hy4_flash

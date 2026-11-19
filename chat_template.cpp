#include "chat_template.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>

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
// `defer_loading` and `strict`, each as `"key": tojson(value)`.
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

std::string ToolsPrompt(const std::vector<ext::ChatTool>& tools) {
  std::string out =
      "# Tools\n\nYou may call one or more functions to assist with the user query."
      "\n\nYou are provided with function signatures within <tools></tools> XML tags:"
      "\n<tools>\n";
  for (std::size_t i = 0; i < tools.size(); ++i) {
    if (i) out += "\n";
    out += ToolJson(tools[i]);
  }
  out += "\n</tools>\n\n";
  out += "For function call returns, you should first print ";
  out += T::kCallsOpen;
  out += "\nFor each function call, you should return object like:\n";
  out += T::kCallOpen;
  out += "{function-name}";
  out += std::string(T::kKeyOpen) + "{arg-key-1}" + std::string(T::kKeyClose);
  out += std::string(T::kValueOpen) + "{arg-value-1}" + std::string(T::kValueClose);
  out += std::string(T::kKeyOpen) + "{arg-key-2}" + std::string(T::kKeyClose);
  out += std::string(T::kValueOpen) + "{arg-value-2}" + std::string(T::kValueClose);
  out += "...";
  out += T::kCallClose;
  out += "\nAt the end of function call returns, you should print ";
  out += T::kCallsClose;
  return out;
}

std::string Content(const ChatMessage& message) { return message.content + message.framing_suffix; }

void Turn(std::string& out, std::string_view role, std::string_view body) {
  out += T::kStart;
  out += role;
  out += T::kMiddle;
  out += body;
  out += T::kEnd;
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

std::string RenderChat(const ext::ChatRequest& request) {
  const auto& messages = request.messages;
  std::vector<ext::ChatTool> tools;
  if (request.tool_choice != ext::ChatRequest::ToolChoice::kNone) tools = request.tools;
  const std::string_view effort = ReasoningEffortName(request.reasoning);
  const bool no_think = effort == "no_think";
  const bool preserved = request.reasoning.preserve_thinking.value_or(!tools.empty());
  const std::string mode = std::string(T::kReasoningMode) + "reasoning_effort:" + std::string(effort);

  std::ptrdiff_t last_user = -1;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (messages[i].role == ChatRole::kUser) last_user = static_cast<std::ptrdiff_t>(i);
  }

  std::string out;
  const bool leading_system =
      !messages.empty() && (messages[0].role == ChatRole::kSystem || messages[0].role == ChatRole::kDeveloper);
  if (!leading_system) {
    Turn(out, "system", (tools.empty() ? std::string() : ToolsPrompt(tools)) + mode);
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
      std::string body;
      for (const ChatMessage* t : matched ? ordered : cluster) {
        body += T::kResponseOpen;
        body += Content(*t);
        body += T::kResponseClose;
      }
      Turn(out, "tool", body);
      call_ids.clear();
      i = end - 1;
      continue;
    }
    if (m.role == ChatRole::kSystem || m.role == ChatRole::kDeveloper) {
      std::string body;
      if (i == 0) {
        if (!tools.empty()) body += ToolsPrompt(tools);
        const std::string content = Content(m);
        if (!tools.empty() && !content.empty()) body += "\n\n";
        body += content;
        body += mode;
      } else {
        body = Content(m);
      }
      Turn(out, "system", body);
    } else if (m.role == ChatRole::kAssistant) {
      std::string body(T::kThinkOpen);
      if (!no_think && (preserved || static_cast<std::ptrdiff_t>(i) > last_user) && !m.thought.empty()) {
        body += m.thought;
      }
      body += T::kThinkClose;
      body += Content(m);
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
        body += T::kCallsOpen;
        for (const auto& call : m.tool_calls) {
          body += T::kCallOpen;
          body += call.name;
          for (const auto& argument : call.arguments) {
            body += T::kKeyOpen;
            body += argument.name;
            body += T::kKeyClose;
            body += T::kValueOpen;
            body += ArgumentValue(argument);
            body += T::kValueClose;
          }
          body += T::kCallClose;
        }
        body += T::kCallsClose;
      }
      Turn(out, "assistant", body);
      if (!m.tool_calls.empty()) continue;  // keep call_ids for the responses
    } else {
      Turn(out, RoleName(m.role), Content(m));
    }
    call_ids.clear();
  }

  if (messages.empty() || messages.back().role != ChatRole::kAssistant) {
    out += T::kStart;
    out += "assistant";
    out += T::kMiddle;
    out += T::kThinkOpen;
    if (no_think) out += T::kThinkClose;
  }
  return out;
}

}  // namespace gufo::models::hy4_flash

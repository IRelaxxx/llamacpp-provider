import {
  LanguageModelV3DataContent,
  LanguageModelV3Prompt,
  LanguageModelV3ToolResultOutput,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import { convertToBase64 } from "@ai-sdk/provider-utils";

type LlamacppToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

type LlamacppUserPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

type LlamacppChatMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: LlamacppUserPart[] }
  | { role: "assistant"; content: string; tool_calls?: LlamacppToolCall[] }
  | {
      role: "tool";
      name: string;
      tool_call_id: string;
      content: string;
    };

function formatFileUrl({
  data,
  mediaType,
}: {
  data: LanguageModelV3DataContent;
  mediaType: string;
}) {
  return data instanceof URL
    ? data.toString()
    : `data:${mediaType};base64,${convertToBase64(data as Uint8Array)}`;
}

function stringifyToolOutput(output: LanguageModelV3ToolResultOutput | string) {
  if (typeof output === "string") {
    return output;
  }

  switch (output?.type) {
    case "text":
    case "error-text":
      return output.value;
    case "execution-denied":
      return output.reason ?? "Tool execution denied.";
    case "content":
    case "json":
    case "error-json":
      return JSON.stringify(output.value);
    default:
      return JSON.stringify(output ?? null);
  }
}

export function convertToLlamacppChatMessages(prompt: LanguageModelV3Prompt) {
  const messages: LlamacppChatMessage[] = [];

  for (const { role, content } of prompt) {
    switch (role) {
      case "system": {
        messages.push({ role: "system", content: String(content) });
        break;
      }

      case "user": {
        messages.push({
          role: "user",
          content: content.map((part) => {
            switch (part.type) {
              case "text":
                return { type: "text", text: part.text };
              case "file": {
                if (!part.mediaType.startsWith("image/")) {
                  throw new UnsupportedFunctionalityError({
                    functionality: "Only image file parts are supported",
                  });
                }

                const mediaType =
                  part.mediaType === "image/*" ? "image/jpeg" : part.mediaType;

                return {
                  type: "image_url",
                  image_url: {
                    url: formatFileUrl({ data: part.data, mediaType }),
                  },
                };
              }
              default:
                throw new UnsupportedFunctionalityError({
                  functionality: "Unsupported user content type",
                });
            }
          }),
        });
        break;
      }

      case "assistant": {
        let text = "";
        const toolCalls: LlamacppToolCall[] = [];

        for (const part of content) {
          switch (part.type) {
            case "text":
            case "reasoning":
              text += part.text;
              break;
            case "tool-call":
              toolCalls.push({
                id: part.toolCallId,
                type: "function",
                function: {
                  name: part.toolName,
                  arguments: JSON.stringify(part.input),
                },
              });
              break;
            default:
              throw new UnsupportedFunctionalityError({
                functionality: `Unsupported assistant content type: ${part.type}`,
              });
          }
        }

        messages.push({
          role: "assistant",
          content: text,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        });
        break;
      }

      case "tool": {
        for (const toolResponse of content) {
          if (toolResponse.type === "tool-approval-response") {
            continue;
          }

          messages.push({
            role: "tool",
            name: toolResponse.toolName,
            tool_call_id: toolResponse.toolCallId,
            content: stringifyToolOutput(toolResponse.output),
          });
        }
        break;
      }

      default: {
        throw new Error("Unsupported role");
      }
    }
  }

  return messages;
}

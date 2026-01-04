import type { LanguageModelV3FinishReason } from "@ai-sdk/provider";

export function mapLlamacppFinishReason(
  stopType: string | null | undefined
): LanguageModelV3FinishReason {
  switch (stopType) {
    case "eos":
      return { unified: "stop", raw: "eos" };
    case "limit":
      return { unified: "length", raw: "limit" };
    case "word":
      return { unified: "stop", raw: "word" };
    case "none":
      return { unified: "other", raw: "none" };
    default:
      return { unified: "other", raw: "unknown" };
  }
}

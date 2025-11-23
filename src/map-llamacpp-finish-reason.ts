import type { LanguageModelV3FinishReason } from '@ai-sdk/provider';

export function mapLlamacppFinishReason(
  stopType: string | null | undefined,
): LanguageModelV3FinishReason {
  switch (stopType) {
    case 'eos':
      return 'stop';
    case 'limit':
      return 'length';
    case 'word':
      return 'stop';
    case 'none':
      return 'unknown';
    default:
      return 'unknown';
  }
}



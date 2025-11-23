import { z } from 'zod/v4';

export type LlamacppChatModelId = string & {};

export const llamacppLanguageModelOptions = z.object({
  temperature: z.number().optional(),
  topP: z.number().optional(),
  topK: z.number().optional(),
  seed: z.number().optional(),
  mirostat: z.number().optional(),
  mirostatTau: z.number().optional(),
  mirostatEta: z.number().optional(),
  repeatPenalty: z.number().optional(),
});

export type LlamacppLanguageModelOptions = z.infer<
  typeof llamacppLanguageModelOptions
>;



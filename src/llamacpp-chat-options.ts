import { z } from "zod/v4";

export type LlamacppChatModelId = string & {};

const loraAdapterSchema = z.object({
  id: z.number(),
  scale: z.number(),
});

export const llamacppLanguageModelOptions = z.object({
  // basic sampling
  temperature: z.number().optional(),
  topP: z.number().optional(),
  topK: z.number().optional(),
  seed: z.number().optional(),

  // sampling / decoding
  nPredict: z.number().optional(),
  dynatempRange: z.number().optional(),
  dynatempExponent: z.number().optional(),
  minP: z.number().optional(),
  topNSigma: z.number().optional(),
  xtcProbability: z.number().optional(),
  xtcThreshold: z.number().optional(),
  typicalP: z.number().optional(),
  repeatLastN: z.number().optional(),
  dryMultiplier: z.number().optional(),
  dryBase: z.number().optional(),
  dryAllowedLength: z.number().optional(),
  dryPenaltyLastN: z.number().optional(),
  drySequenceBreakers: z.array(z.string()).optional(),
  mirostat: z.number().optional(),
  mirostatTau: z.number().optional(),
  mirostatEta: z.number().optional(),
  minKeep: z.number().optional(),
  nProbs: z.number().optional(),
  samplers: z.array(z.string()).optional(),
  postSamplingProbs: z.boolean().optional(),

  // control & constraints
  grammar: z.string().optional(),
  jsonSchema: z.unknown().optional(),
  logitBias: z
    .union([
      z.record(z.string(), z.number().or(z.boolean())),
      z.array(
        z.tuple([
          z.union([z.number(), z.string()]),
          z.union([z.number(), z.boolean()]),
        ])
      ),
    ])
    .optional(),
  ignoreEos: z.boolean().optional(),
  tMaxPredictMs: z.number().optional(),
  stop: z.array(z.string()).optional(),
  nKeep: z.number().optional(),
  nIndent: z.number().optional(),
  presencePenalty: z.number().optional(),
  frequencyPenalty: z.number().optional(),

  // execution / caching
  idSlot: z.number().optional(),
  cachePrompt: z.boolean().optional(),
  returnTokens: z.boolean().optional(),
  timingsPerToken: z.boolean().optional(),
  returnProgress: z.boolean().optional(),

  // LoRA & advanced
  lora: z.array(loraAdapterSchema).optional(),
  responseFields: z.array(z.string()).optional(),

  // request-level escape hatch (merged last into the body)
  extraParams: z.record(z.string(), z.unknown()).optional(),

  // legacy convenience
  repeatPenalty: z.number().optional(),
});

export type LlamacppLanguageModelOptions = z.infer<
  typeof llamacppLanguageModelOptions
>;

import { z } from "zod/v4";

export type LlamacppEmbeddingModelId = string & {};

export const llamacppEmbeddingOptions = z.object({
  maxEmbeddingsPerCall: z.number().optional(),
  embdNormalize: z.number().optional(),
});

export type LlamacppEmbeddingOptions = z.infer<typeof llamacppEmbeddingOptions>;

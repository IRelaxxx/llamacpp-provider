import type { EmbeddingModelV3 } from "@ai-sdk/provider";
import { TooManyEmbeddingValuesForCallError } from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonResponseHandler,
  parseProviderOptions,
  type FetchFunction,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import {
  llamacppEmbeddingOptions,
  type LlamacppEmbeddingModelId,
} from "./llamacpp-embedding-options";
import { llamacppFailedResponseHandler } from "./llamacpp-error";

type LlamacppEmbeddingConfig = {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
};

type EmbeddingResponse = z.infer<typeof llamacppEmbeddingResponseSchema>;

export class LlamacppEmbeddingModel implements EmbeddingModelV3 {
  readonly specificationVersion = "v3";
  readonly modelId: LlamacppEmbeddingModelId;
  readonly maxEmbeddingsPerCall = 32;
  readonly supportsParallelCalls = false;

  private readonly config: LlamacppEmbeddingConfig;

  constructor(
    modelId: LlamacppEmbeddingModelId,
    config: LlamacppEmbeddingConfig
  ) {
    this.modelId = modelId;
    this.config = config;
  }

  get provider() {
    return this.config.provider;
  }

  async doEmbed({
    values,
    abortSignal,
    headers,
    providerOptions,
  }: Parameters<EmbeddingModelV3["doEmbed"]>[0]): Promise<
    Awaited<ReturnType<EmbeddingModelV3["doEmbed"]>>
  > {
    if (values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values,
      });
    }

    const llamacppOptions =
      (await parseProviderOptions({
        provider: "llamacpp",
        providerOptions,
        schema: llamacppEmbeddingOptions,
      })) ?? {};

    const body: Record<string, unknown> = {
      model: this.modelId,
      input: values,
      encoding_format: "float",
    };

    if (llamacppOptions.embdNormalize != null) {
      body.embd_normalize = llamacppOptions.embdNormalize;
    }

    const {
      responseHeaders,
      value: response,
      rawValue,
    } = await postJsonToApi<EmbeddingResponse>({
      url: `${this.config.baseURL}/embeddings`,
      headers: combineHeaders(this.config.headers(), headers),
      body,
      failedResponseHandler: llamacppFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        llamacppEmbeddingResponseSchema
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    return {
      embeddings: response.data.map((item) => item.embedding),
      usage: response.usage
        ? { tokens: response.usage.prompt_tokens }
        : undefined,
      response: { headers: responseHeaders, body: rawValue },
      warnings: [],
    };
  }
}

const llamacppEmbeddingResponseSchema = z.object({
  data: z.array(z.object({ embedding: z.array(z.number()) })),
  usage: z
    .object({
      prompt_tokens: z.number(),
    })
    .nullish(),
});

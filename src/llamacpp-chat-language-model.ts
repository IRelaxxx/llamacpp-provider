import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
  SharedV3Warning,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  parseProviderOptions,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import type { FetchFunction, ParseResult } from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import type { LlamacppChatModelId } from "./llamacpp-chat-options";
import { llamacppLanguageModelOptions } from "./llamacpp-chat-options";
import { llamacppFailedResponseHandler } from "./llamacpp-error";
import { mapLlamacppFinishReason } from "./map-llamacpp-finish-reason";

type LlamacppChatConfig = {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
  generateId: () => string;
};

type CompletionResponse = z.infer<typeof llamacppCompletionResponseSchema>;
type CompletionChunk = z.infer<typeof llamacppCompletionChunkSchema>;

export class LlamacppChatLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3";
  readonly modelId: LlamacppChatModelId;

  private readonly config: LlamacppChatConfig;

  constructor(modelId: LlamacppChatModelId, config: LlamacppChatConfig) {
    this.modelId = modelId;
    this.config = config;
  }

  get provider() {
    return this.config.provider;
  }

  readonly supportedUrls: Record<string, RegExp[]> = {};

  private async getArgs(options: LanguageModelV3CallOptions): Promise<{
    args: Record<string, unknown>;
    warnings: SharedV3Warning[];
  }> {
    const {
      prompt,
      maxOutputTokens,
      temperature,
      topP,
      topK,
      frequencyPenalty,
      presencePenalty,
      stopSequences,
      seed,
      providerOptions,
      tools,
      toolChoice,
    } = options;

    const warnings: SharedV3Warning[] = [];

    const llamacppOptions =
      (await parseProviderOptions({
        provider: "llamacpp",
        providerOptions,
        schema: llamacppLanguageModelOptions,
      })) ?? {};

    if (tools != null || toolChoice != null) {
      warnings.push({
        type: "unsupported",
        feature: "tools",
      });
    }

    const promptText = prompt
      .map((message) => {
        if (message.role === "system") {
          return message.content;
        }

        const parts = Array.isArray(message.content)
          ? message.content
          : [{ type: "text", text: String(message.content) }];

        return parts
          .filter((part) => part.type === "text")
          .map((part) => ("text" in part ? part.text : ""))
          .join("");
      })
      .join("\n");

    const args: Record<string, unknown> = {
      prompt: promptText,
      n_predict: llamacppOptions.nPredict ?? maxOutputTokens ?? -1,
      temperature: llamacppOptions.temperature ?? temperature,
      top_p: llamacppOptions.topP ?? topP,
      top_k: llamacppOptions.topK ?? topK,
      stop: llamacppOptions.stop ?? stopSequences,
      seed: llamacppOptions.seed ?? seed,
    };

    if (llamacppOptions.mirostat != null) {
      args.mirostat = llamacppOptions.mirostat;
    }
    if (llamacppOptions.mirostatTau != null) {
      args.mirostat_tau = llamacppOptions.mirostatTau;
    }
    if (llamacppOptions.mirostatEta != null) {
      args.mirostat_eta = llamacppOptions.mirostatEta;
    }
    if (llamacppOptions.repeatPenalty != null) {
      args.repeat_penalty = llamacppOptions.repeatPenalty;
    }

    const presence = llamacppOptions.presencePenalty ?? presencePenalty;
    if (presence != null) {
      args.presence_penalty = presence;
    }

    const frequency = llamacppOptions.frequencyPenalty ?? frequencyPenalty;
    if (frequency != null) {
      args.frequency_penalty = frequency;
    }

    if (llamacppOptions.dynatempRange != null) {
      args.dynatemp_range = llamacppOptions.dynatempRange;
    }
    if (llamacppOptions.dynatempExponent != null) {
      args.dynatemp_exponent = llamacppOptions.dynatempExponent;
    }
    if (llamacppOptions.minP != null) {
      args.min_p = llamacppOptions.minP;
    }
    if (llamacppOptions.topNSigma != null) {
      args.top_nsigma = llamacppOptions.topNSigma;
    }
    if (llamacppOptions.xtcProbability != null) {
      args.xtc_probability = llamacppOptions.xtcProbability;
    }
    if (llamacppOptions.xtcThreshold != null) {
      args.xtc_threshold = llamacppOptions.xtcThreshold;
    }
    if (llamacppOptions.typicalP != null) {
      args.typical_p = llamacppOptions.typicalP;
    }
    if (llamacppOptions.repeatLastN != null) {
      args.repeat_last_n = llamacppOptions.repeatLastN;
    }
    if (llamacppOptions.dryMultiplier != null) {
      args.dry_multiplier = llamacppOptions.dryMultiplier;
    }
    if (llamacppOptions.dryBase != null) {
      args.dry_base = llamacppOptions.dryBase;
    }
    if (llamacppOptions.dryAllowedLength != null) {
      args.dry_allowed_length = llamacppOptions.dryAllowedLength;
    }
    if (llamacppOptions.dryPenaltyLastN != null) {
      args.dry_penalty_last_n = llamacppOptions.dryPenaltyLastN;
    }
    if (llamacppOptions.drySequenceBreakers != null) {
      args.dry_sequence_breakers = llamacppOptions.drySequenceBreakers;
    }
    if (llamacppOptions.minKeep != null) {
      args.min_keep = llamacppOptions.minKeep;
    }
    if (llamacppOptions.nProbs != null) {
      args.n_probs = llamacppOptions.nProbs;
    }
    if (llamacppOptions.samplers != null) {
      args.samplers = llamacppOptions.samplers;
    }
    if (llamacppOptions.postSamplingProbs != null) {
      args.post_sampling_probs = llamacppOptions.postSamplingProbs;
    }
    if (llamacppOptions.grammar != null) {
      args.grammar = llamacppOptions.grammar;
    }
    if (llamacppOptions.jsonSchema != null) {
      args.json_schema = llamacppOptions.jsonSchema;
    }
    if (llamacppOptions.logitBias != null) {
      args.logit_bias = llamacppOptions.logitBias;
    }
    if (llamacppOptions.ignoreEos != null) {
      args.ignore_eos = llamacppOptions.ignoreEos;
    }
    if (llamacppOptions.tMaxPredictMs != null) {
      args.t_max_predict_ms = llamacppOptions.tMaxPredictMs;
    }
    if (llamacppOptions.nKeep != null) {
      args.n_keep = llamacppOptions.nKeep;
    }
    if (llamacppOptions.nIndent != null) {
      args.n_indent = llamacppOptions.nIndent;
    }
    if (llamacppOptions.idSlot != null) {
      args.id_slot = llamacppOptions.idSlot;
    }
    if (llamacppOptions.cachePrompt != null) {
      args.cache_prompt = llamacppOptions.cachePrompt;
    }
    if (llamacppOptions.returnTokens != null) {
      args.return_tokens = llamacppOptions.returnTokens;
    }
    if (llamacppOptions.timingsPerToken != null) {
      args.timings_per_token = llamacppOptions.timingsPerToken;
    }
    if (llamacppOptions.returnProgress != null) {
      args.return_progress = llamacppOptions.returnProgress;
    }
    if (llamacppOptions.lora != null) {
      args.lora = llamacppOptions.lora;
    }
    if (llamacppOptions.responseFields != null) {
      args.response_fields = llamacppOptions.responseFields;
    }

    if (llamacppOptions.extraParams) {
      Object.assign(args, llamacppOptions.extraParams);
    }

    return { args, warnings };
  }

  async doGenerate(options: LanguageModelV3CallOptions) {
    const { args, warnings } = await this.getArgs(options);

    const {
      responseHeaders,
      value: response,
      rawValue,
    } = await postJsonToApi<CompletionResponse>({
      url: `${this.config.baseURL}/completion`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body: args,
      failedResponseHandler: llamacppFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        llamacppCompletionResponseSchema
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const content: LanguageModelV3Content[] = [];

    if (response.content) {
      content.push({ type: "text", text: response.content });
    }

    const finishReason = mapLlamacppFinishReason(response.stop_type);

    const usage: LanguageModelV3Usage = {
      inputTokens: {
        total: response.tokens_evaluated,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: response.tokens_predicted,
        text: response.tokens_predicted,
        reasoning: 0,
      },
    };

    return {
      content,
      finishReason,
      usage,
      request: { body: args },
      response: { headers: responseHeaders, body: rawValue },
      warnings,
    };
  }

  async doStream(options: LanguageModelV3CallOptions) {
    const { args, warnings } = await this.getArgs(options);
    const body = { ...args, stream: true };

    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL}/completion`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: llamacppFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        llamacppCompletionChunkSchema
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    let finishReason: LanguageModelV3FinishReason = {
      unified: "other",
      raw: "unknown",
    };
    const usage: LanguageModelV3Usage = {
      inputTokens: {
        total: undefined,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: undefined,
        text: undefined,
        reasoning: 0,
      },
    };

    const stream = response.pipeThrough(
      new TransformStream<
        ParseResult<CompletionChunk>,
        LanguageModelV3StreamPart
      >({
        start(controller) {
          controller.enqueue({ type: "stream-start", warnings });
        },
        transform(chunk, controller) {
          if (!chunk.success) {
            controller.enqueue({ type: "error", error: chunk.error });
            return;
          }

          const value = chunk.value;

          if (value.timings) {
            usage.inputTokens.total = value.tokens_evaluated;
            usage.outputTokens.total = value.tokens_predicted;
          }

          if (value.content) {
            controller.enqueue({
              type: "text-delta",
              id: "0",
              delta: value.content,
            });
          }

          if (value.stop_type) {
            finishReason = mapLlamacppFinishReason(value.stop_type);
          }
        },
        flush(controller) {
          controller.enqueue({ type: "text-end", id: "0" });
          controller.enqueue({ type: "finish", finishReason, usage });
        },
      })
    );

    return {
      stream,
      request: { body },
      response: { headers: responseHeaders },
    };
  }
}

const llamacppCompletionTimingsSchema = z
  .object({
    predicted_n: z.number().optional(),
  })
  .optional();

const llamacppCompletionResponseSchema = z.object({
  content: z.string(),
  stop_type: z.string().nullish(),
  tokens_evaluated: z.number().optional(),
  tokens_predicted: z.number().optional(),
  timings: llamacppCompletionTimingsSchema,
});

const llamacppCompletionChunkSchema = z.object({
  content: z.string().optional(),
  stop_type: z.string().nullish(),
  tokens_evaluated: z.number().optional(),
  tokens_predicted: z.number().optional(),
  timings: llamacppCompletionTimingsSchema,
});

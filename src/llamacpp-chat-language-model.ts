import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3CallWarning,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  parseProviderOptions,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import type { FetchFunction, ParseResult } from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import type { LlamacppChatModelId } from './llamacpp-chat-options';
import { llamacppLanguageModelOptions } from './llamacpp-chat-options';
import { llamacppFailedResponseHandler } from './llamacpp-error';
import { mapLlamacppFinishReason } from './map-llamacpp-finish-reason';

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
  readonly specificationVersion = 'v3';
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

  private async getArgs(
    options: LanguageModelV3CallOptions,
  ): Promise<{
    args: Record<string, unknown>;
    warnings: LanguageModelV3CallWarning[];
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

    const warnings: LanguageModelV3CallWarning[] = [];

    const llamacppOptions =
      (await parseProviderOptions({
        provider: 'llamacpp',
        providerOptions,
        schema: llamacppLanguageModelOptions,
      })) ?? {};

    if (frequencyPenalty != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'frequencyPenalty',
      });
    }

    if (presencePenalty != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'presencePenalty',
      });
    }

    if (tools != null || toolChoice != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'tools',
      });
    }

    const promptText = prompt
      .map(message => {
        if (message.role === 'system') {
          return message.content;
        }

        const parts = Array.isArray(message.content)
          ? message.content
          : [{ type: 'text', text: String(message.content) }];

        return parts
          .filter(part => part.type === 'text')
          .map(part => 'text' in part ? part.text : '')
          .join('');
      })
      .join('\n');

    const args: Record<string, unknown> = {
      prompt: promptText,
      n_predict: maxOutputTokens ?? -1,
      temperature: llamacppOptions.temperature ?? temperature,
      top_p: llamacppOptions.topP ?? topP,
      top_k: llamacppOptions.topK ?? topK,
      stop: stopSequences,
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
        llamacppCompletionResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const content: LanguageModelV3Content[] = [];

    if (response.content) {
      content.push({ type: 'text', text: response.content });
    }

    const finishReason = mapLlamacppFinishReason(response.stop_type);

    const usage: LanguageModelV3Usage = {
      inputTokens: response.tokens_evaluated,
      outputTokens: response.tokens_predicted,
      totalTokens:
        response.tokens_evaluated != null && response.tokens_predicted != null
          ? response.tokens_evaluated + response.tokens_predicted
          : undefined,
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

    const { responseHeaders, value: response } = await postJsonToApi<
      CompletionChunk
    >({
      url: `${this.config.baseURL}/completion`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: llamacppFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        llamacppCompletionChunkSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    let finishReason: LanguageModelV3FinishReason = 'unknown';
    const usage: LanguageModelV3Usage = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    };

    const stream = response.pipeThrough(
      new TransformStream<
        ParseResult<CompletionChunk>,
        LanguageModelV3StreamPart
      >({
        start(controller) {
          controller.enqueue({ type: 'stream-start', warnings });
        },
        transform(chunk, controller) {
          if (!chunk.success) {
            controller.enqueue({ type: 'error', error: chunk.error });
            return;
          }

          const value = chunk.value;

          if (value.timings) {
            usage.inputTokens = value.tokens_evaluated;
            usage.outputTokens = value.tokens_predicted;
            usage.totalTokens =
              value.tokens_evaluated != null && value.tokens_predicted != null
                ? value.tokens_evaluated + value.tokens_predicted
                : undefined;
          }

          if (value.content) {
            controller.enqueue({
              type: 'text-delta',
              id: '0',
              delta: value.content,
            });
          }

          if (value.stop_type) {
            finishReason = mapLlamacppFinishReason(value.stop_type);
          }
        },
        flush(controller) {
          controller.enqueue({ type: 'text-end', id: '0' });
          controller.enqueue({ type: 'finish', finishReason, usage });
        },
      }),
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



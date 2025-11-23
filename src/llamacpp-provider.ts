import {
  EmbeddingModelV3,
  LanguageModelV3,
  NoSuchModelError,
  ProviderV3,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  generateId,
  loadApiKey,
  withoutTrailingSlash,
  withUserAgentSuffix,
} from '@ai-sdk/provider-utils';
import { LlamacppChatLanguageModel } from './llamacpp-chat-language-model';
import type { LlamacppChatModelId } from './llamacpp-chat-options';
import { LlamacppEmbeddingModel } from './llamacpp-embedding-model';
import type { LlamacppEmbeddingModelId } from './llamacpp-embedding-options';
import { VERSION } from './version';

export interface LlamacppProvider extends ProviderV3 {
  (modelId: LlamacppChatModelId): LanguageModelV3;

  languageModel(modelId: LlamacppChatModelId): LanguageModelV3;
  chat(modelId: LlamacppChatModelId): LanguageModelV3;

  embedding(modelId: LlamacppEmbeddingModelId): EmbeddingModelV3<string>;
  textEmbedding(modelId: LlamacppEmbeddingModelId): EmbeddingModelV3<string>;
  textEmbeddingModel(
    modelId: LlamacppEmbeddingModelId,
  ): EmbeddingModelV3<string>;
}

export interface LlamacppProviderSettings {
  baseURL?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  fetch?: FetchFunction;
  generateId?: () => string;
}

export function createLlamacpp(
  options: LlamacppProviderSettings = {},
): LlamacppProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL) ?? 'http://127.0.0.1:8080';

  const getHeaders = () => {
    const base: Record<string, string> = {
      ...options.headers,
    };

    const apiKey = loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: 'LLAMACPP_API_KEY',
      description: 'llama.cpp',
      isOptional: true,
    });

    if (apiKey) {
      base.Authorization = `Bearer ${apiKey}`;
    }

    return withUserAgentSuffix(
      base,
      `ai-sdk/llamacpp/${VERSION}`,
    );
  };

  const createChatModel = (modelId: LlamacppChatModelId) =>
    new LlamacppChatLanguageModel(modelId, {
      provider: 'llamacpp.chat',
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
      generateId: options.generateId ?? generateId,
    });

  const createEmbeddingModel = (modelId: LlamacppEmbeddingModelId) =>
    new LlamacppEmbeddingModel(modelId, {
      provider: 'llamacpp.embedding',
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const provider = function (modelId: LlamacppChatModelId) {
    // eslint-disable-next-line @typescript-eslint/no-invalid-this
    if (new.target) {
      throw new Error(
        'The LLaMA.cpp model function cannot be called with the new keyword.',
      );
    }

    return createChatModel(modelId);
  } as LlamacppProvider;

  provider.specificationVersion = 'v3';
  provider.languageModel = createChatModel;
  provider.chat = createChatModel;
  provider.embedding = createEmbeddingModel;
  provider.textEmbedding = createEmbeddingModel;
  provider.textEmbeddingModel = createEmbeddingModel;

  provider.imageModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: 'imageModel' });
  };

  return provider;
}

export const llamacpp = createLlamacpp();



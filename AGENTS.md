# Repository Guide for Agents

## General Rules

- **Conciseness:** Be extremely concise.
- **Types:** Do NOT write explicit return types unless necessary.

You have access to:

- the llama-cpp servers readme in ./references/llama-server-readme.md
- ai sdk custom provider instructions in ./references/01-custom-providers.mdx
- the reference provider implementation in ./references/mistral

When you are searching the codebase, be very careful that you do not read too much at once. Only read a small amount at a time as you're searching, avoid reading dozens of files at once...

## LLaMA.cpp provider options

You can pass advanced llama.cpp generation options via `providerOptions.llamacpp` on AI SDK calls, for example:

```ts
const result = await generateText({
  model: llamacpp('gpt-4o-mini'),
  prompt: 'Hello',
  providerOptions: {
    llamacpp: {
      dynatempRange: 0.3,
      drySequenceBreakers: ['\n', ':'],
      jsonSchema: { type: 'object', properties: { answer: { type: 'string' } } },
      lora: [{ id: 0, scale: 0.5 }],
      extraParams: { custom_flag: 1 },
    },
  },
});
```
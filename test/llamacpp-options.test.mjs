import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLlamacpp } from '../dist/index.mjs';

test('llamacpp providerOptions map to completion body', async () => {
  const completionBodies = [];

  const llamacpp = createLlamacpp({
    baseURL: 'http://localhost',
    apiKey: 'test-key',
    fetch: async (input, init) => {
      const url = String(input);

      if (url.endsWith('/apply-template')) {
        throw new Error('apply-template should be opt-in');
      }

      if (url.endsWith('/completion')) {
        if (init?.body) {
          completionBodies.push(JSON.parse(init.body));
        }

        return new Response(
          JSON.stringify({
            content: '',
            stop_type: null,
            tokens_evaluated: 0,
            tokens_predicted: 0,
            timings: {},
          }),
          {
            status: 200,
            headers: { 'content-type': 'application/json' },
          },
        );
      }

      throw new Error(`Unexpected URL: ${url}`);
    },
  });

  const model = llamacpp.languageModel('test-model');

  await model.doGenerate({
    prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    maxOutputTokens: 16,
    temperature: 0.5,
    providerOptions: {
      llamacpp: {
        nPredict: 42,
        dynatempRange: 0.3,
        drySequenceBreakers: ['\n', ':'],
        jsonSchema: { type: 'object', properties: { answer: { type: 'string' } } },
        logitBias: { Hello: -1 },
        lora: [{ id: 0, scale: 0.5 }],
        prefill: ' should-not-apply',
        extraParams: { custom_flag: 1 },
      },
    },
  });

  assert.equal(completionBodies.length, 1);
  const body = completionBodies[0];

  assert.equal(body.n_predict, 42);
  assert.equal(body.model, 'test-model');
  assert.equal(body.prompt, 'Hello');
  assert.equal(body.temperature, 0.5);
  assert.equal(body.dynatemp_range, 0.3);
  assert.deepEqual(body.dry_sequence_breakers, ['\n', ':']);
  assert.deepEqual(body.json_schema, {
    type: 'object',
    properties: { answer: { type: 'string' } },
  });
  assert.deepEqual(body.logit_bias, { Hello: -1 });
  assert.deepEqual(body.lora, [{ id: 0, scale: 0.5 }]);
  assert.equal(body.custom_flag, 1);

  const firstResult = await model.doGenerate({
    prompt: [{ role: 'user', content: [{ type: 'text', text: 'warn me' }] }],
    providerOptions: {
      llamacpp: {
        prefill: 'prefill-without-template',
      },
    },
  });

  assert.ok(
    firstResult.warnings.some(
      (warning) =>
        warning.type === 'unsupported' && warning.feature === 'prefillWithoutApplyTemplate',
    ),
  );

  await model.doGenerate({
    prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello again' }] }],
    providerOptions: {
      llamacpp: {
        model: 'override-model',
      },
    },
  });

  assert.equal(completionBodies.length, 3);
  assert.equal(completionBodies[2].model, 'override-model');
  assert.equal(completionBodies[2].prompt, 'Hello again');
});

test('llamacpp apply-template is opt-in', async () => {
  const templateBodies = [];
  const completionBodies = [];

  const llamacpp = createLlamacpp({
    baseURL: 'http://localhost',
    apiKey: 'test-key',
    fetch: async (input, init) => {
      const url = String(input);

      if (url.endsWith('/apply-template')) {
        if (init?.body) {
          templateBodies.push(JSON.parse(init.body));
        }

        return new Response(JSON.stringify({ prompt: 'templated-chat-prompt' }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      }

      if (url.endsWith('/completion')) {
        if (init?.body) {
          completionBodies.push(JSON.parse(init.body));
        }

        return new Response(
          JSON.stringify({
            content: 'ok',
            stop_type: 'eos',
            tokens_evaluated: 1,
            tokens_predicted: 1,
            timings: {},
          }),
          {
            status: 200,
            headers: { 'content-type': 'application/json' },
          },
        );
      }

      throw new Error(`Unexpected URL: ${url}`);
    },
  });

  const model = llamacpp.languageModel('test-model');
  const result = await model.doGenerate({
    prompt: [
      { role: 'system', content: 'You are precise.' },
      { role: 'user', content: [{ type: 'text', text: 'Need weather' }] },
      {
        role: 'assistant',
        content: [{ type: 'tool-call', toolCallId: '1', toolName: 'weather', input: '{"city":"berlin"}' }],
      },
      {
        role: 'tool',
        content: [{ type: 'tool-result', toolCallId: '1', toolName: 'weather', output: 'sunny' }],
      },
    ],
    stopSequences: ['A', 'B'],
    providerOptions: {
      llamacpp: {
        stop: ['B', 'C'],
        useApplyTemplate: true,
        prefill: ' Sure!',
      },
    },
  });

  assert.equal(templateBodies.length, 1);
  assert.equal(completionBodies.length, 1);
  assert.equal(completionBodies[0].prompt, 'templated-chat-prompt Sure!');
  assert.deepEqual(completionBodies[0].stop, ['A', 'B', 'C']);
  assert.deepEqual(templateBodies[0].messages, [
    { role: 'system', content: 'You are precise.' },
    { role: 'user', content: [{ type: 'text', text: 'Need weather' }] },
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: '1',
          type: 'function',
          function: {
            name: 'weather',
            arguments: '"{\\"city\\":\\"berlin\\"}"',
          },
        },
      ],
    },
    {
      role: 'tool',
      name: 'weather',
      tool_call_id: '1',
      content: 'sunny',
    },
  ]);
  assert.equal(result.warnings.length, 0);
});

test('llamacpp embedding honors maxEmbeddingsPerCall provider option', async () => {
  let fetchCalled = false;
  const llamacpp = createLlamacpp({
    baseURL: 'http://localhost',
    apiKey: 'test-key',
    fetch: async () => {
      fetchCalled = true;
      return new Response(
        JSON.stringify({ data: [{ embedding: [0.1] }, { embedding: [0.2] }] }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      );
    },
  });

  const embeddingModel = llamacpp.embedding('embed-model');

  await assert.rejects(
    embeddingModel.doEmbed({
      values: ['one', 'two'],
      providerOptions: { llamacpp: { maxEmbeddingsPerCall: 1 } },
    }),
  );

  assert.equal(fetchCalled, false);
});

test('llamacpp stream usage updates without timings and uses generateId', async () => {
  const calls = [];
  const llamacpp = createLlamacpp({
    baseURL: 'http://localhost',
    apiKey: 'test-key',
    generateId: () => 'stream-text-id',
    fetch: async (input, init) => {
      const url = String(input);

      if (init?.body) {
        calls.push({ url, body: JSON.parse(init.body) });
      }

      if (!url.endsWith('/completion')) {
        throw new Error(`Unexpected URL: ${url}`);
      }

      const sse = [
        'data: {"content":"Hi","tokens_evaluated":3,"tokens_predicted":2}',
        '',
        'data: {"stop_type":"eos"}',
        '',
        'data: [DONE]',
        '',
      ].join('\n');

      return new Response(sse, {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      });
    },
  });

  const model = llamacpp.languageModel('test-model');
  const { stream } = await model.doStream({
    prompt: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
  });

  const parts = [];
  for await (const part of stream) {
    parts.push(part);
  }

  const textStart = parts.find((part) => part.type === 'text-start');
  assert.equal(textStart?.id, 'stream-text-id');

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, 'http://localhost/completion');
  assert.equal(calls[0].body.prompt, 'hello');
  assert.equal(calls[0].body.stream, true);

  const finish = parts.find((part) => part.type === 'finish');
  assert.equal(finish?.usage.inputTokens.total, 3);
  assert.equal(finish?.usage.outputTokens.total, 2);
  assert.equal(finish?.usage.outputTokens.text, 2);
});

test('llamacpp apply-template prefill works for stream', async () => {
  const calls = [];
  const llamacpp = createLlamacpp({
    baseURL: 'http://localhost',
    apiKey: 'test-key',
    fetch: async (input, init) => {
      const url = String(input);

      if (init?.body) {
        calls.push({ url, body: JSON.parse(init.body) });
      }

      if (url.endsWith('/apply-template')) {
        return new Response(JSON.stringify({ prompt: 'templated-stream' }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      }

      if (!url.endsWith('/completion')) {
        throw new Error(`Unexpected URL: ${url}`);
      }

      const sse = ['data: {"content":"x"}', '', 'data: [DONE]', ''].join('\n');

      return new Response(sse, {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      });
    },
  });

  const model = llamacpp.languageModel('test-model');
  const { stream } = await model.doStream({
    prompt: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
    providerOptions: {
      llamacpp: {
        useApplyTemplate: true,
        prefill: ' world',
      },
    },
  });

  for await (const part of stream) {
    void part;
  }

  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, 'http://localhost/apply-template');
  assert.equal(calls[1].url, 'http://localhost/completion');
  assert.equal(calls[1].body.prompt, 'templated-stream world');
});

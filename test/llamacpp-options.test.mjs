import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createLlamacpp } from '../dist/index.js';

test('llamacpp providerOptions map to completion body', async () => {
  const bodies = [];

  const llamacpp = createLlamacpp({
    baseURL: 'http://localhost',
    fetch: async (_input, init) => {
      if (init?.body) {
        bodies.push(JSON.parse(init.body));
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
    },
  });

  const model = llamacpp.languageModel('test-model');

  await model.doGenerate({
    prompt: [
      {
        role: 'user',
        content: [{ type: 'text', text: 'Hello' }],
      },
    ],
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
        extraParams: { custom_flag: 1 },
      },
    },
  });

  assert.equal(bodies.length, 1);
  const body = bodies[0];

  assert.equal(body.n_predict, 42);
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
});



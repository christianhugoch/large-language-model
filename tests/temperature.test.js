const {
  describe,
  test,
  expect,
  jest,
  beforeEach,
} = require("@saltcorn/db-common/test_expect");

const mockGenerateText = jest.fn(async () => ({
  text: "ok",
  steps: [],
  response: Promise.resolve({ messages: [] }),
}));

jest.mock("ai", () => ({
  generateText: (...args) => mockGenerateText(...args),
  streamText: jest.fn(),
  tool: jest.fn((definition) => definition),
  jsonSchema: jest.fn((schema) => schema),
  Output: {},
  embed: jest.fn(),
  embedMany: jest.fn(),
  experimental_transcribe: jest.fn(),
  experimental_generateImage: jest.fn(),
  experimental_generateSpeech: jest.fn(),
}));

jest.mock("@saltcorn/data/db", () => ({
  connectObj: { default_schema: "public" },
  getTenantSchema: jest.fn(() => "public"),
}));
jest.mock("@saltcorn/data/models/plugin", () => ({}));
jest.mock("@saltcorn/data/models/file", () => ({}));
jest.mock("@saltcorn/data/db/state", () => ({
  features: {},
  getState: jest.fn(() => ({ log: jest.fn() })),
}));

jest.mock("@ai-sdk/openai", () => ({
  createOpenAI: jest.fn(() => {
    const provider = jest.fn((model) => ({ model }));
    provider.chat = jest.fn((model) => ({ model }));
    provider.responses = jest.fn((model) => ({ model }));
    provider.textEmbeddingModel = jest.fn((model) => ({ model }));
    return provider;
  }),
}));

const { getCompletion } = require("../generate");

const baseConfig = {
  backend: "AI SDK",
  ai_sdk_provider: "OpenAI",
  model: "gpt-4o",
  api_key: "test-key",
};

describe("completion temperature", () => {
  beforeEach(() => mockGenerateText.mockClear());

  test("sends a configured temperature of zero", async () => {
    await getCompletion({ ...baseConfig, temperature: 0 }, { prompt: "hello" });

    expect(mockGenerateText).toHaveBeenCalledWith(
      expect.objectContaining({ temperature: 0 }),
    );
  });

  test("allows a request temperature of zero to override configuration", async () => {
    await getCompletion(
      { ...baseConfig, temperature: 0.8 },
      { prompt: "hello", temperature: 0 },
    );

    expect(mockGenerateText).toHaveBeenCalledWith(
      expect.objectContaining({ temperature: 0 }),
    );
  });

  test("omits temperature when a request explicitly sets null", async () => {
    await getCompletion(
      { ...baseConfig, temperature: 0.8 },
      { prompt: "hello", temperature: null },
    );

    expect(mockGenerateText.mock.calls[0][0]).not.toHaveProperty("temperature");
  });

  test("sends configured zero through OpenAI-compatible backends", async () => {
    const originalFetch = global.fetch;
    global.fetch = jest.fn(async () => ({
      json: async () => ({ choices: [{ message: { content: "ok" } }] }),
    }));

    try {
      await getCompletion(
        {
          backend: "OpenAI-compatible API",
          endpoint: "https://example.test/chat/completions",
          model: "gpt-4o",
          temperature: 0,
        },
        { prompt: "hello" },
      );

      const requestBody = JSON.parse(global.fetch.mock.calls[0][1].body);
      expect(requestBody.temperature).toBe(0);
    } finally {
      global.fetch = originalFetch;
    }
  });
});

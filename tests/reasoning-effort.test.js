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

const mockMakeProvider = () => {
  const provider = jest.fn((model) => ({ model }));
  provider.chat = jest.fn((model) => ({ model }));
  provider.responses = jest.fn((model) => ({ model }));
  provider.textEmbeddingModel = jest.fn((model) => ({ model }));
  return provider;
};

jest.mock("@ai-sdk/openai", () => ({
  createOpenAI: jest.fn(() => mockMakeProvider()),
}));
jest.mock("@ai-sdk/anthropic", () => ({
  createAnthropic: jest.fn(() => mockMakeProvider()),
}));

const { getCompletion } = require("../generate");

const complete = (config, options = {}) =>
  getCompletion(
    {
      backend: "AI SDK",
      model: "test-model",
      ...config,
    },
    { prompt: "hello", ...options },
  );

describe("AI SDK reasoning effort", () => {
  beforeEach(() => mockGenerateText.mockClear());

  test.each(["OpenAI", "Z.ai"])(
    "maps %s reasoning effort to OpenAI provider options",
    async (provider) => {
      await complete({ ai_sdk_provider: provider, reasoning_effort: "none" });

      expect(mockGenerateText).toHaveBeenCalledWith(
        expect.objectContaining({
          providerOptions: {
            openai: { reasoningEffort: "none" },
          },
        }),
      );
    },
  );

  test("maps Anthropic reasoning effort to effort", async () => {
    await complete({
      ai_sdk_provider: "Anthropic",
      reasoning_effort: "low",
    });

    expect(mockGenerateText).toHaveBeenCalledWith(
      expect.objectContaining({
        providerOptions: {
          anthropic: { effort: "low" },
        },
      }),
    );
  });

  test("allows a request override", async () => {
    await complete(
      { ai_sdk_provider: "OpenAI", reasoning_effort: "high" },
      { reasoning_effort: "minimal" },
    );

    expect(mockGenerateText.mock.calls[0][0].providerOptions.openai).toEqual({
      reasoningEffort: "minimal",
    });
  });

  test("leaves provider defaults unchanged when blank", async () => {
    await complete({ ai_sdk_provider: "OpenAI", reasoning_effort: "" });

    expect(mockGenerateText.mock.calls[0][0]).not.toHaveProperty(
      "providerOptions",
    );
  });
});

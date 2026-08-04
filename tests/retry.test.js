/**
 * Retry / fallback tests for getCompletionAISDK().
 *
 * When inference fails we retry with a different configuration: if the main
 * configuration was used we try each alternative configuration in turn, and if
 * an alternative configuration was chosen we try the main configuration first
 * and then the remaining alternatives.
 *
 * The failure is simulated by corrupting an API key. Most of the tests run
 * against a local OpenAI-compatible mock server that rejects any request that
 * does not present the correct key, so they need neither network access nor
 * provider credentials. The tests at the bottom of the file repeat the same
 * scenario against the real providers for which credentials are in the
 * environment (see configs.js).
 */

const http = require("http");
const {
  afterAll,
  beforeAll,
  describe,
  it,
  expect,
  jest,
} = require("@saltcorn/db-common/test_expect");

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

const { getCompletion } = require("../generate");

jest.setTimeout(40000);

const GOOD_KEY = "good-api-key";
const BAD_KEY = "corrupted-api-key";
const ANSWER = "The capital of France is Paris.";
const PROMPT = "What is the capital of France?";

// A minimal OpenAI-compatible chat completions server. Any request that does
// not present GOOD_KEY is rejected with a 401, which is how a configuration
// with a corrupted API key fails.
const startMockProvider = () =>
  new Promise((resolve) => {
    const requests = [];
    const server = http.createServer((req, res) => {
      let raw = "";
      req.on("data", (chunk) => (raw += chunk));
      req.on("end", () => {
        const auth = (req.headers.authorization || "").replace(/^Bearer /, "");
        const body = JSON.parse(raw || "{}");
        requests.push({ auth, model: body.model, stream: !!body.stream });
        if (auth !== GOOD_KEY) {
          res.writeHead(401, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: {
                message: "Incorrect API key provided",
                type: "invalid_request_error",
                code: "invalid_api_key",
              },
            }),
          );
          return;
        }
        const created = Math.round(Date.now() / 1000);
        if (body.stream) {
          res.writeHead(200, {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          });
          const chunk = (delta, finish_reason = null) =>
            `data: ${JSON.stringify({
              id: "chatcmpl-mock",
              object: "chat.completion.chunk",
              created,
              model: body.model,
              choices: [{ index: 0, delta, finish_reason }],
            })}\n\n`;
          res.write(chunk({ role: "assistant", content: "" }));
          for (const part of ["The capital ", "of France ", "is Paris."])
            res.write(chunk({ content: part }));
          res.write(chunk({}, "stop"));
          res.write("data: [DONE]\n\n");
          res.end();
        } else {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              id: "chatcmpl-mock",
              object: "chat.completion",
              created,
              model: body.model,
              choices: [
                {
                  index: 0,
                  message: { role: "assistant", content: ANSWER },
                  logprobs: null,
                  finish_reason: "stop",
                },
              ],
              usage: {
                prompt_tokens: 10,
                completion_tokens: 7,
                total_tokens: 17,
              },
            }),
          );
        }
      });
    });
    server.listen(0, "127.0.0.1", () =>
      resolve({
        requests,
        url: `http://127.0.0.1:${server.address().port}/v1`,
        close: () => new Promise((res) => server.close(res)),
      }),
    );
  });

describe("retry with alternative configurations", () => {
  let mock;
  beforeAll(async () => {
    mock = await startMockProvider();
  });
  afterAll(async () => {
    await mock.close();
  });

  const mainConfig = (api_key, alt_aisdk_configs) => ({
    backend: "AI SDK",
    ai_sdk_provider: "OpenAI-compatible",
    api_url: mock.url,
    api_key,
    model: "mock-model",
    temperature: null,
    alt_aisdk_configs,
  });

  const alt = (name, api_key) => ({
    name,
    alt_provider: "OpenAI-compatible",
    api_url: mock.url,
    api_key,
    model: "mock-model",
  });

  const keysUsed = () => mock.requests.map((r) => r.auth);

  it("falls back to the alternative configuration when the main fails", async () => {
    mock.requests.length = 0;
    const answer = await getCompletion(
      mainConfig(BAD_KEY, [alt("backup", GOOD_KEY)]),
      { prompt: PROMPT },
    );
    expect(answer).toBe(ANSWER);
    expect(keysUsed()).toEqual([BAD_KEY, GOOD_KEY]);
  });

  it("tries the alternative configurations in turn", async () => {
    mock.requests.length = 0;
    const answer = await getCompletion(
      mainConfig(BAD_KEY, [
        alt("first", BAD_KEY),
        alt("second", GOOD_KEY),
        alt("third", GOOD_KEY),
      ]),
      { prompt: PROMPT },
    );
    expect(answer).toBe(ANSWER);
    expect(keysUsed()).toEqual([BAD_KEY, BAD_KEY, GOOD_KEY]);
  });

  it("falls back to the main configuration when the chosen alternative fails", async () => {
    mock.requests.length = 0;
    const answer = await getCompletion(
      mainConfig(GOOD_KEY, [alt("broken", BAD_KEY), alt("other", GOOD_KEY)]),
      { prompt: PROMPT, alt_config: "broken" },
    );
    expect(answer).toBe(ANSWER);
    // the chosen alternative first, then the main configuration
    expect(keysUsed()).toEqual([BAD_KEY, GOOD_KEY]);
  });

  it("retries in streaming mode without duplicating output", async () => {
    mock.requests.length = 0;
    const streamed = [];
    const answer = await getCompletion(
      mainConfig(BAD_KEY, [alt("backup", GOOD_KEY)]),
      { prompt: PROMPT, streamCallback: (s) => streamed.push(s) },
    );
    expect(answer).toBe(ANSWER);
    expect(streamed.join("")).toBe(ANSWER);
    expect(keysUsed()).toEqual([BAD_KEY, GOOD_KEY]);
    expect(mock.requests.every((r) => r.stream)).toBe(true);
  });

  it("does not retry when the number of retries is zero", async () => {
    mock.requests.length = 0;
    await expect(
      getCompletion(
        {
          ...mainConfig(BAD_KEY, [alt("backup", GOOD_KEY)]),
          num_retries: 0,
        },
        { prompt: PROMPT },
      ),
    ).rejects.toThrow();
    expect(mock.requests.length).toBe(1);
  });

  it("honours the configured number of retries", async () => {
    mock.requests.length = 0;
    await expect(
      getCompletion(
        {
          ...mainConfig(BAD_KEY, [
            alt("first", BAD_KEY),
            alt("second", BAD_KEY),
          ]),
          num_retries: 1,
        },
        { prompt: PROMPT },
      ),
    ).rejects.toThrow();
    expect(mock.requests.length).toBe(2);
  });

  it("throws when every configuration fails", async () => {
    mock.requests.length = 0;
    await expect(
      getCompletion(
        mainConfig(BAD_KEY, [alt("first", BAD_KEY), alt("second", BAD_KEY)]),
        { prompt: PROMPT },
      ),
    ).rejects.toThrow();
    expect(mock.requests.length).toBe(3);
  });

  it("reports the error to the stream when every configuration fails", async () => {
    mock.requests.length = 0;
    const streamed = [];
    await expect(
      getCompletion(mainConfig(BAD_KEY, [alt("backup", BAD_KEY)]), {
        prompt: PROMPT,
        streamCallback: (s) => streamed.push(s),
      }),
    ).rejects.toThrow();
    expect(mock.requests.length).toBe(2);
    expect(streamed.length).toBe(1);
  });

  it("does not retry without alternative configurations", async () => {
    mock.requests.length = 0;
    await expect(
      getCompletion(mainConfig(BAD_KEY, []), { prompt: PROMPT }),
    ).rejects.toThrow();
    // an invalid API key is not a retryable error for the AI SDK either
    expect(mock.requests.length).toBe(1);
  });
});

// The same scenario against the real providers: the API key of the main
// configuration is corrupted and the working credentials are moved to an
// alternative configuration. Only runs for providers whose keys are in the
// environment.
const API_KEY_FIELD = {
  OpenAI: "api_key",
  Anthropic: "anthropic_api_key",
  Google: "google_api_key",
  OpenRouter: "openrouter_api_key",
  "Z.ai": "zai_api_key",
};

for (const nameconfig of require("./configs")) {
  const { name, skipTests = [], ...config } = nameconfig;
  if (config.backend !== "AI SDK") continue;
  const keyField = API_KEY_FIELD[config.ai_sdk_provider];
  if (!keyField || !config[keyField]) continue;

  describe("retry with a corrupted API key on " + name, () => {
    const corrupted = {
      ...config,
      [keyField]: "corrupted-" + config[keyField],
      alt_aisdk_configs: [
        {
          name: "fallback",
          alt_provider: config.ai_sdk_provider,
          model: config.model,
          embed_model: config.embed_model,
          [keyField]: config[keyField],
        },
      ],
    };

    it("generates text with the fallback configuration", async () => {
      const answer = await getCompletion(corrupted, {
        prompt: "What is the Capital of France?",
      });
      expect(typeof answer).toBe("string");
      expect(answer).toContain("Paris");
    });

    it("streams text with the fallback configuration", async () => {
      const streamed = [];
      const answer = await getCompletion(corrupted, {
        prompt: "What is the Capital of France?",
        streamCallback: (s) => streamed.push(s),
      });
      expect(typeof answer).toBe("string");
      expect(answer).toContain("Paris");
      expect(streamed.join("")).toContain("Paris");
    });

    it("fails when there is no fallback configuration", async () => {
      await expect(
        getCompletion(
          { ...corrupted, alt_aisdk_configs: [] },
          { prompt: "What is the Capital of France?" },
        ),
      ).rejects.toThrow();
    });
  });
}

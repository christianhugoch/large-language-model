/**
 * Regression tests for toolResponse() / addImageMesssage() message formatting.
 *
 * Background: getCompletion() maps the backends "OpenAI", "OpenAI-compatible
 * API" (Z.ai/GLM, OpenRouter, ...) and "Local Ollama" to the same
 * getCompletionOpenAICompatible() code path, so they all speak the OpenAI
 * chat-completions wire format. But toolResponse() and addImageMesssage() only
 * had a `case "OpenAI":`, so for any other OpenAI-compatible backend the call
 * fell through to the empty `default:` and the message (a tool result, or an
 * image) was silently dropped.
 *
 * A dropped tool result is catastrophic for agents: every assistant tool_call
 * must be followed by a matching `role:"tool"` message, otherwise the LLM never
 * sees the result and re-issues the same call forever (the
 * "Missing tool result for ... inserting placeholder" loop seen with the GLM /
 * Z.ai copilot).
 *
 * These tests exercise the message builders directly, with no network and no
 * Saltcorn state: they only require that generate.js loads and that the right
 * object gets pushed onto the supplied chat array.
 */

const { describe, it, expect, jest } = require("@saltcorn/db-common/test_expect");

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

const { toolResponse, addImageMesssage } = require("../generate");

// Representative OpenAI-compatible backends (the one that originally
// triggered the bug is GLM via Z.ai, i.e. backend "OpenAI-compatible API").
const OPENAI_COMPAT_CONFIGS = [
  { label: '"OpenAI" (native)', backend: "OpenAI" },
  { label: '"OpenAI-compatible API" (Z.ai/GLM, OpenRouter)', backend: "OpenAI-compatible API" },
  { label: '"Local Ollama"', backend: "Local Ollama" },
];

for (const { label, backend } of OPENAI_COMPAT_CONFIGS) {
  describe(`toolResponse with backend ${label}`, () => {
    it("appends a role:tool message matching the tool_call_id", async () => {
      const chat = [
        {
          role: "assistant",
          tool_calls: [
            {
              id: "call_abc",
              type: "function",
              function: { name: "get_view_config", arguments: '{"name":"Ficha"}' },
            },
          ],
        },
      ];
      const before = chat.length;

      await toolResponse({ backend }, {
        chat,
        prompt: { layout: {}, blocks: [] },
        tool_call: { tool_call_id: "call_abc", tool_name: "get_view_config" },
      });

      expect(chat.length).toBe(before + 1);
      const result = chat[chat.length - 1];
      expect(result.role).toBe("tool");
      expect(result.tool_call_id).toBe("call_abc");
      // object results are JSON-stringified
      expect(typeof result.content).toBe("string");
      expect(JSON.parse(result.content)).toEqual({ layout: {}, blocks: [] });
    });

    it("keeps string results as-is and defaults to 'Action run'", async () => {
      const chat1 = [];
      await toolResponse({ backend }, {
        chat: chat1,
        prompt: "CONFIG_RESULT_JSON",
        tool_call: { tool_call_id: "call_str", tool_name: "get_view_config" },
      });
      expect(chat1[0].role).toBe("tool");
      expect(chat1[0].tool_call_id).toBe("call_str");
      expect(chat1[0].content).toBe("CONFIG_RESULT_JSON");

      const chat2 = [];
      await toolResponse({ backend }, {
        chat: chat2,
        prompt: undefined,
        tool_call: { tool_call_id: "call_def", tool_name: "noop" },
      });
      expect(chat2[0].content).toBe("Action run");
    });

    it("resolves the tool_call from the last assistant message when not given", async () => {
      const chat = [
        {
          role: "assistant",
          tool_calls: [
            { id: "call_inferred", function: { name: "list_entities" } },
          ],
        },
      ];
      await toolResponse({ backend }, { chat, prompt: "OK" });
      expect(chat[1].role).toBe("tool");
      expect(chat[1].tool_call_id).toBe("call_inferred");
    });
  });
}

describe("toolResponse regression: OpenAI-compatible API used to drop the result", () => {
  // This is the exact shape that the Saltcorn copilot / agents call site uses
  // (see ensureToolResults in saltcorn-agents/common.js), and the one that was
  // silently a no-op before the fix.
  it("pushes a tool result for backend 'OpenAI-compatible API'", async () => {
    const chat = [];
    await toolResponse(
      { backend: "OpenAI-compatible API" },
      {
        chat,
        prompt: "The tool call was not completed.",
        tool_call: { tool_call_id: "call_-7388952334631430260", tool_name: "get_view_config" },
      },
    );
    expect(chat.length).toBe(1);
    expect(chat[0]).toMatchObject({
      role: "tool",
      tool_call_id: "call_-7388952334631430260",
    });
  });
});

describe("addImageMesssage with OpenAI-compatible backends", () => {
  for (const { label, backend } of OPENAI_COMPAT_CONFIGS) {
    it(`appends an image_url user message for backend ${label}`, async () => {
      const chat = [];
      await addImageMesssage({ backend }, { chat, prompt: "https://x/y.png" });
      expect(chat.length).toBe(1);
      expect(chat[0].role).toBe("user");
      // OpenAI-compatible (non-responses_api) image format
      expect(Array.isArray(chat[0].content)).toBe(true);
      expect(chat[0].content[0].type).toBe("image_url");
    });
  }
});

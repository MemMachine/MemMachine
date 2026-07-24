import { queueSearchResults, resetMockMemMachine } from "./__mocks__/memmachine-client.js";
import { registerPlugin } from "./plugin-test-helpers.js";

const episode = (uid: string, content = uid) => ({
  uid,
  content,
  producer_id: "user-1",
  producer_role: "user",
});

const searchResult = (episodic_memory?: Record<string, unknown> | null) => ({
  status: 200,
  content: { episodic_memory },
});

async function recall(result: Record<string, unknown>) {
  queueSearchResults(result);
  const hook = registerPlugin({ autoRecall: true }).hook("before_agent_start");
  return hook(
    { prompt: "What do I remember?" },
    { sessionId: "session-1", sessionKey: "session-key-1" },
  );
}

describe("nested episodic search parsing", () => {
  beforeEach(resetMockMemMachine);

  it("includes short-term results", async () => {
    const result = await recall(searchResult({
      short_term_memory: { episodes: [episode("short-1", "short memory")] },
    }));

    expect(result).toMatchObject({ prependContext: expect.stringContaining("short memory") });
  });

  it("includes long-term results", async () => {
    const result = await recall(searchResult({
      long_term_memory: { episodes: [episode("long-1", "long memory")] },
    }));

    expect(result).toMatchObject({ prependContext: expect.stringContaining("long memory") });
  });

  it("combines long-term and short-term results in the existing order", async () => {
    const result = await recall(searchResult({
      long_term_memory: { episodes: [episode("long-1", "long memory")] },
      short_term_memory: { episodes: [episode("short-1", "short memory")] },
    }));

    const context = (result as { prependContext: string }).prependContext;
    expect(context.indexOf("long memory")).toBeLessThan(context.indexOf("short memory"));
  });

  it("preserves duplicate search entries", async () => {
    const result = await recall(searchResult({
      long_term_memory: { episodes: [episode("same", "duplicate memory")] },
      short_term_memory: { episodes: [episode("same", "duplicate memory")] },
    }));

    const context = (result as { prependContext: string }).prependContext;
    expect(context.match(/duplicate memory/g)).toHaveLength(2);
  });

  it.each([
    ["absent", searchResult(undefined)],
    ["null", searchResult(null)],
    ["empty", searchResult({})],
  ])("returns no recall context for %s episodic sections", async (_name, result) => {
    await expect(recall(result)).resolves.toBeUndefined();
  });
});

import {
  mockMemory,
  queueListResults,
  resetMockMemMachine,
} from "./__mocks__/memmachine-client.js";
import { registerPlugin } from "./plugin-test-helpers.js";

const episodic = (uid: string) => ({
  uid,
  content: `episode ${uid}`,
  session_key: "session-1",
  created_at: "2026-07-15T00:00:00Z",
  producer_id: "user-1",
  producer_role: "user",
});

const semantic = (id: string) => ({
  set_id: "set-1",
  category: "profile",
  tag: "preference",
  feature_name: "drink",
  value: "tea",
  metadata: { id },
});

const listResult = (content: Record<string, unknown>) => ({ status: 200, content });

describe("memory_list", () => {
  beforeEach(resetMockMemMachine);

  it("returns flat episodic list results", async () => {
    queueListResults(
      listResult({ episodic_memory: [episodic("ep-1")] }),
      listResult({ semantic_memory: [] }),
    );

    const registered = registerPlugin();
    const result = await registered.tool("memory_list", { sessionKey: "session-1" }).execute(
      "call-1",
      { scope: "session", pageSize: 20, pageNum: 2 },
    );

    expect(result).toMatchObject({
      scope: "session",
      pageSize: 20,
      pageNum: 2,
      result: { episodic: [episodic("ep-1")], semantic: [] },
    });
    expect(mockMemory.listCalls).toEqual([
      {
        page_size: 20,
        page_num: 2,
        filter: "metadata.run_id = 'session-1'",
        type: "episodic",
      },
      {
        page_size: 20,
        page_num: 2,
        filter: "metadata.run_id = 'session-1'",
        type: "semantic",
      },
    ]);
  });

  it("returns semantic list results", async () => {
    queueListResults(
      listResult({ episodic_memory: [] }),
      listResult({ semantic_memory: [semantic("sem-1")] }),
    );

    const result = await registerPlugin().tool("memory_list").execute("call-1", {});

    expect(result).toMatchObject({
      result: { episodic: [], semantic: [semantic("sem-1")] },
    });
  });

  it("returns both collections and preserves list deduplication", async () => {
    queueListResults(
      listResult({ episodic_memory: [episodic("ep-1"), episodic("ep-1"), episodic("ep-2")] }),
      listResult({ semantic_memory: [semantic("sem-1"), semantic("sem-1"), semantic("sem-2")] }),
    );

    const result = await registerPlugin().tool("memory_list").execute("call-1", {});

    expect(result).toMatchObject({
      result: {
        episodic: [episodic("ep-1"), episodic("ep-2")],
        semantic: [semantic("sem-1"), semantic("sem-2")],
      },
    });
  });

  it.each([
    ["absent", listResult({}), listResult({})],
    ["null", listResult({ episodic_memory: null }), listResult({ semantic_memory: null })],
    ["empty", listResult({ episodic_memory: [] }), listResult({ semantic_memory: [] })],
  ])("returns empty arrays for %s sections", async (_name, episodicResult, semanticResult) => {
    queueListResults(episodicResult, semanticResult);

    const result = await registerPlugin().tool("memory_list").execute("call-1", {});

    expect(result).toMatchObject({ result: { episodic: [], semantic: [] } });
  });

  it("preserves the existing null result when a request fails", async () => {
    queueListResults(new Error("list failed"));

    const result = await registerPlugin().tool("memory_list").execute("call-1", {});

    expect(result).toMatchObject({ result: null });
  });
});

describe("memory_get", () => {
  beforeEach(resetMockMemMachine);

  it("finds an episodic memory by ID without semantic fallback", async () => {
    queueListResults(listResult({ episodic_memory: [episodic("ep-1")] }));

    const result = await registerPlugin().tool("memory_get").execute("call-1", {
      id: "ep-1",
      type: "auto",
    });

    expect(result).toEqual({ id: "ep-1", episodic: [episodic("ep-1")], semantic: [] });
    expect(mockMemory.listCalls).toEqual([
      { type: "episodic", filter: "uid = 'ep-1'", page_size: 1 },
    ]);
  });

  it("falls back to semantic memory when the episodic ID is missing", async () => {
    queueListResults(
      listResult({ episodic_memory: [] }),
      listResult({ semantic_memory: [semantic("sem-1")] }),
    );

    const result = await registerPlugin().tool("memory_get").execute("call-1", {
      id: "sem-1",
      type: "auto",
    });

    expect(result).toEqual({ id: "sem-1", episodic: [], semantic: [semantic("sem-1")] });
    expect(mockMemory.listCalls[1]).toEqual({
      type: "semantic",
      filter: "metadata.id = 'sem-1'",
      page_size: 1,
    });
  });

  it("preserves the empty not-found result", async () => {
    queueListResults(
      listResult({ episodic_memory: [] }),
      listResult({ semantic_memory: [] }),
    );

    const result = await registerPlugin().tool("memory_get").execute("call-1", {
      id: "missing",
    });

    expect(result).toEqual({ id: "missing", episodic: [], semantic: [] });
  });

  it("preserves request failures", async () => {
    queueListResults(new Error("list failed"));

    await expect(
      registerPlugin().tool("memory_get").execute("call-1", { id: "ep-1" }),
    ).rejects.toThrow("list failed");
  });
});

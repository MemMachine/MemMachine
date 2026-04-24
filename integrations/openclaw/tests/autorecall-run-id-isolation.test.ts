/**
 * Regression tests for #1318: autoRecall scope leaks previous-conversation
 * memories when userId is the shared default constant "openclaw".
 *
 * These tests exercise buildAutoRecallFilter and buildScopeFilter as named
 * exports. They are pure-function unit tests — no network, no MemMachine
 * server required.
 *
 * External dependencies (@sinclair/typebox, @memmachine/client,
 * openclaw/plugin-sdk) are intercepted by moduleNameMapper in jest.config.cjs
 * and replaced with minimal JS stubs from tests/__mocks__/.
 */
import { buildAutoRecallFilter, buildScopeFilter } from "../index.mts";

// ---------------------------------------------------------------------------
// buildAutoRecallFilter — the new helper that fixes #1318
// ---------------------------------------------------------------------------

describe("buildAutoRecallFilter", () => {
  const DEFAULT_USER = "openclaw";
  const REAL_USER = "user-alice-slack-U012AB3CD";
  const SESSION_ID = "sess-ephemeral-uuid-abc123";

  describe("sessionId undefined → safe degradation", () => {
    it("returns undefined when sessionId is undefined and userId is default", () => {
      expect(buildAutoRecallFilter(undefined, DEFAULT_USER)).toBeUndefined();
    });

    it("returns undefined when sessionId is undefined and userId is real", () => {
      expect(buildAutoRecallFilter(undefined, REAL_USER)).toBeUndefined();
    });

    it("returns undefined when sessionId is undefined and userId is undefined", () => {
      expect(buildAutoRecallFilter(undefined, undefined)).toBeUndefined();
    });
  });

  describe("sessionId defined + default userId → session-only filter", () => {
    it("returns run_id filter when userId is the default constant", () => {
      const filter = buildAutoRecallFilter(SESSION_ID, DEFAULT_USER);
      expect(filter).toBe(`metadata.run_id = '${SESSION_ID}'`);
    });

    it("returns run_id filter when userId is undefined", () => {
      const filter = buildAutoRecallFilter(SESSION_ID, undefined);
      expect(filter).toBe(`metadata.run_id = '${SESSION_ID}'`);
    });

    it("does NOT include OR user_id arm with default userId (the #1318 bug)", () => {
      const filter = buildAutoRecallFilter(SESSION_ID, DEFAULT_USER);
      // Pre-fix this would have been:
      //   `(metadata.run_id = '...' OR metadata.user_id = 'openclaw')`
      // The OR arm is what caused Q1 memory to leak into Q2 recall.
      expect(filter).not.toContain("OR");
      expect(filter).not.toContain("user_id");
    });
  });

  describe("sessionId defined + real per-user userId → user-scoped filter", () => {
    it("returns user_id filter for cross-session long-term recall", () => {
      const filter = buildAutoRecallFilter(SESSION_ID, REAL_USER);
      expect(filter).toBe(`metadata.user_id = '${REAL_USER}'`);
    });

    it("does not include run_id when a real userId is configured", () => {
      const filter = buildAutoRecallFilter(SESSION_ID, REAL_USER);
      expect(filter).not.toContain("run_id");
    });
  });

  describe("sanitization — single-quote stripping", () => {
    it("strips single quotes from sessionId", () => {
      const filter = buildAutoRecallFilter("sess-' OR '1'='1", DEFAULT_USER);
      // After sanitization the injected content is gone
      expect(filter).toBe("metadata.run_id = 'sess- OR 1=1'");
    });

    it("strips single quotes from userId", () => {
      const filter = buildAutoRecallFilter(SESSION_ID, "real-user-' DROP TABLE");
      expect(filter).toBe("metadata.user_id = 'real-user- DROP TABLE'");
    });
  });
});

// ---------------------------------------------------------------------------
// buildScopeFilter — existing helper, exported for completeness
// ---------------------------------------------------------------------------

describe("buildScopeFilter", () => {
  const SESSION_KEY = "main/telegram/peer123";
  const USER_ID = "openclaw";

  describe("scope = 'session'", () => {
    it("returns run_id filter when sessionKey is defined", () => {
      expect(buildScopeFilter("session", SESSION_KEY, USER_ID)).toBe(
        `metadata.run_id = '${SESSION_KEY}'`,
      );
    });

    it("returns undefined when sessionKey is undefined", () => {
      expect(buildScopeFilter("session", undefined, USER_ID)).toBeUndefined();
    });
  });

  describe("scope = 'long-term'", () => {
    it("returns user_id filter when userId is defined", () => {
      expect(buildScopeFilter("long-term", SESSION_KEY, USER_ID)).toBe(
        `metadata.user_id = '${USER_ID}'`,
      );
    });

    it("returns undefined when userId is undefined", () => {
      expect(buildScopeFilter("long-term", SESSION_KEY, undefined)).toBeUndefined();
    });
  });

  describe("scope = 'all'", () => {
    it("returns OR filter when both are defined", () => {
      expect(buildScopeFilter("all", SESSION_KEY, USER_ID)).toBe(
        `(metadata.run_id = '${SESSION_KEY}' OR metadata.user_id = '${USER_ID}')`,
      );
    });

    it("returns run_id-only filter when userId is undefined", () => {
      expect(buildScopeFilter("all", SESSION_KEY, undefined)).toBe(
        `metadata.run_id = '${SESSION_KEY}'`,
      );
    });

    it("returns user_id-only filter when sessionKey is undefined", () => {
      expect(buildScopeFilter("all", undefined, USER_ID)).toBe(
        `metadata.user_id = '${USER_ID}'`,
      );
    });

    it("returns undefined when both are undefined", () => {
      expect(buildScopeFilter("all", undefined, undefined)).toBeUndefined();
    });
  });

  /**
   * Documents the pre-fix bug: buildScopeFilter("all", sessionKey, "openclaw")
   * produced an OR filter that matched ALL stored memories because user_id is
   * the same shared constant for every install without a configured userId.
   *
   * buildAutoRecallFilter replaces this call in the before_agent_start hook.
   * This test is kept as a red/green signal: if someone reverts the hook to
   * use buildScopeFilter("all", ...) again, this test documents WHY that is
   * wrong — the OR user_id arm leaks cross-conversation memories.
   */
  it("DOCUMENTS BUG #1318: scope=all with shared userId produces OR filter that matches all memories", () => {
    const filter = buildScopeFilter("all", "main/slack/U111", "openclaw");
    // This filter matches any memory where user_id = 'openclaw', i.e. every
    // memory stored by any session on this install. That is the bug.
    expect(filter).toContain("OR metadata.user_id = 'openclaw'");
    // Confirm buildAutoRecallFilter does NOT produce this dangerous filter:
    const safeFilter = buildAutoRecallFilter("sess-ephemeral", "openclaw");
    expect(safeFilter).not.toContain("OR");
    expect(safeFilter).not.toContain("user_id");
  });
});

/**
 * Regression tests for #1268: when `baseUrl` is unset/blank, the OpenClaw
 * plugin must pass `undefined` to MemMachineClient so the TS client's own
 * default (`https://api.memmachine.ai/v2`) is used. Passing an empty string
 * would defeat the default in axios and produce 404s against the cloud API.
 *
 * As with the other tests in this package, external deps are stubbed via
 * moduleNameMapper in jest.config.cjs, so this is a pure-function unit test.
 */
import { resolveBaseUrl } from "../index.mts";

describe("resolveBaseUrl", () => {
  it("returns undefined when value is not a string", () => {
    expect(resolveBaseUrl(undefined)).toBeUndefined();
    expect(resolveBaseUrl(null)).toBeUndefined();
    expect(resolveBaseUrl(42)).toBeUndefined();
    expect(resolveBaseUrl({})).toBeUndefined();
  });

  it("returns undefined for empty string", () => {
    expect(resolveBaseUrl("")).toBeUndefined();
  });

  it("returns undefined for whitespace-only strings", () => {
    expect(resolveBaseUrl("   ")).toBeUndefined();
    expect(resolveBaseUrl("\t\n")).toBeUndefined();
  });

  it("returns the trimmed value for non-empty strings", () => {
    expect(resolveBaseUrl("https://api.memmachine.ai/v2")).toBe(
      "https://api.memmachine.ai/v2",
    );
    expect(resolveBaseUrl("  https://api.memmachine.ai/v2  ")).toBe(
      "https://api.memmachine.ai/v2",
    );
  });

  it("preserves self-hosted URLs", () => {
    expect(resolveBaseUrl("http://localhost:8080/api/v2")).toBe(
      "http://localhost:8080/api/v2",
    );
  });
});

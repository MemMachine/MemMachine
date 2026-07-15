// Runtime stub for openclaw/plugin-sdk/core used in Jest tests.
export function jsonResult(value) { return value; }
export function readNumberParam(params, key) {
  return typeof params[key] === "number" ? params[key] : undefined;
}
export function readStringParam(params, key, options = {}) {
  const value = typeof params[key] === "string" ? params[key] : undefined;
  if (options.required && value === undefined) {
    throw new Error(`Missing required parameter: ${key}`);
  }
  return value;
}

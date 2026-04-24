// Runtime stub for @sinclair/typebox used in Jest tests.
// The real package is used at build time via tsup; tests only need the Type
// object to not throw when the top-level schema constants are initialised.
const stub = (...args) => ({ __stub: true, args });
export const Type = {
  Object: stub,
  String: stub,
  Number: stub,
  Boolean: stub,
  Optional: stub,
  Union: stub,
  Literal: stub,
  Array: stub,
  Record: stub,
};

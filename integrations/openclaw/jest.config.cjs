/** @type {import('jest').Config} */
const config = {
  // Use the ESM preset so ts-jest can handle .mts files.
  // The test script passes --experimental-vm-modules via NODE_OPTIONS.
  preset: "ts-jest/presets/default-esm",
  testEnvironment: "node",
  extensionsToTreatAsEsm: [".ts", ".mts"],
  // Tests live under tests/
  testMatch: ["<rootDir>/tests/**/*.test.ts"],
  // Map path aliases to runtime JS stubs so tests can import index.mts without
  // pulling in the real external packages (which have no test registry entry).
  moduleNameMapper: {
    // Strip the .js extension that ESM imports add (ts-jest ESM quirk)
    "^(\\.{1,2}/.*)\\.js$": "$1",
    "^@sinclair/typebox$": "<rootDir>/tests/__mocks__/typebox.js",
    "^@memmachine/client$": "<rootDir>/tests/__mocks__/memmachine-client.js",
    "^openclaw/plugin-sdk$": "<rootDir>/tests/__mocks__/openclaw-plugin-sdk.js",
  },
  transform: {
    "^.+\\.m?ts$": [
      "ts-jest",
      {
        useESM: true,
        tsconfig: {
          module: "ESNext",
          moduleResolution: "Bundler",
          esModuleInterop: true,
          strict: false,
          skipLibCheck: true,
        },
      },
    ],
  },
};

module.exports = config;

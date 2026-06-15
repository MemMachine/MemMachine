import MemMachinePlugin from "../index.mts";
import {
  __getMemMachineClientConfigs,
  __resetMemMachineClientConfigs,
} from "@memmachine/client";

describe("baseUrl normalization", () => {
  beforeEach(() => {
    __resetMemMachineClientConfigs();
  });

  it.each(["", "   "])(
    "passes undefined base_url to MemMachineClient when baseUrl is %j",
    async (baseUrl) => {
      let memorySearchFactory:
        | ((ctx: { sessionKey?: string }) => { execute: (toolCallId: string, params: Record<string, unknown>) => Promise<unknown> })
        | undefined;

      const api = {
        pluginConfig: {
          apiKey: "mm-test-key",
          baseUrl,
          orgId: "org-test",
          projectId: "project-test",
          userId: "user-test",
        },
        logger: {
          info: () => undefined,
          warn: () => undefined,
          error: () => undefined,
        },
        registerMemoryPromptSection: () => undefined,
        registerTool: (factory, meta) => {
          if (meta?.name === "memory_search") {
            memorySearchFactory = factory;
          }
        },
        registerCli: () => undefined,
        registerService: () => undefined,
        on: () => undefined,
      };

      MemMachinePlugin.register(api as never);

      expect(memorySearchFactory).toBeDefined();

      const tool = memorySearchFactory!({ sessionKey: "session-test" });
      await tool.execute("tool-call-id", { query: "hello world" });

      expect(__getMemMachineClientConfigs()).toEqual([
        { api_key: "mm-test-key", base_url: undefined },
      ]);
    },
  );
});

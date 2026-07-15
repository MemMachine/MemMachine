import plugin from "../index.mts";

type Tool = {
  execute(toolCallId: string, params: Record<string, unknown>): Promise<unknown>;
};

type Hook = (event: Record<string, unknown>, context: Record<string, unknown>) => Promise<unknown>;

export function registerPlugin(pluginConfig: Record<string, unknown> = {}) {
  const tools = new Map<string, (context: Record<string, unknown>) => Tool>();
  const hooks = new Map<string, Hook>();
  const api = {
    pluginConfig,
    logger: {
      info: () => undefined,
      warn: () => undefined,
      error: () => undefined,
    },
    registerMemoryPromptSection: () => undefined,
    registerTool(factory: (context: Record<string, unknown>) => Tool, options: { name: string }) {
      tools.set(options.name, factory);
    },
    registerCli: () => undefined,
    registerService: () => undefined,
    on(name: string, hook: Hook) {
      hooks.set(name, hook);
    },
  };

  plugin.register(api as never);

  return {
    tool(name: string, context: Record<string, unknown> = {}): Tool {
      const factory = tools.get(name);
      if (!factory) throw new Error(`Tool not registered: ${name}`);
      return factory(context);
    },
    hook(name: string): Hook {
      const hook = hooks.get(name);
      if (!hook) throw new Error(`Hook not registered: ${name}`);
      return hook;
    },
  };
}

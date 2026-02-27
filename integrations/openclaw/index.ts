import { Type } from "@sinclair/typebox";
import MemMachineClient, {
  type AddMemoryResult,
  type EpisodicMemory,
  type MemoryType,
  type SearchMemoriesResult,
  type SemanticMemory,
} from "@memmachine/client";
import {
  jsonResult,
  readNumberParam,
  readStringParam,
  type OpenClawPluginApi,
} from "openclaw/plugin-sdk";

type PluginConfig = {
  apiKey?: string;
  baseUrl?: string;
  userId?: string;
  orgId?: string;
  projectId?: string;
  autoCapture?: boolean;
  autoRecall?: boolean;
  searchThreshold?: number;
  topK?: number;
};

type MemoryScope = "session" | "long-term" | "all";

type MemoryHandle = {
  client: MemMachineClient;
  memory: ReturnType<ReturnType<MemMachineClient["project"]>["memory"]>;
  config: Required<Pick<PluginConfig, "orgId" | "projectId">> & PluginConfig;
};

const DEFAULT_TOP_K = 5;
const DEFAULT_SEARCH_THRESHOLD = 0.5;
const DEFAULT_FORGET_THRESHOLD = 0.85;
const DEFAULT_PAGE_SIZE = 10;

const PluginConfigJsonSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    apiKey: { type: "string" },
    baseUrl: { type: "string" },
    userId: { type: "string" },
    orgId: { type: "string" },
    projectId: { type: "string" },
    autoCapture: { type: "boolean" },
    autoRecall: { type: "boolean" },
    searchThreshold: { type: "number" },
    topK: { type: "number" },
  },
  required: [],
} as const;

const MemorySearchSchema = Type.Object({
  query: Type.String({ description: "Search query" }),
  scope: Type.Optional(Type.Union([Type.Literal("session"), Type.Literal("all")])),
  limit: Type.Optional(Type.Number({ description: "Max results" })),
  minScore: Type.Optional(Type.Number({ description: "Minimum score (0-1)" })),
});

const MemoryStoreSchema = Type.Object({
  text: Type.String({ description: "Memory content" }),
  role: Type.Optional(
    Type.Union([
      Type.Literal("user"),
      Type.Literal("assistant"),
      Type.Literal("system"),
    ]),
  ),
  metadata: Type.Optional(Type.Record(Type.String(), Type.String())),
  types: Type.Optional(
    Type.Array(Type.Union([Type.Literal("episodic"), Type.Literal("semantic")]))
  ),
});

const MemoryGetSchema = Type.Object({
  id: Type.String({ description: "Memory ID" }),
  type: Type.Optional(
    Type.Union([
      Type.Literal("episodic"),
      Type.Literal("semantic"),
      Type.Literal("auto"),
    ]),
  ),
});

const MemoryListSchema = Type.Object({
  scope: Type.Optional(
    Type.Union([
      Type.Literal("session"),
      Type.Literal("long-term"),
      Type.Literal("all"),
    ]),
  ),
  pageSize: Type.Optional(Type.Number({ description: "Page size" })),
  pageNum: Type.Optional(Type.Number({ description: "Page number" })),
});

const MemoryForgetSchema = Type.Object({
  memoryId: Type.Optional(Type.String({ description: "Memory ID" })),
  query: Type.Optional(Type.String({ description: "Search query" })),
  scope: Type.Optional(
    Type.Union([
      Type.Literal("session"),
      Type.Literal("long-term"),
      Type.Literal("all"),
    ]),
  ),
  minScore: Type.Optional(Type.Number({ description: "Auto-delete threshold" })),
});

function resolvePluginConfig(api: OpenClawPluginApi): PluginConfig {
  const raw = (api.pluginConfig ?? {}) as Record<string, unknown>;
  return {
    apiKey: typeof raw.apiKey === "string" ? raw.apiKey.trim() : undefined,
    baseUrl: typeof raw.baseUrl === "string" ? raw.baseUrl.trim() : undefined,
    userId: typeof raw.userId === "string" ? raw.userId.trim() : undefined,
    orgId: typeof raw.orgId === "string" ? raw.orgId.trim() : undefined,
    projectId: typeof raw.projectId === "string" ? raw.projectId.trim() : undefined,
    autoCapture: typeof raw.autoCapture === "boolean" ? raw.autoCapture : undefined,
    autoRecall: typeof raw.autoRecall === "boolean" ? raw.autoRecall : undefined,
    searchThreshold:
      typeof raw.searchThreshold === "number" ? raw.searchThreshold : undefined,
    topK: typeof raw.topK === "number" ? raw.topK : undefined,
  };
}

function resolveApiKey(cfg: PluginConfig): string | undefined {
  const envKey = typeof process !== "undefined" ? process.env.MEMMACHINE_API_KEY : undefined;
  return cfg.apiKey || envKey?.trim() || undefined;
}

function requireProjectConfig(
  cfg: PluginConfig,
): Required<Pick<PluginConfig, "orgId" | "projectId">> {
  if (!cfg.orgId || !cfg.projectId) {
    throw new Error("Missing orgId/projectId in plugin config.");
  }
  return { orgId: cfg.orgId, projectId: cfg.projectId };
}

function sanitizeFilterValue(value: string): string {
  return value.replace(/'/g, "");
}

function buildScopeFilter(
  scope: MemoryScope,
  sessionKey?: string,
  userId?: string,
): string | undefined {
  const safeSession = sessionKey ? sanitizeFilterValue(sessionKey) : undefined;
  const safeUser = userId ? sanitizeFilterValue(userId) : undefined;

  if (scope === "session") {
    if (!safeSession) {
      return undefined;
    }
    return `metadata.run_id = '${safeSession}'`;
  }

  if (scope === "long-term") {
    if (!safeUser) {
      return undefined;
    }
    return `metadata.user_id = '${safeUser}'`;
  }

  if (safeSession && safeUser) {
    return `(metadata.run_id = '${safeSession}' OR metadata.user_id = '${safeUser}')`;
  }
  if (safeSession) {
    return `metadata.run_id = '${safeSession}'`;
  }
  if (safeUser) {
    return `metadata.user_id = '${safeUser}'`;
  }
  return undefined;
}

function normalizeScope(value: string | undefined, fallback: MemoryScope): MemoryScope {
  if (value === "session" || value === "long-term" || value === "all") {
    return value;
  }
  return fallback;
}

function resolveMemoryHandle(api: OpenClawPluginApi, ctx: { sessionKey?: string }): MemoryHandle {
  const cfg = resolvePluginConfig(api);
  const { orgId, projectId } = requireProjectConfig(cfg);
  const apiKey = resolveApiKey(cfg);
  const client = new MemMachineClient({ api_key: apiKey, base_url: cfg.baseUrl });
  const memory = client.project({ org_id: orgId, project_id: projectId }).memory({
    user_id: cfg.userId,
    session_id: ctx.sessionKey,
  });
  return {
    client,
    memory,
    config: { ...cfg, orgId, projectId },
  };
}

function toMetadata(
  base: Record<string, string> | undefined,
  extras: Record<string, string | undefined>,
): Record<string, string> {
  const merged: Record<string, string> = { ...(base ?? {}) };
  for (const [key, value] of Object.entries(extras)) {
    if (value) {
      merged[key] = value;
    }
  }
  return merged;
}

function extractEpisodicEpisodes(result: SearchMemoriesResult | null): EpisodicMemory[] {
  const episodic = result?.content?.episodic_memory;
  if (!episodic) {
    return [];
  }
  const longTerm = episodic.long_term_memory?.episodes ?? [];
  const shortTerm = episodic.short_term_memory?.episodes ?? [];
  return [...longTerm, ...shortTerm].filter(Boolean);
}

function extractSemanticMemories(result: SearchMemoriesResult | null): SemanticMemory[] {
  return result?.content?.semantic_memory ?? [];
}

function dedupeBy<T>(items: T[], key: (item: T) => string): T[] {
  const seen = new Set<string>();
  const result: T[] = [];
  for (const item of items) {
    const id = key(item);
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    result.push(item);
  }
  return result;
}

async function listMemories(params: {
  handle: MemoryHandle;
  scope: MemoryScope;
  sessionKey?: string;
  pageSize?: number;
  pageNum?: number;
}): Promise<{ episodic: EpisodicMemory[]; semantic: SemanticMemory[] } | null> {
  const { handle, scope, sessionKey, pageSize, pageNum } = params;
  const filter = buildScopeFilter(scope, sessionKey, handle.config.userId);
  const listOptions = {
    page_size: pageSize,
    page_num: pageNum,
    filter: filter ?? "",
  };

  try {
    const episodicResult = await handle.memory.list({
      ...listOptions,
      type: "episodic",
    });
    const semanticResult = await handle.memory.list({
      ...listOptions,
      type: "semantic",
    });

    const episodic = dedupeBy(extractEpisodicEpisodes(episodicResult), (entry) => entry.uid);
    const semantic = dedupeBy(
      extractSemanticMemories(semanticResult),
      (entry) => entry.metadata?.id ?? "",
    );

    return { episodic, semantic };
  } catch {
    return null;
  }
}

function formatRecallContext(params: {
  episodic: EpisodicMemory[];
  semantic: SemanticMemory[];
  limit: number;
}): string {
  const { episodic, semantic, limit } = params;
  const lines: string[] = [];

  for (const entry of episodic.slice(0, limit)) {
    const score = typeof entry.score === "number" ? ` (${entry.score.toFixed(2)})` : "";
    lines.push(`- [episodic] ${entry.content}${score}`);
  }

  for (const entry of semantic.slice(0, Math.max(0, limit - lines.length))) {
    lines.push(`- [semantic] ${entry.feature_name}: ${entry.value}`);
  }

  return lines.join("\n");
}

async function deleteById(handle: MemoryHandle, id: string): Promise<{
  ok: boolean;
  type?: MemoryType;
  error?: string;
}> {
  try {
    await handle.memory.delete(id, "episodic");
    return { ok: true, type: "episodic" };
  } catch {
    try {
      await handle.memory.delete(id, "semantic");
      return { ok: true, type: "semantic" };
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) };
    }
  }
}

function extractMessageTextBlocks(message: Record<string, unknown>): string[] {
  const content = message.content;
  if (typeof content === "string") {
    return [content];
  }
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (block && typeof block === "object" && "type" in block) {
          const record = block as Record<string, unknown>;
          if (record.type === "text" && typeof record.text === "string") {
            return record.text;
          }
        }
        return null;
      })
      .filter((text): text is string => Boolean(text));
  }
  return [];
}

async function autoCaptureMessages(params: {
  api: OpenClawPluginApi;
  handle: MemoryHandle;
  sessionKey?: string;
  messages: unknown[];
}): Promise<void> {
  const { api, handle, sessionKey, messages } = params;
  const recent = messages.slice(-8); // Only consider the last 3 messages for auto-capture
  let stored = 0;

  for (const msg of recent) {
    if (!msg || typeof msg !== "object") {
      continue;
    }
    const record = msg as Record<string, unknown>;
    const role = typeof record.role === "string" ? record.role : "user";
    if (role !== "user" && role !== "assistant" && role !== "system") {
      continue;
    }
    const blocks = extractMessageTextBlocks(record);
    for (const text of blocks) {
      if (!text || text.trim().length < 5) {
        continue;
      }
      if (text.includes("relevant-memories")) {
        continue; // Skip system-inserted recall contexts
      }
      try {
        await handle.memory.add(text, {
          role,
          producer: role,
          metadata: toMetadata(undefined, {
            run_id: sessionKey,
            user_id: handle.config.userId,
          }),
        });
        stored += 1;
      } catch (err) {
        api.logger.warn(`openclaw-memmachine: auto-capture failed: ${String(err)}`);
      }
    }
  }

  if (stored > 0) {
    api.logger.info(`openclaw-memmachine: auto-captured ${stored} memories`);
  }
}

const memmachinePlugin = {
  id: "openclaw-memmachine",
  name: "MemMachine",
  description: "MemMachine-backed memory tools with auto recall/capture",
  kind: "memory" as const,
  configSchema: {
    jsonSchema: PluginConfigJsonSchema,
  },

  register(api: OpenClawPluginApi) {
    const cfg = resolvePluginConfig(api);

    // Tools
    api.registerTool(
      (ctx) => ({
        name: "memory_search",
        label: "Memory Search",
        description: "Search memories with scope: session | all.",
        parameters: MemorySearchSchema,
        async execute(_toolCallId, params) {
          const query = readStringParam(params, "query", { required: true });
          const scope = normalizeScope(readStringParam(params, "scope"), "all");
          if (scope === "session" && !ctx.sessionKey) {
            return jsonResult({ error: "Session scope requires sessionKey" });
          }
          const limit = readNumberParam(params, "limit") ?? cfg.topK ?? DEFAULT_TOP_K;
          const minScore =
            readNumberParam(params, "minScore") ??
            cfg.searchThreshold ??
            DEFAULT_SEARCH_THRESHOLD;

          const handle = resolveMemoryHandle(api, { sessionKey: ctx.sessionKey });
          const filter = buildScopeFilter(scope, ctx.sessionKey, handle.config.userId) ?? "";

          const result = await handle.memory.search(query, {
            top_k: limit,
            score_threshold: minScore,
            filter,
          });

          return jsonResult({
            scope,
            filter,
            result,
          });
        },
      }),
      { name: "memory_search" },
    );

    api.registerTool(
      (ctx) => ({
        name: "memory_store",
        label: "Memory Store",
        description: "Store memory in the current session (run_id).",
        parameters: MemoryStoreSchema,
        async execute(_toolCallId, params) {
          const text = readStringParam(params, "text", { required: true });
          if (!ctx.sessionKey) {
            return jsonResult({ error: "No active session for memory_store" });
          }
          const role = readStringParam(params, "role");
          const types = params.types as MemoryType[] | undefined;
          const metadata = (params.metadata ?? {}) as Record<string, string>;
          const handle = resolveMemoryHandle(api, { sessionKey: ctx.sessionKey });
          const producer = role ?? "user";

          const result: AddMemoryResult = await handle.memory.add(text, {
            role: role as "user" | "assistant" | "system" | undefined,
            producer,
            types,
            metadata: toMetadata(metadata, {
              run_id: ctx.sessionKey,
              user_id: handle.config.userId,
            }),
          });

          return jsonResult({ result });
        },
      }),
      { name: "memory_store" },
    );

    api.registerTool(
      (ctx) => ({
        name: "memory_get",
        label: "Memory Get",
        description: "Fetch memory by ID.",
        parameters: MemoryGetSchema,
        async execute(_toolCallId, params) {
          const id = readStringParam(params, "id", { required: true });
          const type = readStringParam(params, "type") ?? "auto";
          const handle = resolveMemoryHandle(api, { sessionKey: ctx.sessionKey });

          const idFilter = `uid = '${sanitizeFilterValue(id)}'`;
          const metadataFilter = `metadata.id = '${sanitizeFilterValue(id)}'`;

          let episodic: EpisodicMemory[] = [];
          let semantic: SemanticMemory[] = [];

          if (type === "episodic" || type === "auto") {
            const result = await handle.memory.list({
              type: "episodic",
              filter: idFilter,
              page_size: 1,
            });
            episodic = extractEpisodicEpisodes(result);
          }

          if (type === "semantic" || (type === "auto" && episodic.length === 0)) {
            const result = await handle.memory.list({
              type: "semantic",
              filter: metadataFilter,
              page_size: 1,
            });
            semantic = extractSemanticMemories(result);
          }

          return jsonResult({ id, episodic, semantic });
        },
      }),
      { name: "memory_get" },
    );

    api.registerTool(
      (ctx) => ({
        name: "memory_list",
        label: "Memory List",
        description: "List memories by scope: session | all (deduped).",
        parameters: MemoryListSchema,
        async execute(_toolCallId, params) {
          const scope = normalizeScope(readStringParam(params, "scope"), "all");
          if (scope === "session" && !ctx.sessionKey) {
            return jsonResult({ error: "Session scope requires sessionKey" });
          }
          const pageSize =
            readNumberParam(params, "pageSize", { integer: true }) ?? DEFAULT_PAGE_SIZE;
          const pageNum = readNumberParam(params, "pageNum", { integer: true }) ?? 0;
          const handle = resolveMemoryHandle(api, { sessionKey: ctx.sessionKey });
          const result = await listMemories({
            handle,
            scope,
            sessionKey: ctx.sessionKey,
            pageSize,
            pageNum,
          });

          return jsonResult({ scope, pageSize, pageNum, result });
        },
      }),
      { name: "memory_list" },
    );

    api.registerTool(
      (ctx) => ({
        name: "memory_forget",
        label: "Memory forget",
        description: "Forget memory by memoryId or search+forget (high-confidence).",
        parameters: MemoryForgetSchema,
        async execute(_toolCallId, params) {
          const memoryId = readStringParam(params, "memoryId");
          const query = readStringParam(params, "query");
          const minScore =
            readNumberParam(params, "minScore") ?? DEFAULT_FORGET_THRESHOLD;
          const scope = normalizeScope(readStringParam(params, "scope"), "all");
          if (scope === "session" && !ctx.sessionKey) {
            return jsonResult({ error: "Session scope requires sessionKey" });
          }
          const handle = resolveMemoryHandle(api, { sessionKey: ctx.sessionKey });

          if (memoryId) {
            const result = await deleteById(handle, memoryId);
            return jsonResult({ action: "forget", memoryId, result });
          }

          if (!query) {
            return jsonResult({ error: "Provide memoryId or query" });
          }

          const filter = buildScopeFilter(scope, ctx.sessionKey, handle.config.userId) ?? "";
          const searchResult = await handle.memory.search(query, {
            top_k: cfg.topK ?? DEFAULT_TOP_K,
            score_threshold: minScore,
            filter,
          });

          const episodic = extractEpisodicEpisodes(searchResult)
            .filter((entry) => typeof entry.score === "number")
            .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

          if (episodic.length === 0) {
            return jsonResult({ action: "search", found: 0 });
          }

          const [best, second] = episodic;
          if ((best.score ?? 0) >= minScore && (second?.score ?? 0) < minScore) {
            const result = await deleteById(handle, best.uid);
            return jsonResult({
              action: "auto-delete",
              memoryId: best.uid,
              score: best.score,
              result,
            });
          }

          const candidates = episodic.slice(0, 5).map((entry) => ({
            uid: entry.uid,
            content: entry.content,
            score: entry.score,
          }));

          return jsonResult({ action: "candidates", candidates });
        },
      }),
      { name: "memory_forget" },
    );

    // CLI commands
    api.registerCli(
      ({ program, logger }) => {
        const root = program.command("memmachine").description("MemMachine commands");

        root
          .command("search")
          .description("Search memories")
          .argument("<query>", "Search query")
          .option("--limit <n>", "Max results", `${cfg.topK ?? DEFAULT_TOP_K}`)
          .option("--scope <scope>", "session | long-term | all", "all")
          .action(async (query: string, opts: { limit: string; scope: string }) => {
            try {
              const handle = resolveMemoryHandle(api, { sessionKey: undefined });
              const limit = Number.parseInt(opts.limit, 10) || DEFAULT_TOP_K;
              const scope = normalizeScope(opts.scope, "all");
              const filter = buildScopeFilter(scope, undefined, handle.config.userId) ?? "";
              const result = await handle.memory.search(query, {
                top_k: limit,
                score_threshold: cfg.searchThreshold ?? DEFAULT_SEARCH_THRESHOLD,
                filter,
              });
              console.log(JSON.stringify(result, null, 2));
            } catch (err) {
              logger.error(`memmachine search failed: ${String(err)}`);
            }
          });

        root
          .command("stats")
          .description("Show memory stats")
          .action(async () => {
            try {
              const handle = resolveMemoryHandle(api, { sessionKey: undefined });
              const project = handle.client.project({
                org_id: handle.config.orgId,
                project_id: handle.config.projectId,
              });
              const total = await project.getEpisodicCount();
              const flags = {
                autoCapture: cfg.autoCapture ?? false,
                autoRecall: cfg.autoRecall ?? false,
              };
              const mode = resolveApiKey(cfg) ? "platform" : "unknown";

              console.log(
                JSON.stringify(
                  {
                    mode,
                    user: handle.config.userId ?? "",
                    total,
                    flags,
                  },
                  null,
                  2,
                ),
              );
            } catch (err) {
              logger.error(`memmachine stats failed: ${String(err)}`);
            }
          });
      },
      { commands: ["memmachine"] },
    );

    // Hooks
    if (cfg.autoRecall) {
      api.on("before_agent_start", async (event, ctx) => {
        if (!event.prompt || event.prompt.trim().length < 3) {
          return;
        }
        try {
          const handle = resolveMemoryHandle(api, { sessionKey: ctx.sessionKey });
          const filter = buildScopeFilter("all", ctx.sessionKey, handle.config.userId) ?? "";
          const result = await handle.memory.search(event.prompt, {
            top_k: cfg.topK ?? DEFAULT_TOP_K,
            score_threshold: cfg.searchThreshold ?? DEFAULT_SEARCH_THRESHOLD,
            filter,
          });
          const episodic = extractEpisodicEpisodes(result);
          const semantic = extractSemanticMemories(result);
          if (episodic.length === 0 && semantic.length === 0) {
            return;
          }
          const context = formatRecallContext({
            episodic,
            semantic,
            limit: cfg.topK ?? DEFAULT_TOP_K,
          });
          return {
            prependContext:
              `<relevant-memories>\n` +
              `The following memories may be relevant to this conversation:\n` +
              `${context}\n` +
              `</relevant-memories>`,
          };
        } catch (err) {
          api.logger.warn(`openclaw-memmachine: recall failed: ${String(err)}`);
        }
      });
    }

    if (cfg.autoCapture) {
      api.on("agent_end", async (event, ctx) => {
        if (!event.success || !Array.isArray(event.messages) || event.messages.length === 0) {
          return;
        }
        try {
          const handle = resolveMemoryHandle(api, { sessionKey: ctx.sessionKey });
          await autoCaptureMessages({
            api,
            handle,
            sessionKey: ctx.sessionKey,
            messages: event.messages,
          });
        } catch (err) {
          api.logger.warn(`openclaw-memmachine: capture failed: ${String(err)}`);
        }
      });
    }

    // Service
    api.registerService({
      id: "openclaw-memmachine",
      start: () => {
        api.logger.info("openclaw-memmachine: initialized");
      },
      stop: () => {
        api.logger.info("openclaw-memmachine: stopped");
      },
    });
  },
};

export default memmachinePlugin;

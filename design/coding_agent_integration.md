# Coding-agent integration: Claude Code and Codex over the MemMachine server

Memory for coding agents, served by a MemMachine server through the event
memory (events, segments, derivatives) rather than the current server API's
episodes, and reached by the agents through their own extension points:
MCP tools for recall, lifecycle hooks for capture. Claude Code and Codex
first; OpenCode and others later. This document fixes what is built first,
what is deferred, and which decisions are expensive to reverse.

It carries over what `agentic_expansion/claude_memory/DESIGN.md` settled
for the in-process version (one daemon, local embedder, SQLite stores) and
what `design/server_redesign.md` ("Server API") already specifies for the
tenant-scoped HTTP API, so that the integration is a client of the redesign's
API from the first PR, on today's server.

## 1. What is carried over, and what is not

Carried over from `claude_memory`:

- Two recall modes. Deliberate recall through tools the model calls
  (`memory_query`, `memory_expand`), which is where multi-hop happens
  (query, read, query the lead). Ambient recall through a
  `UserPromptSubmit` hook stays designed but off, per the 2026-08
  measurement: about a quarter of injections changed the answer, 1.3%
  caused a correction, and no benefit signal exists to weigh against it.
- One timeline, one search surface. Everything a session does lands on a
  session's timeline (messages, tool calls, tool results, injected text),
  and only messages are embedded. Tool calls and results are reached by
  expansion from a message seed, never searched.
- Append-only injection. Tool results and hook context are appends; there
  is no mutable memory panel and no mid-session discard, so the agent's
  prompt cache is preserved.
- The tool surface's rules: a cue re-evokes the context a memory was
  encoded in, not a bare entity; following a lead is another query, not
  a tool; an expansion step has a fixed size and a direction, and the
  model decides only whether to step again.
- Ids are handles the store can resolve, never a registry.

Not carried over:

- The daemon and the in-process stores. The server owns the embedder and
  the stores; the agent-side pieces are thin clients of HTTP.
- The tool names' `mem:` prefix. Ids are the markers `render_segments`
  writes (#1632): `[session:"<id>"]`, `[segment:<hex32>]`,
  `[segments:<first>..<last>]`.
- `memory_outline`, `memory_demote`, `memory_annotate`. Outline and table
  of contents depend on the agent's transcript retention and on features
  the timeline does not have yet (below, "Deferred"); demote and annotate
  are the daemon's and need server-side design of their own.

## 2. Architecture

One verb for the operation, from `EventMemory.query` up: the route is
`query`, the tool is `memory_query`, and the hits are query hits.
"Search" names only the vector store's technique.

Three layers, each with one job.

```
Claude Code / Codex                      MemMachine server
  MCP client  ── streamable HTTP ──►  /v1/mcp   (memory_query, memory_expand)
  Stop hook   ── HTTPS JSON ───────►  /v1/tenants/{tenant}/events        (capture, slice 3)
  installer   ── writes the agent's config (MCP server entry, hooks)
```

### 2.1 Server: the v1 event-memory API

A new router, prefix `/v1`, with the routes `server_redesign.md` specifies
and the same bodies, implemented over the objects today's server already
has. It does not go through the v2 router, the `Episode` model, or the
v2 MCP tools (`api_v2/mcp.py`), which are built on episodes.

| Method and path | Effect |
| --- | --- |
| `POST /v1/tenants/{tenant}/episodic-memory/query` | `EventMemory.query`, then `rerank` when the tenant has a reranker and the body asks; hits rendered with `render_segments` |
| `POST /v1/tenants/{tenant}/episodic-memory/expand` | `EventMemory.expand` from a segment uuid, `before` and `after` in segments, rendered |
| `POST /v1/tenants/{tenant}/events` | `EventMemory.encode_events` with `Event` bodies; whole batch rejected with 409 when any event is already held (#1659) |
| `POST /v1/tenants/{tenant}/events/delete` | `EventMemory.forget_events` |

Bodies and hit shapes are the redesign's, verbatim, with two departures
that today's server forces and that are called out in the OpenAPI
description so a client written now keeps working later:

- `{tenant}` is a tenant *name*, an opaque string the server resolves, not
  a UUID. A tenant is, roughly, one human user: every session of every
  agent that user runs writes into it, and sessions carry no lifecycle
  of their own, they are ids on events. Today the name is resolved to a
  partition through the current episodic memory manager, under a
  namespace of its own so v1 tenants and the v2 API's sessions cannot
  collide or clobber each other; when the tenant registry lands, the
  same name is looked up in it (`GET /v1/tenants?name=`). The router
  never sees a lifecycle: it asks one seam,
  `resolve_event_memory(tenant) -> EventMemory` (and the tenant's
  reranker and defaults), and the seam's implementation is what changes
  with the redesign. This is what "tenant-lifecycle-agnostic" means
  here: the address is stable, the resolution is replaceable. The v1
  lifecycle is the minimum the address needs, `PUT /v1/tenants/{tenant}`
  to create (idempotent) and `DELETE` to remove; the data routes answer
  404 `tenant_not_found` for a name never created.
- No `position` on segments and no `watermark` route: there is no event
  store yet. Ingest is synchronous (`encode_events` returns when the
  segments and records are written), so `?wait=` is accepted and ignored.

Errors map as the redesign says (`{error: {code, message}}`, closed set of
codes); `SegmentStoreEventAlreadyStoredError` is 409 `event_exists` with
the uuids in the message. Query and expand defaults (`limit`,
`expand_context`, `before`, `after`, rerank candidates) come from the
deployment's episodic-memory settings, so the tools need not name them.

### 2.2 Server: MCP at `/v1/mcp`

One MCP server, streamable HTTP, mounted by the same server, with the
tenant taken from a request header (`X-MemMachine-Tenant`), which is the
rule the redesign carries over for MCP. Both agents speak HTTP MCP with
static headers (Claude Code `claude mcp add --transport http ... --header`,
Codex `mcp_servers.<id>.url` and `http_headers`), so one implementation
serves both, and a deployment upgrades the tools without touching any
agent's machine. A stdio shim per agent was the daemon's shape and is not
needed when the server is remote.

Tools, over the v1 services (not over HTTP to itself):

```
memory_query(cue: str, within: str | None = None,
              kinds: list[str] | None = None,
              since: str | None = None, until: str | None = None) -> str
memory_expand(id: str, direction: "around" | "earlier" | "later" = "around") -> str
```

- `memory_query` runs `query` with the tenant's default limit, expansion
  and rerank, then renders every hit's window with
  `render_segments(ids=("session", "segment"))`, hits separated by a blank
  line, best first. `within` is a session id (the value inside
  `[session:"..."]`), `kinds` block kinds, `since` and `until` ISO
  timestamps. No count parameter is exposed: the redesign's rule is that
  segments are not an intuitive unit for a model, so the deployment
  chooses the expansion and the model asks for context, not for an
  amount of it.
- `memory_expand` takes a segment marker's hex (with or without the
  `[segment:...]` wrapper) and a direction. `around` spends a quarter of
  the tenant's expansion budget backward and the rest forward; `earlier`
  and `later` spend it all on one side, from the given segment outward.
  The window is rendered with ids, so its two edges carry the handles a
  further step continues from; a side that comes back empty means the
  session ran out that way.
- Tool descriptions carry the cue guidance from `claude_memory` (context
  over bare entities, the user's wording is a fine cue, query the
  surroundings and expand when the target cannot be pinned) and the
  reading rule for markers. They are the only prompting the integration
  does.

### 2.3 Agents

Claude Code: an MCP server entry (`claude mcp add --transport http
memmachine <server>/v1/mcp --header "X-MemMachine-Tenant: <tenant>"`,
user scope) for recall; hooks for capture in slice 3: `Stop` runs the
capture client with the hook's `transcript_path` and `session_id`.
`SessionStart` needs nothing (no daemon to warm). `UserPromptSubmit` is
the ambient channel, registered off.

Codex: `~/.codex/config.toml` gets `[mcp_servers.memmachine]` with `url`
and `http_headers`; `~/.codex/hooks.json` gets the same `Stop` handler in
slice 3. Codex's hooks (`SessionStart`, `UserPromptSubmit`, `Stop`,
`PreCompact`, `PostCompact`, `SessionEnd`, ...) receive `session_id`,
`transcript_path` and `cwd` on stdin and accept `additionalContext` on
stdout, the same contract as Claude Code's, so the capture client and the
ambient client are one script each with an agent-specific transcript
parser.

An installer, `memmachine agent install {claude-code,codex} --server
<url> --tenant <name> [--scope user|project]`, writes those entries
idempotently, backs the config up, and `memmachine agent disable` removes
them. It lives in the client package, since it is client-side glue.

### 2.4 Tenant, session, source

- Tenant: the human user, named by the installer (`--tenant`), so every
  session of every agent that user runs on that machine shares one
  memory, the "shared space" the in-process version moved to. A project
  is an agent concept, not a MemMachine one, so it is a user-defined
  property: capture stamps `properties.project` (the repo root or cwd)
  on every event, and `memory_query` accepts no project argument now (a
  later `where` can).
- Session: the agent's own session id (`session_id` from the hooks), so a
  walk stays inside one agent session, which is the conversation.
  Subagent transcripts are separate sessions with the parent's id in
  `properties.parent_session`.
- Source: the agent (`claude-code`, `codex`), so a query can be confined
  to one agent's memory or span both.

### 2.5 Ingestion is never the model's decision

No tool the model can call writes to memory: the MCP app at `/v1/mcp`
serves `memory_query` and `memory_expand` and nothing else, and a test
pins that surface. Capture is the `Stop` hook's alone, and it posts every
entry the transcript holds since the session's mark: messages, tool
calls, tool results, injected text, and the agent's reasoning, each under
its block kind, with nothing chosen or omitted by the model. The one
thing left out is a restatement of an entry already captured (Codex's
`event_msg` records restate its `response_item`s), since that is the same
entry twice, not a choice about content.

## 3. Ids

Full ids now: the tools read and write the marker grammar of #1632, and
`memory_expand` takes the 32-hex segment uuid. Shortening is a client
concern by construction (the marker grammar is regular, so a client can
substitute unique prefixes on the way out and expand them on the way in),
and it is deferred until the tools are in use and the token cost of full
ids is measured against real transcripts. Nothing in the server changes
for it: the server always speaks full ids.

## 4. Capture (slice 3, with the installer)

The `Stop` hook reads the transcript from the hook's `transcript_path`,
converts the entries since the session's high-water mark into `Event`s,
and posts them to `/v1/tenants/{tenant}/events`.

- Event identity: the transcript entry's own uuid (Claude Code entries
  carry one; Codex's `history.jsonl` entries are to be checked, and an
  entry without one gets `uuid5(session_id, entry_index)` on the client
  side, which is the client's convention and not the store's). Retries of
  a batch are then either wholly rejected (already held) or wholly
  stored, since the store rejects whole batches and writes whole batches
  (#1659); the client treats 409 `event_exists` as done and advances its
  mark.
- One event per message; tool calls, tool results, injected text and
  the agent's reasoning as events of their own block kinds (`tool_call`,
  `tool_result`, `injected`, `thinking`), registered with the server
  (#1611, `blocks.md`, #1692), so a
  deriver that embeds only message kinds is a table lookup and the tool
  events stay reachable by expansion. The policy is the deriver's and
  applies to what is ingested from then on: widening it later needs no
  rebuild, only a re-derivation of what should have been embedded. Injected text (hook context, skill
  bodies, compaction summaries) is classified at capture as in
  `claude_memory`'s `wire.user_text_source` and lands as its own kind, on
  the timeline and off the search surface.
- `session_id`, `source_id`, `context` (`Author` for the speaker),
  `properties.project`, `properties.tool_name` where there is one.
- The high-water mark is per (agent, session), kept in the agent's own
  state directory, and is only a shortcut: a lost mark costs one rejected
  batch, never a duplicate.

## 5. Deferred, and why

- Session outline and table of contents. Both depend on the agent: what a
  "turn" is, how subagents nest, and how long the agent keeps its
  transcripts (Claude Code deletes transcripts after `cleanupPeriodDays`;
  Codex caps `history.jsonl` by `history.max_bytes`), so an outline built
  from agent files decays, and one built from the memory's timeline needs
  turn boundaries the timeline does not record yet. Follow-up PRs, per
  agent, after capture exists to record what they need.
- Id shortening (section 3).
- Ambient recall (registered off).
- Demote and annotate (server-side design needed).
- OpenCode and other agents: once the two-agent shape has settled which
  parts are shared.

## 6. PR plan, as built (2026-09-17)

Three PRs on main, `[coding agents N/3]`, plus one prerequisite in the
event-memory line:

1. #1690, the v1 API, on the event-memory port stack.
2. #1691, the MCP tools, on #1692.
3. #1693, the whole client side: the installer, the `Stop` hook it
   registers, and capture; directly on main, merging after 2 and #1692.

#1692, the block kinds `thinking`, `tool_call`, `tool_result` and
`injected`, sits between 1 and 2 as event-memory data model rather than
as a slice of this stack, since it is what the event memory holds, not
what an agent does with it.

Two facts settled in the building: a tool result is one segment however
long, so the capture client caps tool output, injected text and reasoning
at 8 KB with a truncation marker; and a batch the server already holds
in part, as a resumed or forked transcript produces, is posted again one
event at a time with the mark moving past each held event, so nothing is
lost or duplicated and a slow link makes progress on every `Stop`. One
finding: every Codex rollout on the development machine carries its
reasoning encrypted with an empty summary, so Codex reasoning records
produce nothing until a deployment's Codex emits summaries.

## 7. Decisions taken (2026-09-17)

The ones that decide where data lands or what clients depend on, settled
before any code:

1. Tenant addressing by name in the path and header, resolved by a seam,
   rather than by the registry's UUID. A tenant is roughly one human
   user, with any number of lifecycle-free sessions. The v1 tenant
   lifecycle may diverge from the v2 API's entirely, as long as neither
   steps on the other.
2. Projects are agent concepts and therefore user-defined properties,
   not tenants and not MemMachine's org and project.
3. The captured event's uuid is the transcript entry's uuid, the dedupe
   key for the life of the tenant.
4. Messages only on the search surface, tool events on the timeline only,
   as the initial deriver policy; it applies to new ingestions, so
   changing it is not a rebuild.
5. MCP served by the server over HTTP, one implementation for every
   agent, the tenant in a header.

# MemMachine deployment sizing calculator

Work out what hardware a MemMachine deployment needs to serve a given amount of
traffic.

You give it a **design peak** — the worst rate the system must sustain for five
minutes, in operations per second — and a traffic mix, and it tells you how many
API servers, vector-store machines and PostgreSQL servers to order, how many
embedding and agent-model GPU cards, how much RAM and disk the stored episodes
will take, how many PostgreSQL connections the deployment will open, and how
much network bandwidth it will use.

## Two things that sound alike and are not

Read this before you set anything. Two ideas in this calculator use words that
look similar, and a reader who mixes them up will size the deployment wrongly.

**Agent-mode search is a property of a request.** It is the `agent_mode` flag on
a MemMachine search. A search sent with that flag on does not do one lookup; it
fans out into a multi-hop retrieval of about 22 embedding calls, 22 vector
searches, 44 database reads and one or two language-model calls. It is set per
request. It is the third share of the traffic mix, and the flag that sets it is
`--agent`. It is a **cost** multiplier: it says how expensive a request is.

**An automated client is a property of a caller.** It is a program that sends
requests in a loop rather than a person typing in a chat window. One in a
five-second tool loop is assumed to send about 0.4 operations per second, where
a human chat session sends 0.011 to 0.028. It is a population count on the
`users` subcommand, and the flag that sets it is `--automated`. It is a **rate**
multiplier: it says how fast a caller sends requests.

The two are independent. A person can send agent-mode searches, and an automated
client can send nothing but plain searches. That is why each population carries
its own traffic mix: see [`users`](#users--a-caller-population-to-the-capacity-it-needs).

Every number in the report's tables of findings carries one of four labels, so
you can always see how much weight it will bear:

| Label | Meaning |
| --- | --- |
| `measured` | It came out of a real benchmark run, and the label names that run: its date, the configuration it ran with, or both. |
| `derived` | The program computed it from measured numbers and from the assumptions below. |
| `estimate` | Nobody has measured it. Benchmark it before ordering hardware against it. |
| `assumption` | A planning choice, not a finding. |

Two tables are different. The **Qdrant node choice** table and the
**sensitivity** table show what-ifs rather than findings — one row of each is
the answer and the rest are what the alternatives would have cost — so their
rows carry no label of their own, and the note above each table says where its
numbers come from instead.

The JSON, from `--json` or from `/api/calc`, carries the numbers without
labels; its top-level `run_name` field is the name of the run (`pilot`,
`target`, `scale`, `custom` or `web`), not one of the four labels above.

## What it does not do

- **It does not price high availability.** A second copy of every vector, a
  PostgreSQL standby, a second gateway — none of these are in the counts. They
  are a separate decision, and a replication factor of 2 doubles both the hot
  vector RAM and the vector-store machine count.
- **It does not price a cross-encoder reranker.** The measured throughput anchor
  was taken with the `rrf-hybrid` reranker, which runs on the API server's own
  CPU. A cross-encoder scores every query-and-result pair on a GPU; it would add
  GPU cards and it would invalidate the anchor.
- **It does not choose a machine class for PostgreSQL or the vector store.**
  Only the API server has one that was benchmarked. The RAM of a vector-store
  machine is sized by the model, so the report gives it; the vCPU of either
  machine, and the RAM of the PostgreSQL machine, are undecided and the report
  says so rather than repeating the API server's figures.
- **It does not measure your deployment.** It is arithmetic over a small set of
  named inputs. Meter your real operations per second per API key and feed the
  metered rate back in.

## Requirements

Python 3.10 or newer, and nothing else. The calculator is a single file that
imports only the standard library.

## Running it

Use [uv](https://docs.astral.sh/uv/) with `--no-project`. The calculator needs
no dependencies, and `--no-project` tells uv to ignore this repository's own
project environment rather than build it just to run one script:

```bash
cd tools/sizing
uv run --no-project python memmachine_sizing.py --help
```

Plain `python memmachine_sizing.py --help` works just as well if you already
have an interpreter on your path.

## Subcommands

### `tier` — the full report for one named tier

Three tiers are built in: `pilot` (20 ops/s), `target` (100 ops/s) and `scale`
(1,000 ops/s).

```bash
uv run --no-project python memmachine_sizing.py tier target
```

Prints the inputs, the per-second demand, the machine counts, how the API server
count was reached, storage, the vector-store machine choice, PostgreSQL,
network, how many callers of each kind the capacity holds, and a sensitivity
table showing what
the agent-mode search rate costs. That table always holds the run's own
agent-mode rate as well as the four fixed ones, and marks it `<- this run`,
because a rate of 2.04 and a rate of 2.0 both print as "2.0" and are two
different sizings. Add `--json` for the raw result.

Any work at all costs a machine. A traffic mix with no adds in it stores
nothing, and so does a retention of zero days, but both still search vectors, so
the order is never fewer than one vector-store machine.

### `calc` — the full report for any rate

```bash
uv run --no-project python memmachine_sizing.py calc --ops 250 --agent 4 --plain 51
```

The same report for any design peak, traffic mix, retention period and vector
shape. `--ops` is required.

### `users` — a caller population to the capacity it needs

```bash
uv run --no-project python memmachine_sizing.py users --humans 5000 --automated 40
```

Converts a population of callers into the capacity it needs. There are two kinds
of caller: **human chat sessions**, people typing, at 0.011 to 0.028 operations
per second each; and **automated clients**, programs sending requests in a loop,
at 0.4 operations per second each. `--automated` counts callers. It has nothing
to do with `--agent`, which is the agent-mode share of the requests themselves.

Each population carries its own traffic mix, because the kind of caller and the
kind of request go together in practice: automated clients may use agent-mode
search on nearly every call while people rarely do. Give each one three numbers,
`adds/plain/agent-mode`, separated by `/` or by `,`:

```bash
uv run --no-project python memmachine_sizing.py users \
  --humans 5000 --automated 200 --human-mix 48/50/2 --automated-mix 20/20/60
```

Both mixes default to the model's own default mix, `45/45/10`, so leaving them
off changes nothing.

The report gives four things: the operations per second each population demands,
the two mixes and the **blended mix** across the whole population, the smallest
tier that holds the demand, and the machines that demand needs. The blended mix
is the two mixes averaged, each weighted by the operations its population demands
at the busy end of the human rate — the rate the report tells you to plan for.
That blended mix, and not the program's default mix, is what sizes the machines,
so a population that is mostly automated clients doing multi-hop retrieval orders
more hardware than the same operations per second at the default mix would.

The same headcount can need a pilot tier or a scale tier depending on how many of
the callers are automated clients rather than people, and on how much agent-mode
search each population does. A population of nobody demands no operations, and
the report says there is nothing to size rather than naming a tier for it.

`--agents`, which used to mean the count of automated clients, is refused by
name. It was one letter from `--agent` and meant something completely different.

### `validate` — the published figures for all three tiers

```bash
uv run --no-project python memmachine_sizing.py validate --out sizing-numbers.json
```

Prints the figures this program publishes for the `pilot`, `target` and `scale`
tiers, together with every named constant listed in
[Every input, and what it is set to](#every-input-and-what-it-is-set-to), as
`name: value` lines — `name` here is the key, not one of the four labels above.
It writes the same pairs to a JSON file: `sizing-numbers.json` in the current
directory unless `--out` says otherwise. Use it to check figures quoted
elsewhere against this program mechanically.

Which keys it writes is decided by a named list in the program rather than by
walking the model's internals, and the test suite pins that list, so a key that
quietly disappears is a failing test. Two parts of it follow the inputs: there
is one sensitivity entry per row of the printed sensitivity table, and forcing
a vector-store machine size that is not one of the three offered adds that size
to the comparison.

It is not a dump of everything the model computes. The raw byte counts behind
the GB and Mbps figures stay out. The chosen vector-store machine size is
exported in full — its usable RAM, its total RAM bought and its fill — but the
other sizes in the comparison are exported only as a machine count.

### `serve` — the web form and a JSON endpoint

```bash
uv run --no-project python memmachine_sizing.py serve --port 8899
```

Serves an HTML form at `/` and the same figures as JSON at `/api/calc`. The page
is self-contained: it loads no external stylesheets, fonts or scripts and needs
no internet access. There is also a `/healthz` endpoint that answers `ok`.

**This is a local development server, not a service.** It has no
authentication, no rate limiting and no request logging you would want to keep,
and it answers whoever can reach the port. Run it on your own machine and do
not put it on an address the public can reach. It drops a connection that stays
silent for 10 seconds (`SERVER_REQUEST_TIMEOUT_S`), so a client that connects
and then sends nothing cannot hold a thread open indefinitely.

The form binds to `127.0.0.1` unless `--host` says otherwise. It carries every
input that `tier`, `calc` and `users` accept as a flag, so nothing has to be set
by editing code. Leave the "RAM per vector-store machine" box empty — or type
`automatic` — and the size is chosen for you, exactly as it is when `--node-gb`
is not given.

Below the sizing boxes are four more for a caller population: "Human chat
sessions", "Automated clients" and a traffic mix for each. Leave both counts
empty and the page says nothing about a population, exactly as it did before it
could take one. Fill in either count and the page adds the same tables the
`users` subcommand prints — the demand from each population, the blended mix and
the machines that mix needs — above the sizing report. "Automated clients" counts
callers that are programs; the "Agent-mode searches per 100 ops" box above it is
the share of the requests themselves.

Text that is not a number at all comes back as a 400 whose reason names the box
it came from and quotes what was typed into it, and the page puts the cursor in
that box. A number that is out of range — negative, zero, or above one of the
sanity bounds — also comes back as a 400 with the reason, on the page and in the
JSON alike, but that reason is worded in the model's own terms and no box is
highlighted.

Either way an input has one name across every message about it: the box
labelled "Bytes per number" is called "bytes per number" whether what was typed
in it is zero or too large, never "bytes per value". The value is quoted as it
was given, so a refused 1,000,000,001 never prints as the 1,000,000,000 it is
said to exceed.

Every other sizing box has to be answered. A form submission always sends all
eight of them, so an empty one means the reader cleared it, and the page says
which box is blank rather than quietly sizing for the default. A parameter left
out of the URL altogether is different, and still takes the default — which is
why `/api/calc?ops=100` works. The "RAM per vector-store machine" box is the one
place where empty is itself an answer, and the two population count boxes are
the other: empty there means no callers of that kind.

One misspelling is answered specially. `agents` is not a setting, and it is not
answered with "did you mean agent?", because a reader who types `agents` almost
certainly wants `automated`, the count of callers that are programs — not the
agent-mode share of the traffic mix. The message says so.

A setting the calculator does not know comes back as a 400 rather than being
ignored. `/api/calc?ops=100&retention_day=1` — `retention_day` singular, a typo
— names the setting it does not recognise and suggests `retention_days`, on the
page and in the JSON alike, instead of quietly sizing for the 90-day default.
The suggestion ignores case, so `OPS` and `Ops` are both answered with `ops`. A
setting given twice in one web address is refused the same way, because there
is no way to tell which value was meant.

There is no favicon: `/favicon.ico` answers `204 No Content`, so a page load
leaves nothing in the browser's console.

```
http://127.0.0.1:8899/api/calc?ops=100&add=45&plain=45&agent=10&retention_days=90&dims=1024&bytes_per_value=1
```

Add `&node_gb=512` to force the vector-store machine size:

```
http://127.0.0.1:8899/api/calc?ops=100&node_gb=512
```

Add `&humans=5000&automated=40` to size from a caller population instead, with
`&human_mix=48/50/2&automated_mix=20/20/60` to give each population its own
traffic mix. The answer then carries a `population` block alongside the sizing:

```
http://127.0.0.1:8899/api/calc?ops=100&humans=5000&automated=40&human_mix=48/50/2&automated_mix=20/20/60
```

## Every input, and what it is set to

This is the whole model. Anything with a flag in the "How to set it" column is a
knob you can turn from the command line, and every input that `tier`, `calc` and
`users` accept as a flag is also a box on the web form. The flags under "Output
and serving" control how a result is printed or served, not what is sized, so
they are not boxes. Everything else is a named constant at the top of
`memmachine_sizing.py`, and you change it by editing that file.

| Input | How to set it | Default | Label |
| --- | --- | --- | --- |
| **Traffic** | | | |
| Design peak, operations per second | `--ops` (`calc`), or the tier name (`tier`) | required for `calc`; `pilot` 20, `target` 100, `scale` 1,000 | assumption |
| Built-in tier rates | `TIER_OPS_PER_S` | pilot 20, target 100, scale 1,000 ops/s | assumption |
| Adds per 100 operations | `--add` | 45 | assumption — never measured |
| Plain searches per 100 operations | `--plain` | 45 | assumption — never measured |
| Agent-mode searches per 100 operations (a request flag, not a kind of caller) | `--agent` | 10 | assumption — never measured |
| Agent-mode rates in the sensitivity table | `SENSITIVITY_AGENT_RATES` | 0, 2, 10, 25 per second | assumption (display only) |
| **Fan-out per request** | | | |
| Embedding calls per add | `ADD_EMBEDS` | 1 | derived — read from the MemMachine source, 30 Aug 2026 |
| Vector writes per add | `ADD_VECTOR_WRITES` | 1 | derived — read from the source |
| PostgreSQL statements per add | `ADD_POSTGRES_STATEMENTS` | 2 | derived — read from the source |
| Language-model calls per add | `ADD_LLM_CALLS` | 0 | derived — read from the source |
| Embedding calls per plain search | `PLAIN_EMBEDS` | 2 | derived — read from the source |
| Embedding calls per plain search, once every request sends `types: ["episodic"]` | `PLAIN_EMBEDS_WITH_TYPES_FIX` | 1 | derived — read from the source |
| Vector searches per plain search | `PLAIN_VECTOR_SEARCHES` | 1 | derived — read from the source |
| PostgreSQL statements per plain search | `PLAIN_POSTGRES_STATEMENTS` | 2 | derived — read from the source |
| Language-model calls per plain search | `PLAIN_LLM_CALLS` | 0 | derived — read from the source |
| Embedding calls per agent-mode search | `AGENT_EMBEDS` | 22 | derived — read from the source |
| Vector searches per agent-mode search | `AGENT_VECTOR_SEARCHES` | 22 | derived — read from the source |
| PostgreSQL statements per agent-mode search | `AGENT_POSTGRES_STATEMENTS` | 44 | derived — read from the source |
| Language-model calls per agent-mode search, low | `AGENT_LLM_CALLS_LOW` | 1 | estimate |
| Language-model calls per agent-mode search, high | `AGENT_LLM_CALLS_HIGH` | 2 | estimate |
| Language-model calls per agent-mode search, used for sizing | `AGENT_LLM_CALLS_PLANNING` | 1.5 | estimate — the midpoint |
| **API servers** | | | |
| Searches per second per server | `API_SEARCHES_PER_S_PER_SERVER` | 180 | measured 30 Aug 2026 |
| Utilization ceiling | `API_UTILIZATION_CEILING` | 0.60 | assumption |
| Worker processes per server | `API_WORKERS_PER_SERVER` | 8 | measured 30 Aug 2026 — 8 is the knee |
| vCPU per API server | `API_SERVER_VCPU` | 16 | measured — the machine class benchmarked on 30 Aug 2026 |
| RAM per API server | `API_SERVER_RAM_GB` | 32 GB | measured — the machine class benchmarked on 30 Aug 2026 |
| Cost of one add, in plain-search-equivalents | `ADD_COST_IN_PLAIN_SEARCH_EQUIVALENTS` | 1.0 | estimate — rounds the order up |
| **GPU cards** | | | |
| Embedding requests per second per card, low | `EMBED_CARD_REQUESTS_PER_S_LOW` | 300 | estimate — never benchmarked |
| Embedding requests per second per card, high | `EMBED_CARD_REQUESTS_PER_S_HIGH` | 500 | estimate — never benchmarked |
| Language-model calls per second per card | `AGENT_LLM_CALLS_PER_S_PER_CARD` | 15 | estimate — never benchmarked |
| GPU utilization ceiling | `GPU_UTILIZATION_CEILING` | 0.60 | assumption |
| Spare cards per GPU role | `GPU_SPARE_CARDS` | 1 | assumption |
| **Vector store** | | | |
| Retention, days | `--retention-days` | 90 | assumption — a placeholder; retention is undecided |
| Vector dimensions | `--dims` | 1,024 | assumption — a whole number, fixed before the first episode is ingested |
| Bytes stored per number | `--bytes-per-value` | 1 (int8 quantized) | assumption — fix before the first episode is ingested |
| RAM per vector-store machine | `--node-gb`, or the `node_gb` box on the web form | chosen automatically | assumption — the automatic choice buys the least total RAM |
| Machine sizes offered | `QDRANT_NODE_RAM_OPTIONS_GB` | 256, 512, 768 GB | assumption |
| Share of a machine's RAM that may be used | `QDRANT_NODE_FILL_LIMIT` | 0.70 | assumption |
| Index overhead on hot vector RAM | `QDRANT_INDEX_OVERHEAD_FACTOR` | 1.5 | assumption |
| Fill at which the report warns of a tight fit | `QDRANT_TIGHT_FIT_WARN_FRACTION` | 0.95 | assumption (display only — it changes no machine count) |
| **Sanity bounds — a refusal limit only; no machine count moves because of these** | | | |
| Largest design peak accepted | `MAX_OPS_PER_S` | 1,000,000,000 ops/s | assumption — far above the 1,000 ops/s scale tier |
| Longest retention accepted | `MAX_RETENTION_DAYS` | 36,500 days (100 years) | assumption |
| Most vector dimensions accepted | `MAX_VECTOR_DIMS` | 1,000,000 | assumption — models today are 384 to 4,096 |
| Most bytes per number accepted | `MAX_BYTES_PER_VALUE` | 64 | assumption — int8 is 1, float64 is 8 |
| Largest vector-store machine accepted | `MAX_NODE_GB` | 1,000,000 GB | assumption |
| **Disk** | | | |
| Bytes per number in the full-precision vector kept on disk | `ORIGINAL_VECTOR_BYTES_PER_VALUE` | 4 | estimate |
| Identifier and payload bytes per episode on disk | `QDRANT_DISK_PAYLOAD_BYTES_PER_EPISODE` | 256 | estimate |
| Segment and index overhead on vector-store disk | `QDRANT_DISK_OVERHEAD_FACTOR` | 1.3 | estimate |
| Episode text, low case | `EPISODE_TEXT_BYTES_LOW` | 800 bytes | estimate |
| Episode text, high case | `EPISODE_TEXT_BYTES_HIGH` | 2,400 bytes | estimate |
| PostgreSQL row overhead per episode | `POSTGRES_ROW_OVERHEAD_BYTES` | 400 bytes | estimate |
| PostgreSQL index bytes per episode | `POSTGRES_INDEX_BYTES_PER_EPISODE` | 300 bytes | estimate |
| PostgreSQL bloat between vacuums | `POSTGRES_BLOAT_FACTOR` | 1.4 | estimate |
| **PostgreSQL** | | | |
| Connection pool size per worker | `POSTGRES_POOL_SIZE` | 5 | measured 30 Aug 2026 |
| Connection overflow per worker | `POSTGRES_MAX_OVERFLOW` | 10 | measured 30 Aug 2026 |
| Gateway connections per API server | `GATEWAY_CONNECTIONS_PER_API_SERVER` | 20 | assumption |
| Chart default for `max_connections` | `POSTGRES_CHART_DEFAULT_MAX_CONNECTIONS` | 100 | measured — this default ran out of connections on 30 Aug 2026 |
| Largest `max_connections` ever proven to work | `POSTGRES_PROVEN_MAX_CONNECTIONS` | 600 | measured 30 Aug 2026 — cleared every error |
| PostgreSQL servers per deployment | `POSTGRES_SERVERS_PER_TIER` | 1 | assumption — never benchmarked at these statement rates |
| **Network message sizes** | | | |
| Add request | `NS_ADD_REQUEST_BYTES` | 1,200 bytes | estimate |
| Add reply | `NS_ADD_RESPONSE_BYTES` | 300 bytes | estimate |
| Search request | `NS_SEARCH_REQUEST_BYTES` | 600 bytes | estimate |
| Bytes per episode returned to the caller | `NS_RESPONSE_BYTES_PER_EPISODE` | 900 bytes | estimate |
| Episodes returned per plain search | `PLAIN_SEARCH_EPISODES_RETURNED` | 10 | measured configuration (`top_k` 10) |
| Episodes returned per agent-mode search | `AGENT_SEARCH_EPISODES_RETURNED` | 20 | estimate |
| Written answer in an agent-mode reply | `NS_AGENT_ANSWER_BYTES` | 2,000 bytes | estimate |
| Embedding request | `EMBED_REQUEST_BYTES` | 1,000 bytes | estimate |
| Embedding reply envelope | `EMBED_RESPONSE_ENVELOPE_BYTES` | 200 bytes | estimate |
| Vector-store query envelope | `QDRANT_SEARCH_REQUEST_ENVELOPE_BYTES` | 300 bytes | estimate |
| Candidates returned per vector search | `QDRANT_CANDIDATES_PER_SEARCH` | 50 | measured configuration (`vector_search_limit` 50) |
| Bytes per candidate | `QDRANT_BYTES_PER_CANDIDATE` | 200 bytes | estimate |
| Vector-store write envelope | `QDRANT_UPSERT_ENVELOPE_BYTES` | 500 bytes | estimate |
| Vector-store write reply | `QDRANT_UPSERT_RESPONSE_BYTES` | 200 bytes | estimate |
| PostgreSQL bytes per statement, both directions | `POSTGRES_BYTES_PER_STATEMENT` | 1,800 bytes | estimate |
| Language-model prompt | `LLM_CALL_REQUEST_BYTES` | 8,000 bytes | estimate |
| Language-model answer | `LLM_CALL_RESPONSE_BYTES` | 2,000 bytes | estimate |
| TLS, HTTP and TCP framing overhead | `NETWORK_PROTOCOL_OVERHEAD_FACTOR` | 1.2 | estimate |
| **Callers** — how fast a caller sends requests, which is a different question from what kind of request it sends | | | |
| Human chat sessions in a population | `--humans` (`users`), or the `humans` box on the web form | required for `users`; blank on the form | input |
| Automated clients in a population | `--automated` (`users`), or the `automated` box on the web form | 0 | input |
| Traffic mix of the human chat sessions | `--human-mix` (`users`), or the `human_mix` box on the web form | `45/45/10` | assumption — never measured |
| Traffic mix of the automated clients | `--automated-mix` (`users`), or the `automated_mix` box on the web form | `45/45/10` | assumption — never measured |
| Operations per second per human chat session, low | `HUMAN_SESSION_OPS_PER_S_LOW` | 0.011 | estimate — never measured |
| Operations per second per human chat session, high | `HUMAN_SESSION_OPS_PER_S_HIGH` | 0.028 | estimate — never measured |
| Operations per second per automated client | `AUTOMATED_CLIENT_OPS_PER_S` | 0.4 (a 5-second tool loop) | estimate — never measured |
| Operations per human prompt | `OPS_PER_HUMAN_PROMPT` | 2 | estimate — used only to describe the two rates above |
| **Output and serving** | | | |
| Where `validate` writes its JSON | `--out` (`validate`) | `sizing-numbers.json` in the current directory | input |
| Raw JSON instead of a table | `--json` (`tier`, `calc`) | off | input |
| Address the web server binds to | `--host` (`serve`) | `127.0.0.1` | input |
| Port the web server binds to | `--port` (`serve`) | 8000 | input |
| Seconds the web server waits on a silent connection | `SERVER_REQUEST_TIMEOUT_S` | 10 | assumption (serving only — it changes no machine count) |

Units: **GB** means 10<sup>9</sup> bytes, **TB** means 10<sup>12</sup> bytes and
**Mbps** means 10<sup>6</sup> bits per second, throughout.

## The measured anchor

One measurement carries most of the machine counts: **180 plain searches per
second per API server**.

It was measured on 30 August 2026 on a 16-vCPU AMD EPYC server (AWS c8a.4xlarge
class), with 8 worker processes and 128 concurrent requests, using OpenAI
`text-embedding-3-small` at about 180–190 ms per call, over a 12,000-episode
corpus, at `top_k` 10 and `expand` 0, with the `rrf-hybrid` reranker enabled and
with the vector store and PostgreSQL each on their own host.

If that anchor is wrong, every API server count moves in direct proportion:
halve it and the server count doubles.

## What the model assumes

**The traffic mix is an assumption nobody has measured.** The default split of
45 adds, 45 plain searches and 10 agent-mode searches per 100 operations is a
guess about how the service will be used. It is the second-largest lever in the
whole model, because one agent-mode search costs about 22 plain searches — so
the agent-mode share moves the hardware order more than any tuning does. Every
report prints a sensitivity table showing what happens to the API server count
as that share changes. Measure your own mix and pass it in with `--add`,
`--plain` and `--agent`. Remember what that share is: agent-mode search is a
request flag, not a kind of caller. If your traffic comes from two very
different kinds of caller, give each one its own mix on the `users` subcommand
with `--human-mix` and `--automated-mix`, and let it blend them for you.

**The reranker costs nothing extra, because its cost is already in the anchor.**
The 180/s figure was measured with `rrf-hybrid` switched on — reciprocal rank
fusion over BM25 and identity, running on the API server's own CPU at roughly
one core-millisecond per search. The model therefore adds nothing for it. Switch
to a cross-encoder reranker and this stops being true: that scores every
query-and-result pair on a GPU, so it adds cards to the order and invalidates
the anchor.

**The embedding-card rate has never been benchmarked.** The 300–500 embedding
requests per second per H100-class card is the single largest unmeasured number
in the model, and the embedding GPU count is wrong in direct proportion if it is
wrong. Benchmark your own card with your own model before buying any GPU.

**One add is charged as one plain search of API work.** The benchmark measured
search only, so the cost of an add relative to a search is unknown. Charging
them equally rounds the order up, not down.

**Nothing here demonstrates five minutes of sustained service.** Every benchmark
run started from a freshly restarted server, so these are clean-start numbers.
A design peak is the worst rate the system must hold for five minutes; treat the
anchor accordingly.

## Tests

```bash
cd tools/sizing
uv run --no-project python -m unittest -v
```

The tests are written from the model rather than from the code: nearly every
expected number is worked out by hand and written as a literal, so a failure
means the program and the model disagree.

A few tests are different. They check that the command line and the web
endpoint return the same numbers as the library function both of them call.
Those catch a broken front door, not a broken model, and each one sits beside a
hand-worked literal test of the same figures.

The suite also checks that the calculator imports nothing outside the standard
library, that every bad input exits with a message rather than a traceback,
that the web server answers correctly, that every row of every labelled report
table carries a label, and that every named constant in the table above reaches
the file `validate` writes.

The repository's linter must also be clean:

```bash
uv run --no-project --with ruff ruff check tools/sizing
```

Name the path: `tools` is in the `exclude` list in the repository's
`pyproject.toml`, so `ruff check .` skips this directory entirely.

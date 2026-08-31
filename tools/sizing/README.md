# MemMachine deployment sizing calculator

Work out what hardware a MemMachine deployment needs to serve a given amount of
traffic.

You give it a **design peak** — the worst rate the system must sustain for five
minutes, in operations per second — and a traffic mix, and it tells you how many
API servers, vector-store machines and PostgreSQL servers to order, how many
embedding and agent-model GPU cards, how much RAM and disk the stored episodes
will take, how many PostgreSQL connections the deployment will open, and how
much network bandwidth it will use.

If you do not know your design peak, the [`users`](#users--a-caller-population-to-the-capacity-it-needs)
subcommand works it out from a population of callers. Start from concurrent
sessions if you know them, or from a user count if that is what you have — a
user count needs two more figures, and each has an **example default** so a
bare user count always gets an answer. See
[The four conversion figures are example defaults](#the-four-conversion-figures-are-example-defaults).

**The counting rule.** The unit this model counts is a session sending requests
at the same moment, whoever is behind it. One developer driving a ten-user load
test is ten sessions, not one. If you already know how many sessions will be in
flight at once, give that number directly with `--humans` and `--automated` and
skip the conversion. A user count and a share of users active at the busiest
moment exist only for estimating that number from a population of people.

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
multiplier: it says how fast a caller sends requests. Both rates now cite a
published measurement — see
[Where the two per-caller rates come from](#where-the-two-per-caller-rates-come-from).

The two are independent. A person can send agent-mode searches, and an automated
client can send nothing but plain searches. That is why each population carries
its own traffic mix: see [`users`](#users--a-caller-population-to-the-capacity-it-needs).

Every number in the report's tables of findings carries one of four labels, so
you can always see how much weight it will bear:

| Label | Meaning |
| --- | --- |
| `measured` | It came out of a real benchmark run, and the label names that run: its date, the configuration it ran with, or both. |
| `derived` | The program computed it from measured numbers and from the assumptions below. |
| `estimate` | Not measured on this system. Usually nobody has measured it anywhere; where a published measurement of another system stands behind it, the label names that source. Benchmark it before ordering hardware against it. |
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
of caller: **concurrent human chat sessions**, people typing, at 0.011 to 0.028
operations per second each; and **concurrent automated client sessions**,
programs sending requests in a loop, at 0.4 operations per second each.
`--automated` counts callers. It has nothing to do with `--agent`, which is the
agent-mode share of the requests themselves.

The report also prints two extra rows that no machine count uses: a **headroom
check** at 0.06 operations per second, what the busiest 1 human session in 100
demands, and an **average-load check** at 0.07 operations per second per
automated client, what one demands over a whole session including the time it
sits idle. Both are labelled and both are there to be read, not multiplied.

Both counts are **concurrent**: sessions running at the same moment, not
registered accounts and not visitors in a day. Each count is multiplied by a
rate per session, so typing a total user count where a concurrent count belongs
gives an answer that is wrong by a factor of hundreds. The web form labels both
boxes "Concurrent ...", in the same words as these two flags.

**The counting rule, again, because it decides the answer.** The unit this
model counts is a session sending requests at the same moment, whoever is
behind it. One developer driving a ten-user load test is ten sessions, not one
— the person is irrelevant, the ten in-flight sessions are the load. If you
already know how many sessions will be in flight at once, give that number
directly with `--humans` and `--automated` and skip the conversion below. The
user count and the share-active figure exist only for estimating that number
from a population of people.

#### Starting from a user count instead

Most people who ask for a deployment do not know their concurrent sessions. They
know their users: "we have 50,000 users." Give that instead, with the two figures
that turn a user count into a session count:

```bash
uv run --no-project python memmachine_sizing.py users \
  --human-users 50000 --human-peak-share 2 --human-sessions-per-active-user 1 \
  --automated-users 200 --automated-peak-share 25 \
  --automated-sessions-per-active-user 20
```

`--human-peak-share` is how many of every 100 of those people are using the
service at the busiest moment. `--human-sessions-per-active-user` is how many
sessions one of those active people holds at that moment. The same two figures
exist for automated clients, as `--automated-peak-share` and
`--automated-sessions-per-active-user`, and `--automated-users` counts the people
who run automated clients rather than the sessions those clients open.

All four have an **example default**, so a user count on its own answers:

```bash
uv run --no-project python memmachine_sizing.py users --human-users 50000
```

That is 50,000 × 10 per 100 × 1 session = **5,000 concurrent human chat
sessions**, and the report prints a note saying which figures it chose for you
and what to replace them with. The defaults and their basis are in
[The four conversion figures are example defaults](#the-four-conversion-figures-are-example-defaults).

The three multiply:

```
concurrent sessions = users x share active / 100 x sessions per active user
```

so the example above is 50,000 x 2 / 100 x 1 = 1,000 concurrent human chat
sessions and 200 x 25 / 100 x 20 = 1,000 concurrent automated client sessions.
Two very different user counts, arriving at the same number of sessions. The
report prints that multiplication as its own table, headed "From users to
concurrent sessions". Its label column marks **every** figure as either
*supplied* by you or an *example default* this program chose, so the two can
never be mistaken for each other.

Give `--humans` or `--human-users`, never both: they answer the same question two
different ways, and giving both is refused rather than one of them quietly
winning. The same goes for `--automated` and `--automated-users`.

#### The four conversion figures are example defaults

Each of the four has a default, so a bare user count always gets an answer. Every
one of them is an **example**. It is labelled *example default* in the report, a
note under the table says which ones were used, and each is meant to be replaced.

**`--human-peak-share`, 10 per 100 users. A convention, not a measurement.**
Microsoft's SharePoint capacity guidance states: *"A concurrency rate of 10
percent is assumed, with 1 percent of concurrent users making requests at a given
moment. For example, for 10,000 users, 1,000 users are actively using the
solution simultaneously"*
(<https://learn.microsoft.com/en-us/previous-versions/office/sharepoint-2007-products-and-technologies/cc263100(v=office.12)>).
Read the words: *is assumed*. Teletraffic engineering offers a comparable figure
as an explicit rule of thumb — 10 to 16% of telephone subscribers busy during the
busy hour (Iversen, *Teletraffic Engineering and Network Planning*, DTU).

These are two conventions, from two different fields, that happen to agree.
Agreement between conventions is weaker evidence than one measurement, and
neither is called a measurement here. The plausible range is **5 to 20 per 100
users**, so the figure can be wrong by a factor of two either way, and every
machine count moves with it.

**`--human-sessions-per-active-user`, 1. No published evidence exists.** That is
the whole basis, and it is printed in those words. One session per active person
is the simplest thing that can be true.

**`--automated-peak-share`, 100 per 100, and
`--automated-sessions-per-active-user`, 1. Identity.** For a program the natural
unit is the running client, and a client that is running *is* a session, so there
is nothing to convert. The 10-per-100 convention above describes people, most of
whom are asleep or busy at any one moment. Applying it to load generators or to
deployed clients would divide the load by ten.

**The warning that used to be a refusal.** A user count without both figures used
to be rejected with exit code 2. It now answers, and everything the refusal said
is printed as a note with the conversion table instead:

```
$ uv run --no-project python memmachine_sizing.py users --human-users 50000
  ... EXAMPLE DEFAULTS WERE USED HERE. The share of people active at the
  busiest moment and the sessions one active person holds are not figures you
  gave: this calculator chose them so that a user count always gets an answer,
  and they are marked "example default" in the table above. A user count is not
  a count of concurrent sessions, and this calculator cannot know the difference
  between them for your service. The share of users active at the busiest moment
  is the figure that hurts: 10 per 100 users is a convention rather than a
  measurement, the plausible range is 5 to 20 per 100, so it can be wrong by a
  factor of two either way - and every machine count below moves with it.
  Replace these examples with your own numbers: meter the concurrent sessions
  the deployment actually holds - callers sending requests at the same moment -
  from the first day of running, and give those session counts instead of a user
  count.
```

Supply your own figures and that note is not printed at all: you are not warned
about numbers you did not use.

**Why the share matters so much.** A user count and a session count are different
numbers, and the gap between them is large. Only a small share of a user base is
using a service at any one moment — for most services a few per 100 — so 50,000
users might be 500 concurrent sessions or 5,000. That is a tenfold spread, and it
moves the answer across two whole tiers. The second figure moves it again: a
person holds about one session, maybe two across devices, but one person running
an automated client framework can hold ten or fifty open at once, so a count of
automated sessions can bear almost no relation to a count of people.

The cost of guessing, in machines: 50,000 typed into `--humans` is 50,000
concurrent sessions, which is 1,400 operations per second and **41 API servers**,
above every named tier. The same 50,000 users at 2 per 100 active holding one
session each is 1,000 concurrent sessions, which is 28 operations per second and
**1 API server**, inside the target tier.

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

Below the sizing boxes are four more for a caller population: "Concurrent human
chat sessions", "Concurrent automated client sessions" and a traffic mix for
each. Both counts are sessions running at the same moment, not accounts and not
visitors in a day. "Concurrent automated client sessions" counts callers that
are programs; the "Agent-mode searches per 100 operations" box above it is the
share of the requests themselves.

Below those are six more, for a reader who knows a user count rather than a
session count: "People in the user base", "Share of people active at the busiest
moment, per 100 users" and "Sessions per active person", and the same three for
automated clients. They are the boxes for `--human-users`, `--human-peak-share`
and `--human-sessions-per-active-user` and their automated twins, and they follow
the same rules as the flags. The two shares and the two sessions-per-active-user
boxes each have an example default: leave one blank beside a filled-in user count
and the example is used, marked as an example in the table and warned about in a
note under it. A user count typed on its own answers. Filling in a user count and
its count box for the same kind of caller is refused, because the two answer the
same question different ways. When a user count is used, the page adds the "From
users to concurrent sessions" table above the others and says under the button
that the session counts were worked out rather than typed in.

**Which boxes size the deployment.** The form asks about the same traffic in two
ways — a design peak with a traffic mix, and a caller population — and only one
of them can size the machines:

| What you filled in | What sizes the machines |
| --- | --- |
| Both count boxes empty | The design peak box and the three mix boxes, exactly as before the form could take a population. No population tables are drawn. |
| Either count box filled | The population. The design peak becomes the demand it makes at the busy end of the human rate, and the traffic mix becomes its blended mix. The design peak box and the three mix boxes are not read at all. |
| One count box filled, the other empty | The same, with the empty box counting as none of that kind of caller — which is what `--automated` defaulting to 0 does on the command line. |
| Both counts filled in as zero | Nobody sends anything, so there is no deployment to size from the population. The design peak box sizes it after all, and the population tables report the zero demand. |
| A user count box filled, with its share and its sessions per active user | The population again, with the concurrent sessions worked out from those three numbers first. The conversion table shows the multiplication. |
| A user count box filled with either figure missing | The population again. The missing figure takes its example default, the table marks it *example default*, and a note under the table says what to replace it with. |

The page says which of the two it used, in a sentence under the button headed
**What sized these machines**, and it quotes back the design peak you typed when
that number was superseded — so you never have to work out from the tables
whether the box you filled in mattered. The JSON carries the same sentence in
`sized_from_note`, and `inputs.sized_from` says `the caller population` when the
population is what sized the run.

This is why the page and the command line now agree. `users --humans 2500
--automated 75 --human-mix 90/9/1 --automated-mix 10/20/70` and
`/api/calc?humans=2500&automated=75&human_mix=90/9/1&automated_mix=10/20/70`
both order 6 API servers, 1 PostgreSQL server, 5 vector-store machines of 256 GB,
3 to 5 embedding GPU cards and 4 agent-model GPU cards. Until this was fixed the
page worked out the blended mix, printed it, and then sized the machines from the
mix boxes anyway, so the same population bought 6 API servers on the command line
and 3 on the page.

Fill in either count and the page also adds the tables the `users` subcommand
prints — the demand from each population, the blended mix and the machines that
mix needs — above the sizing report.

The four boxes that describe the store rather than the traffic — retention,
vector dimensions, bytes per number and the RAM of a vector-store machine — are
read either way. A population says how much traffic there is; it says nothing
about how long an episode is kept or how wide a vector is.

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

Every sizing box the run actually uses has to be answered. A form submission
always sends every box, so an empty one means the reader cleared it, and the
page says which box is blank rather than quietly sizing for the default. A box
the run does not read is not asked for: when a population sizes the deployment,
the design peak and mix boxes may be blank and the answer still comes back. A parameter left
out of the URL altogether is different, and still takes the default — which is
why `/api/calc?ops=100` works. The "RAM per vector-store machine" box is the one
place where empty is itself an answer, and the two population count boxes are
the other: empty there means no callers of that kind. The six user boxes are
empty until you use them, and then they have to be filled in as a set: a user
count needs its share active and its sessions per active user, and those two
need a user count to convert.

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
traffic mix. The population then sizes the answer — `inputs.ops_per_s` is the
demand it makes and `inputs.mix` is its blended mix — and a `population` block
carries the working alongside the sizing. Any `ops`, `add`, `plain` or `agent`
in the same address is not read; `sized_from_note` says so in words:

```
http://127.0.0.1:8899/api/calc?ops=100&humans=5000&automated=40&human_mix=48/50/2&automated_mix=20/20/60
```

To start from a user count instead of a session count, use
`human_users`, `human_peak_share` and `human_sessions_per_active_user`, and the
same three with `automated_` in front. Each of those four may be left out and
takes its example default. The population block then carries a `conversion`
field with the user count, the two figures, a flag beside each saying whether it
was supplied or defaulted, and the concurrent sessions they multiply to:

```
http://127.0.0.1:8899/api/calc?human_users=50000&human_peak_share=2&human_sessions_per_active_user=1
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
| Largest user count accepted | `MAX_USERS` | 1,000,000,000,000 | assumption — more users than there are people |
| Largest share active at the busiest moment | `MAX_PEAK_ACTIVE_PER_100` | 100 per 100 users | definition — all of a user base is 100 per 100 |
| Most sessions per active user accepted | `MAX_SESSIONS_PER_ACTIVE_USER` | 10,000 | assumption |
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
| Concurrent human chat sessions in a population — sessions at one moment, not accounts and not visitors a day | `--humans` (`users`), or the `humans` box on the web form | `users` needs this or `--human-users`; blank on the form | input |
| Concurrent automated client sessions in a population — sessions at one moment, not programs installed | `--automated` (`users`), or the `automated` box on the web form | 0 | input |
| **Users to concurrent sessions** — a user count is not a session count, and these turn one into the other | | | |
| People in the user base — a user count, not a session count | `--human-users` (`users`), or the `human_users` box on the web form | none; give this or `--humans` | input |
| Share of those people using the service at the busiest moment, per 100 users | `--human-peak-share` (`users`), or the `human_peak_share` box on the web form | 10 per 100 users | **example default** — a convention, not a measurement: Microsoft SharePoint capacity guidance assumes 10% concurrency, teletraffic engineering uses 10–16% of subscribers busy in the busy hour. Plausible range 5 to 20 |
| Sessions one of those active people holds at that moment | `--human-sessions-per-active-user` (`users`), or the `human_sessions_per_active_user` box on the web form | 1 | **example default** — no published evidence exists |
| Users who run automated clients — a user count, not a session count | `--automated-users` (`users`), or the `automated_users` box on the web form | none; give this or `--automated` | input |
| Share of those users running automated clients at the busiest moment, per 100 users | `--automated-peak-share` (`users`), or the `automated_peak_share` box on the web form | 100 per 100 users | **example default** — identity: a client that is running is already a session |
| Automated client sessions one of them holds at that moment | `--automated-sessions-per-active-user` (`users`), or the `automated_sessions_per_active_user` box on the web form | 1 | **example default** — identity: a client that is running is already a session |
| Plausible range of the share of people active at the busiest moment — **reported only, no machine count uses it** | `HUMAN_PEAK_ACTIVE_PER_100_LOW`, `HUMAN_PEAK_ACTIVE_PER_100_HIGH` | 5 to 20 per 100 users | estimate — the spread across the two conventions above |
| Traffic mix of the human chat sessions | `--human-mix` (`users`), or the `human_mix` box on the web form | `45/45/10` | assumption — never measured |
| Traffic mix of the automated client sessions | `--automated-mix` (`users`), or the `automated_mix` box on the web form | `45/45/10` | assumption — never measured |
| Operations per second per human chat session, low | `HUMAN_SESSION_OPS_PER_S_LOW` | 0.011 | estimate — about the median of a session's busiest five minutes in BurstGPT, "BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems" (KDD '25, 2025), <https://arxiv.org/abs/2401.17644> |
| Operations per second per human chat session, high | `HUMAN_SESSION_OPS_PER_S_HIGH` | 0.028 | estimate — about the 90th percentile of a session's busiest five minutes in the same BurstGPT measurement, <https://arxiv.org/abs/2401.17644> |
| Operations per second per a heavy human chat session — **reported only, no machine count uses it** | `HUMAN_SESSION_OPS_PER_S_HEAVY` | 0.06 | estimate — the 99th percentile of the same BurstGPT measurement, <https://arxiv.org/abs/2401.17644> |
| Operations per second per automated client, the design peak | `AUTOMATED_CLIENT_OPS_PER_S` | 0.4 (a 5-second tool loop) | estimate — TraceLab's measured 5.0-second median step (2026), <https://arxiv.org/abs/2606.30560> |
| Operations per second per automated client, sustained — **reported only, no machine count uses it** | `AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED` | 0.07 | estimate — TraceLab's measured 28.3-second mean step (2026), <https://arxiv.org/abs/2606.30560> |
| Operations per human prompt | `OPS_PER_HUMAN_PROMPT` | 2 | estimate — used only to describe the human rates above |
| Peak load against average load, one organisation — **reported only, no machine count uses it** | `PEAK_TO_AVERAGE_SINGLE_ORGANISATION` | 4x | estimate — measured traces: 4.64x for a service of a few hundred users, <https://arxiv.org/abs/1207.6295> |
| Peak against average, a large consumer service — **reported only** | `PEAK_TO_AVERAGE_LARGE_SERVICE_LOW`, `PEAK_TO_AVERAGE_LARGE_SERVICE_HIGH` | 1.3x to 1.64x | measured elsewhere — a Google production cell of 12,500 machines, <https://www.cs.virginia.edu/~cr4bd/papers/socc12.pdf>, and Hotmail |
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

### The five minutes, and getting a peak from an average

**Why five minutes.** A design peak is the worst rate the system must sustain for
five minutes, not an average. Five minutes is not this program's invention.
ITU-T E.500, the international recommendation for measuring telephone traffic,
independently requires measurement windows *"greater than 5 minutes... so that
resources are not dimensioned for infrequent small interval peak traffic levels"*
(<https://www.itu.int/rec/T-REC-E.500-199811-I/en>). A shorter window sizes the
order for a spike that lasts seconds; a longer one hides the spike inside an
average.

**If what you know is your average load, multiply it by 4.** That gives a design
peak for a deployment inside one organisation. The multiplier is **reported
only**: no machine count in this program reads it, and nothing here multiplies by
it for you — you do the multiplication and pass the result in as the design peak.

The evidence is measured, and it disagrees by scale:

| Service | Peak against its own mean | Source |
| --- | --- | --- |
| A service for a few hundred users | **4.64x** | Wang et al. (2012), <https://arxiv.org/abs/1207.6295> |
| Hotmail | 1.64x | the same paper |
| A Google production cell, 12,500 machines | 1.3x | Reiss et al., SoCC '12, <https://www.cs.virginia.edu/~cr4bd/papers/socc12.pdf> |

A small population is burstier because scale flattens the curve: one team going
to lunch together is visible in a few hundred users and invisible in a few
million. A MemMachine deployment inside one organisation is the small case, which
is why 4 is the figure reported and not 1.3.

### Where the two per-caller rates come from

Both per-caller rates were guesses until now. Neither number has changed, but
each one now has a published measurement standing behind it, named here so you
can check it without asking anybody. Both are still labelled `estimate`,
because neither was measured on a MemMachine deployment.

**A human chat session: 0.011 to 0.028 operations per second.** The source is
**BurstGPT**, a public workload dataset of 110 consecutive days of one real
Azure OpenAI deployment, released with the KDD '25 paper *BurstGPT: A
Real-world Workload Dataset to Optimize LLM Serving Systems* (2025,
<https://arxiv.org/abs/2401.17644>, dataset
<https://github.com/HPMLL/BurstGPT>). Version 2 of that release added a session
identifier, which is what makes a per-session rate recoverable at all. Across
55,295 conversation sessions and 176,466 gaps between one prompt and the next,
the median gap was 131 seconds and a session held a mean of 4.19 prompts. Over
a session's **busiest five minutes** the rate was a median of 0.0067 prompts
per second, a 90th percentile of 0.0167 and a 99th percentile of 0.030 — at two
operations per prompt, 0.013, 0.033 and 0.060 operations per second. So 0.011
is about the median of a session's busiest five minutes and 0.028 is about its
90th percentile: 9 sessions in 10 are slower than that.

What BurstGPT is not: it is one regional deployment, and who its users were is
not published. It is a guide, not your traffic.

**A heavy human chat session: 0.06 operations per second.** The 99th percentile
of the same measurement — the busiest 1 session in 100, about twice the top of
the band. It is **reported only**. It appears as a headroom line in the report
and no machine count uses it, because sizing every session at the busiest one
in a hundred would buy hardware for a population that does not exist.

**An automated client: 0.4 operations per second as a design peak, 0.07
sustained.** The source is **TraceLab** (2026,
<https://arxiv.org/abs/2606.30560>), which instrumented about 4,300 real Claude
Code and Codex coding-agent sessions: roughly 350,000 model steps and 430,000
tool calls from 43 developers over about 8 months. A step is one model
generation plus the tool call it asks for.

| What TraceLab measured | Time per step | At two operations per step |
| --- | --- | --- |
| Median step | 4.9 s generating + 0.1 s executing the tool = **5.0 s** | **0.40 operations/s** — the design peak this model sizes on |
| Mean step | 11.5 s generating + 16.8 s executing the tool = **28.3 s** | **0.07 operations/s** — sustained, reported only |

The median step of 5.0 seconds is exactly the "five-second tool loop" this
program has always described, and 0.40 is the constant it has always used.

**The two automated figures differ by about six times, and that is not an
error.** An agent is idle most of the wall-clock time, waiting on the person:
TraceLab measured human thinking at 92.3% of session wall-clock time. Size from
the sustained figure and a burst is under-provisioned; size from the design peak
and a day is over-provisioned. This model sizes for the worst five minutes it
must sustain, and an agent that is actively working runs at the median step
pace, so it uses 0.4. The 0.07 figure is printed beside it for average-load
planning and enters no machine count. Anthropic's own production Claude Code
data — about 10 actions per prompt, median turn about 45 seconds (2026,
<https://www.anthropic.com/research/claude-code-expertise>) — implies roughly
0.22 actions per second within a turn, between the two.

What TraceLab is not: 43 developers using coding agents. It is not a
measurement of automated clients in general.

**Meter your own traffic anyway.** Record real operations per second per API
key from the first day of the pilot and re-check the tier choice against it.

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

#!/usr/bin/env python3
"""MemMachine deployment sizing calculator.

Work out what a MemMachine deployment needs to serve a given design peak: how
many API servers, vector-store machines and PostgreSQL servers, how many
embedding and agent-model GPU cards, how much storage, how many PostgreSQL
connections and how much network bandwidth. Run ``validate`` to print the
figures this program publishes for all three tiers, together with every named
constant the model is built from, and to write them to a JSON file, so any
figures quoted elsewhere can be checked against this program mechanically.

Two different things in this program are easy to confuse, and a reader who
confuses them will size the deployment wrongly.

  Agent-mode search is a property of a REQUEST. It is the agent_mode flag on a
  MemMachine search. A search sent with that flag on fans out into a multi-hop
  retrieval that costs about 22 plain searches. It is set per request, not per
  caller. In this program it is the third share of the traffic mix, and the
  flag that sets it is --agent.

  An automated client is a property of a CALLER. It is a program that sends
  requests in a loop rather than a person typing in a chat window. It is
  assumed to send about 0.4 operations per second, where a human chat session
  sends 0.011 to 0.028. In this program it is a population count on the users
  subcommand, and the flag that sets it is --automated. Both rates now cite a
  published measurement; see the estimates section below.

  One says how expensive a request is. The other says how fast a caller sends
  requests. A caller of either kind can send requests of either kind, which is
  why each population carries its own traffic mix.

THE COUNTING RULE. The unit this model counts is a session sending requests at
the same moment, whoever is behind it. One developer driving a ten-user load
test is ten sessions, not one. If you already know how many sessions will be in
flight at once, give that number directly as a count of concurrent sessions -
--humans and --automated on the command line - and skip the conversion below. A
user count and a share of users active at the busiest moment exist only for
estimating that number from a population of people.

Everything the model uses is listed below. A reader should be able to audit the
whole model from this docstring alone, without reading the code.

Four labels are used throughout. Every number in the report's tables of
findings carries one, and so does every constant listed below:

  measured    - it came out of a real test, and the label names that test:
                its date, the configuration it ran with, or both.
  derived     - this program computed it from measured numbers and from the
                assumptions listed here.
  estimate    - not measured on this system. Usually nobody has measured it
                anywhere; where a published measurement of another system
                stands behind it, the label names that source. Benchmark it
                before ordering hardware against it.
  assumption  - a planning choice, not a finding.

Two tables in the report are the exception, because their rows are what-ifs
rather than findings: the Qdrant node choice table and the sensitivity table.
The note above each says where its numbers come from. The result dictionary
that size_deployment returns, and the JSON built from it, carry the numbers
without labels.

--------------------------------------------------------------------------------
MEASURED
--------------------------------------------------------------------------------

API server throughput: 180 plain searches per second per server.
  Where it came from: a benchmark run of 30 August 2026. Every serving host was
  a 16-vCPU AMD EPYC server (AWS c8a.4xlarge class): 16 vCPU (virtual CPU cores),
  32 GiB of RAM, AMD EPYC Turin. The core API measured 178.66 searches per second
  and the full platform measured 180.31, both at 8 uvicorn worker processes and
  128 concurrent client requests. Other settings that produced the number: the
  real OpenAI embedding endpoint (text-embedding-3-small, about 180-190 ms per
  call), a corpus of 12,000 stored episodes, top_k 10, expand 0 (which makes the
  internal vector_search_limit 50), the rrf-hybrid reranker enabled, and Qdrant
  and PostgreSQL each on their own separate host. The platform figure was driven
  from four client load-generator processes; a single load-generator process
  measures itself, not the server, at that concurrency.
  If it is wrong: every API server count below moves in direct proportion.
  Halve the anchor and the server count doubles.

Eight workers per 16-vCPU server is the knee.
  Where it came from: the same run. Going from 8 to 16 worker processes bought
  only 3-5% more throughput, and nothing at all on the steady-state rate. This
  program assumes 8 workers per server, which is what sets the PostgreSQL
  connection arithmetic below.
  If it is wrong: the connection total changes, not the server count.

PostgreSQL connection exhaustion, 30 August 2026.
  Each worker process opens up to 15 connections (SQLAlchemy pool size 5 plus
  max_overflow 10). At 8 workers that is 120 connections against PostgreSQL's
  default max_connections of 100. The core filled the connection table, the
  gateway could then not get its own connection to check API keys, and it
  returned HTTP 401 Unauthorized on valid keys. Raising max_connections to 600
  cleared every error in all 36 test runs. 600 is therefore the largest
  connection limit this deployment has ever been proven to work at.
  If it is wrong: nothing in the machine counts moves; the max_connections
  recommendation does.

The reranker cost is already inside the 180/s anchor.
  The reranker used in the measured run is rrf-hybrid: reciprocal rank fusion
  over BM25 and identity. It runs in the API server's own process on its own CPU,
  at roughly one core-millisecond per search, and needs no GPU. Because the
  anchor was measured with it switched on, its cost is already paid for in the
  180/s figure and this program adds nothing for it. Two warnings. First, the
  library-level benchmark of 30 August ran with no reranker at all, which is one
  of several reasons the library and API numbers are not comparable. Second, a
  cross-encoder reranker - the option to reach for when retrieval quality
  matters more than throughput - is a completely different machine: it scores
  every query-and-result pair on a GPU, so it would add GPU cards to the order
  and it would invalidate the 180/s anchor entirely.

--------------------------------------------------------------------------------
DERIVED
--------------------------------------------------------------------------------

Fan-out per request. These counts were read from the MemMachine source code on
30 August 2026. They are not guesses, but they are also not timings - they are
how many internal calls one API request makes.

  add             1 embedding call,  1 vector write,    2 PostgreSQL statements,
                  0 language-model calls
  plain search    2 embedding calls (1 once every request sends
                  types: ["episodic"]), 1 vector search, 2 PostgreSQL statements,
                  0 language-model calls
  agent search   22 embedding calls, 22 vector searches, 44 PostgreSQL
                  statements, 1 to 2 language-model calls. "Agent search" here
                  means agent-mode search: a request flag, not a kind of
                  caller.

  If these are wrong: every demand figure in the program is wrong by the same
  factor, and the machine counts follow.

API server count. Work is measured in plain-search-equivalents per second:
  work = vector searches/s + adds/s. One add is counted as one
  plain-search-equivalent. That is an ESTIMATE that deliberately rounds up: the
  30 August run measured search only and never measured an add, so the cost of an
  add against the cost of a search is not known. Servers are then filled to at
  most 60% of the measured anchor, which leaves headroom for spikes:
  180 x 0.60 = 108 usable searches/s per server, and
  servers = ceil(work / 108), always rounding up to a whole machine.

Storage. Episodes stored = adds/s x 86,400 seconds x retention days.
  Hot vector RAM in Qdrant = episodes x dimensions x bytes per value x 1.5, where
  the 1.5 is index overhead. Throughout this program GB means 10^9 bytes, not
  2^30 bytes - decimal gigabytes, the unit hardware is sold in.

One year with nothing ever deleted. The report also publishes hot vector RAM,
  Qdrant NVMe and PostgreSQL disk for a full year of adds with no deletion at
  all: episodes in a year = adds/s x 86,400 x 365, multiplied by the same
  per-episode byte sizes as the retained figures. These three numbers are a year
  of adds and nothing else, so they do NOT move with the retention setting - at
  retention 0 they are still a full year. They exist because they are the
  numbers that make retention a requirement rather than an option.

PostgreSQL connections = API servers x 8 workers x 15 connections per worker,
  plus a gateway allowance of 20 connections per API server. That total is the
  max_connections the tier needs. The program prints it next to PostgreSQL's
  chart default of 100 and next to the 600 that cleared every error on
  30 August, and it says plainly when the tier needs more connections than have
  ever been proven to work.

--------------------------------------------------------------------------------
ESTIMATE - not measured on this system
--------------------------------------------------------------------------------

Nothing below has been measured on a MemMachine deployment. Most of it has
never been measured anywhere, and each entry says which case it is. The two
per-caller rates are the exception: they now carry published measurements of
other systems, named in full with the paper, the year and the web address, so
a reader can check the source without asking anybody. Measured elsewhere is
still an estimate here.

Embedding GPU card rate: 300 to 500 embedding requests per second per H100-class
  card. Never benchmarked, on any card, with the planned model. Filled to 60%,
  that is 180 to 300 usable requests per second per card. Cards needed =
  ceil(demand / usable) + 1 spare card. The program sizes on the embedding
  demand WITHOUT the types: ["episodic"] fix, because that is the larger and
  therefore the safer of the two figures.
  If it is wrong: the embedding GPU order is wrong in direct proportion. This is
  the single largest unmeasured number in this model and it must be benchmarked
  before any GPU is bought.

Agent-model GPU: one 8B-class card serves 15 language-model calls per second.
  This comes from an assumed range of 10 to 20 calls per second at the target
  tier on a single card; 15 is the planning figure. One spare card is added.
  If it is wrong: the agent-model card count moves in direct proportion.

Language-model calls per agent-mode search: between 1 and 2. The program sizes
  on 1.5, the midpoint, and reports the 1-to-2 range alongside it.

One add costs at most one plain search of API work. Never measured; see the API
  server count note above. It rounds the order up, not down.

Per-call message sizes, used only for the network figures. Every one of these is
  an estimate, declared as a named constant in the code so that the network
  numbers can be reproduced by hand:
    episode text about 800 bytes (a low case) to 2,400 bytes (a high case);
    add request 1,200 bytes and its reply 300 bytes;
    search request 600 bytes; 900 bytes per episode returned to the caller;
    10 episodes returned per plain search (top_k 10, the measured configuration);
    20 episodes plus a 2,000-byte written answer returned per agent-mode search;
    embedding request 1,000 bytes, its reply the vector at 4 bytes per number
      plus 200 bytes of envelope;
    vector-store query the query vector at 4 bytes per number plus 300 bytes,
      its reply 50 candidates (vector_search_limit 50, the measured
      configuration) at 200 bytes each;
    vector-store write the vector at 4 bytes per number plus 500 bytes, reply
      200 bytes;
    PostgreSQL 1,800 bytes per statement counting both directions;
    one language-model call 8,000 bytes of prompt and 2,000 bytes of answer;
    and a flat 1.2x multiplier for TLS, HTTP and TCP framing overhead.
  The east-west total is built on the embedding demand WITHOUT the
  types: ["episodic"] fix and on the 1.5-call planning figure for language-model
  calls, which is the same pair of choices the embedding GPU count and the
  agent-model GPU count are sized on, so the two sizing paths agree.
  If these are wrong: only the network section moves. The conclusion that
  network is not a constraint has a very large margin, so these would have to be
  wrong by more than a factor of ten to change the answer. They are named here
  because a bandwidth figure that cannot be reproduced from its own inputs
  cannot be checked by anybody.

Disk sizes, also estimates and also declared as named constants:
    Qdrant NVMe per episode = dimensions x 4 bytes (the full-precision original
      vector, which Qdrant keeps on disk even when the searchable copy in RAM is
      quantized) plus 256 bytes of identifier and payload, all multiplied by 1.3
      for segment and index overhead.
    PostgreSQL per episode = episode text (800 bytes low, 2,400 bytes high) plus
      400 bytes of row overhead plus 300 bytes of index, multiplied by 1.4 for
      table bloat between vacuums. That gives a low-to-high range.

Callers to capacity. There are two kinds of caller, and four rates between
  them. Two of the four size the deployment; the other two are reported and
  nothing else.

  A human chat session is a person typing in a chat window, estimated at 0.011
  to 0.028 operations per second. At roughly two operations per prompt that is
  about 20 prompts an hour at the low end (40 operations an hour, 0.011 ops/s)
  and about 50 prompts an hour at the high end (about 101 operations an hour,
  0.028 ops/s). The band is backed by BurstGPT, a public workload dataset of
  110 consecutive days of one real Azure OpenAI deployment, released with the
  KDD '25 paper "BurstGPT: A Real-world Workload Dataset to Optimize LLM
  Serving Systems" (2025, https://arxiv.org/abs/2401.17644, dataset
  https://github.com/HPMLL/BurstGPT). Version 2 of that release added a
  session identifier, which is what makes a per-session rate recoverable.
  Across 55,295 conversation sessions and 176,466 gaps between one prompt and
  the next, the median gap was 131 seconds and a session held a mean of 4.19
  prompts. Over a session's BUSIEST FIVE MINUTES the rate was a median of
  0.0067 prompts per second, a 90th percentile of 0.0167 and a 99th percentile
  of 0.030; at two operations per prompt, 0.013, 0.033 and 0.060 operations
  per second. So 0.011 is about the median of a session's busiest five minutes
  and 0.028 is about its 90th percentile. It is still an ESTIMATE for this
  deployment: BurstGPT is one regional deployment, and who its users were is
  not published.

  Heavy human chat session: 0.06 operations per second, the 99th percentile of
  the same measurement - the busiest 1 session in 100, and about twice the top
  of the band above. It is REPORTED ONLY, as a headroom line, and no machine
  count uses it. Sizing every session at the busiest one in a hundred would
  buy hardware for a population that does not exist.

  The rate counts operations of all three types, split by that population's
  own traffic mix, so about one operation in ten of a default session's
  traffic is an agent-mode search - without that, dividing a tier's rate by a
  session rate made only of adds and plain searches would count the agent-mode
  share twice.

  An automated client is a program that sends requests in a loop rather than a
  person; one running a 5-second tool loop is estimated at 0.4 operations per
  second. That figure is backed by TraceLab (2026,
  https://arxiv.org/abs/2606.30560), which instrumented about 4,300 real
  Claude Code and Codex coding-agent sessions: roughly 350,000 model steps and
  430,000 tool calls from 43 developers over about 8 months. A step is one
  model generation plus the tool call it asks for. The MEDIAN step took 5.0
  seconds (4.9 s generating, 0.1 s executing the tool), which at two
  operations per step is 0.40 operations per second - this constant, and the
  same 5-second tool loop this program has always described. How many times a
  human that is comes out of the constants themselves (0.4 / 0.028 = 14 and
  0.4 / 0.011 = 36, so 14 to 36 times), rather than being written down where
  it could drift.

  Sustained automated client: 0.07 operations per second, from TraceLab's MEAN
  step of 28.3 seconds (11.5 s generating, 16.8 s executing the tool - the gap
  between mean and median is a tail of slow tool calls). It is REPORTED ONLY,
  as an average-load line, and no machine count uses it. The two automated
  figures differ by about six times because an agent is idle most of the
  wall-clock time, waiting on the person: TraceLab measured human thinking at
  92.3% of session wall-clock time. Anthropic's production Claude Code data
  (2026, https://www.anthropic.com/research/claude-code-expertise: about 10
  actions per prompt, median turn about 45 seconds) implies roughly 0.22
  actions per second within a turn, between the two. Sizing from the sustained
  figure would under-provision a burst; sizing from the design peak
  over-provisions a day. This model sizes for the worst five minutes, so it
  uses 0.4. TraceLab is 43 developers using coding agents, not a measurement
  of automated clients in general.

  Meter real operations per second per API key from the first day of the pilot
  and re-check the tier choice against it. A population of nobody demands no
  operations, and the report says there is nothing to size rather than naming
  the pilot tier for it.

Peak against average: multiply an average load by 4. REPORTED ONLY - no machine
  count in this program reads the figure, and nothing here multiplies by it for
  you. It is for a reader who knows their AVERAGE load rather than their peak
  and needs to turn one into the other before typing it in. The evidence is
  measured, and it disagrees by scale: a service for a few hundred users ran at
  4.64 times its own mean and Hotmail at 1.64 times (Wang et al. 2012,
  https://arxiv.org/abs/1207.6295), while a Google production cell of 12,500
  machines ran at 1.3 times (Reiss et al., SoCC '12,
  https://www.cs.virginia.edu/~cr4bd/papers/socc12.pdf). 4 is the figure for a
  deployment inside one organisation, which is the small case. A small
  population is burstier because scale flattens the curve: one team going to
  lunch together is visible in a few hundred users and invisible in a few
  million.

Each population carries its own traffic mix, because the kind of caller and the
  kind of request are correlated: a room full of people asking one question at
  a time is not the same traffic as a fleet of automated clients that use
  agent-mode search on nearly every call. The users subcommand takes
  --human-mix and --automated-mix, each three numbers as adds/plain/agent-mode,
  and both default to the model's own default mix so that leaving them off
  changes nothing. It then reports a blended mix: each population's mix
  weighted by the operations that population demands, at the busy end of the
  human rate, which is the rate the report tells you to plan for. That blended
  mix, and not the global default, is what sizes the deployment for a
  population.

--------------------------------------------------------------------------------
ASSUMPTIONS - choices, not findings
--------------------------------------------------------------------------------

Traffic mix: per 100 operations, 45 adds, 45 plain searches, 10 agent-mode
  searches (a request flag, not a kind of caller). NOBODY HAS MEASURED THIS. It
  is a planning assumption about how the service will be used, and it is the
  second-biggest lever in the whole model
  after the embedding card rate. The agent-mode share in particular sets the
  hardware order more than any tuning does, because one agent-mode search costs
  about 22 plain searches. The mix is adjustable on every subcommand of this
  program, and the tier report always prints a sensitivity table showing what
  happens to the API server count at 0, 2, 10 and 25 agent-mode searches per
  second, plus the deployment's own agent-mode rate when that is not already one
  of them - so the table always contains the row that matches the headline. That
  row is marked "this run", because two rates that a rounded label cannot tell
  apart, such as 2.0 and 2.04, are still two different sizings.

Tiers: pilot 20 ops/s, target 100 ops/s, scale 1,000 ops/s. Each is a DESIGN
  PEAK - the worst rate the system must sustain for five minutes - and not an
  average. Five minutes is not this program's invention. ITU-T E.500, the
  international recommendation for measuring telephone traffic, independently
  requires measurement windows "greater than 5 minutes... so that resources
  are not dimensioned for infrequent small interval peak traffic levels"
  (https://www.itu.int/rec/T-REC-E.500-199811-I/en). A design peak must be greater than zero: zero operations per second
  is not a deployment to size, so the program refuses it with a message and
  exit code 2 rather than printing a report that orders machines for no
  traffic. It must also be below MAX_OPS_PER_S, and retention, dimensions,
  bytes per number and the vector-store machine size each have a bound of
  their own. Those bounds change no machine count. They are there so that a
  number far larger than any deployment is refused by name, instead of passing
  the finite check and then overflowing to infinity part-way through the byte
  arithmetic - which used to come back as a complaint about "inf", a value
  nobody had typed. Nothing in the 30 August run demonstrates five minutes of sustained
  service: every test run started from a freshly restarted pod because of an
  unresolved fault in which searches hang while the health endpoint still
  answers. These are clean-start numbers.

Utilization ceiling 60% on API servers and on GPU cards. One spare card on every
  GPU role.

Retention 90 days by default, purely as a placeholder. Retention is undecided and
  it is the decision that moves storage the most. Vector dimensions default to
  1,024 and storage to 1 byte per number (int8 quantization); both must be fixed
  before the first episode is ingested, because changing either later means
  re-embedding everything. Dimensions must be a whole number: a vector cannot
  hold 1,024.7 numbers, and a fraction is refused rather than quietly cut down
  to 1,024, precisely because the count has to be right before the first
  episode is stored.

Qdrant nodes are filled to at most 70% of their RAM, leaving room for the
  operating system, for Qdrant's own metadata and for shards that come out
  uneven. This fixes a real defect found in review: an earlier version of this
  model filled seven 768 GB servers to 5.376 TB with a requirement of 5.375 TB,
  which left no headroom at all. Node RAM options are 256, 512 and 768 GB.
  Unless a size is forced - with --node-gb, or with the "RAM per vector-store
  machine" box on the web form - the program prints the node count for all
  three sizes and recommends the one that buys the least total RAM, breaking a
  tie towards fewer machines. A forced size is used whatever it costs, and is
  added to the comparison table when it is not one of the three. The report
  names whichever of the two set it, so a reader of the web page is never told
  they typed a command-line flag.
  When the chosen size is more than 95%
  full within that 70% allowance the report prints a WARNING and the result
  carries a qdrant_tight_fit flag: at that point a small increase in retention
  adds a whole extra machine, so the tier is one policy change away from a
  bigger order. The 95% figure is a display threshold only - it changes no
  machine count.
  A deployment that searches or writes vectors at all orders at least one
  vector-store machine, whatever the stored bytes come to. A search-only
  traffic mix stores nothing, and so does a retention of zero days, and both
  used to come back as zero machines beside a demand table asking for hundreds
  of vector searches a second.

The report states no machine class for the PostgreSQL and vector-store
  machines. Only the API server has one that was measured. The RAM of a
  vector-store machine is chosen by the model, so the report gives it; the vCPU
  of either machine, and the RAM of the PostgreSQL machine, are undecided and
  the report says so rather than borrowing the API server's figures.

One PostgreSQL server per tier. This is an ASSUMPTION, not a derived count.
  PostgreSQL was never benchmarked at these statement rates. What the 30 August
  run did establish is that the failures seen were connection limits, not
  compute.

On the web form, an empty box is a blank answer and not a request for the
  default. The command line takes the default for a flag that is left off, and
  a bare /api/calc call with no parameters does the same; but a form
  submission always sends every box, so an empty one means the reader cleared
  it, and the page says which box is blank instead of inventing a number. The
  "RAM per vector-store machine" box is the one exception, because empty there
  already means "choose the size for me" and the hint under the box says so.

A caller population, when one is given, sizes the deployment. The form asks
  about the same traffic in two ways - a design peak with a traffic mix, and a
  population of callers - and only one of the two can size the machines. With
  both caller-count boxes empty the design peak and mix boxes size it, exactly
  as they did before the form could take a population. With either count
  filled in, the population sizes it: the design peak is the demand that
  population makes at the busy end of the human rate and the mix is its
  blended mix, which is what the users subcommand does with the same numbers,
  so the page and the command line order the same hardware. A population that
  sends nothing at all is not a deployment to size, and the design peak box
  governs after all. The page says which of the two it used and quotes back a
  design peak it did not use, so nobody has to guess whether the box they
  typed in mattered.
  Both caller counts are CONCURRENT sessions - callers active at the same
  moment, not registered accounts and not visitors in a day - because each
  count is multiplied by a rate per session.

Users to concurrent sessions, on four EXAMPLE DEFAULTS. Whoever asks for a
  deployment answers in users - "50,000 users" - and the model counts
  concurrent sessions. Two conversions sit between the two, and they multiply.
  The first is the share of the user base using the service at the busiest
  moment: for most services a few per 100 users, so 50,000 users may be 500
  concurrent sessions or 5,000, a tenfold spread that moves the answer across
  two whole tiers. The second is how many sessions one active user holds at
  that moment: about one for a person, perhaps two across devices, but ten or
  fifty for one person running an automated client framework, so a count of
  automated sessions may bear almost no relation to a count of people.

  Each of the four has a default, so a bare user count always gets an answer.
  Every one of them is an EXAMPLE, is labelled "example default" in the report
  wherever it is printed, and is meant to be replaced.

    --human-peak-share defaults to 10 per 100 users. This is a CONVENTION, not
      a measurement. Microsoft's SharePoint capacity guidance states "A
      concurrency rate of 10 percent is assumed, with 1 percent of concurrent
      users making requests at a given moment. For example, for 10,000 users,
      1,000 users are actively using the solution simultaneously" - note the
      words "is assumed" (https://learn.microsoft.com/en-us/previous-versions/
      office/sharepoint-2007-products-and-technologies/cc263100(v=office.12)).
      Teletraffic engineering offers a comparable figure as an explicit rule
      of thumb: 10 to 16% of telephone subscribers are busy during the busy
      hour (Iversen, Teletraffic Engineering and Network Planning, DTU). Two
      conventions from two different fields that happen to agree is weaker
      evidence than one measurement, and this program says so rather than
      calling either of them measured. The plausible range is 5 to 20 per 100
      users: the figure can be wrong by a factor of two either way.

    --human-sessions-per-active-user defaults to 1. NO PUBLISHED EVIDENCE
      EXISTS for this figure. That is the label, printed in those words.

    --automated-peak-share defaults to 100 per 100 and
      --automated-sessions-per-active-user to 1, which is identity: for a
      program the natural unit is the running client, and a client that is
      running IS a session, so there is nothing to convert. The 10-per-100
      convention describes people, most of whom are asleep or busy at any one
      moment. Applying it to load generators or to deployed clients would
      divide the load by ten.

  On the web form the same six inputs are six boxes with the same names. The
  report prints the multiplication as its own table, and the label column of
  that table marks every figure as either supplied by the reader or an example
  default, so the two can never be mistaken for each other. Whenever an
  example default was used, a note under the table says which figures were
  defaulted, that they are examples, that the share active can be wrong by a
  factor of two either way, and what to replace them with: metered concurrent
  sessions from the first day of running. A reader who already knows their
  concurrent sessions gives --humans and --automated as before, and nothing
  about that answer moves.

Availability additions - a second copy of every vector, a PostgreSQL standby, a
  second gateway - are NOT priced in by this program. They are a separate
  decision, and a replication factor of 2 doubles the hot vector RAM and the
  Qdrant machine count.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import sys
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# =============================================================================
# CONSTANTS
# Every constant is grouped and labelled measured / derived / estimate /
# assumption. Nothing in this program uses a number that is not named here.
# =============================================================================

# --- Tiers ------------------------------------------------------------------
# ASSUMPTION. Each rate is a design peak: the worst rate the system must sustain
# for five minutes, not an average.
TIER_OPS_PER_S = {"pilot": 20.0, "target": 100.0, "scale": 1000.0}
TIER_ORDER = ("pilot", "target", "scale")

# --- Sanity bounds on the inputs that scale the arithmetic -------------------
# ASSUMPTION, and a refusal limit only: no machine count anywhere moves because
# of these. They exist so that a number far larger than any deployment is
# refused by name, rather than passing the finite check at the front of
# size_deployment and overflowing to infinity part-way through the byte
# arithmetic - which came back as a complaint about "inf", a value the reader
# never typed. Every bound is far above the largest tier (1,000 operations/s)
# and far below the point at which the arithmetic overflows.
MAX_OPS_PER_S = 1e9                 # a billion operations a second
MAX_RETENTION_DAYS = 36_500         # 100 years
MAX_VECTOR_DIMS = 1_000_000         # embedding models today are 384 to 4,096
MAX_BYTES_PER_VALUE = 64            # int8 is 1, float64 is 8
MAX_NODE_GB = 1_000_000             # a petabyte of RAM in one machine
MAX_USERS = 1e12                    # more users than there are people
# A share of a user base is written per 100 users, so all of them is 100. This
# is a definition rather than a guess, but it is refused by name for the same
# reason as the bounds above: 150 per 100 users is not a share.
MAX_PEAK_ACTIVE_PER_100 = 100.0
MAX_SESSIONS_PER_ACTIVE_USER = 10_000

# How much floating-point dust a machine count is allowed to absorb, as a
# fraction of the count itself. See ceil_up.
FLOAT_DUST_FRACTION = 1e-9

# --- Traffic mix ------------------------------------------------------------
# ASSUMPTION, never measured. Operations per 100 operations. The third share is
# agent-mode search, which is a request flag and not a kind of caller.
DEFAULT_MIX_ADD = 45.0
DEFAULT_MIX_PLAIN = 45.0
DEFAULT_MIX_AGENT = 10.0
MIX_TOTAL = 100.0
# How a whole mix is written where it has to fit in one flag or one box:
# three numbers separated by "/" or by ",", so 45/45/10 and 45,45,10 are read
# the same way. See parse_mix_text.
MIX_TRIPLE_EXAMPLE = "45/45/10"

# --- Fan-out per request ----------------------------------------------------
# DERIVED by reading the MemMachine source code on 30 August 2026.
ADD_EMBEDS = 1
ADD_VECTOR_WRITES = 1
ADD_POSTGRES_STATEMENTS = 2
ADD_LLM_CALLS = 0

PLAIN_EMBEDS = 2                    # 2 today
PLAIN_EMBEDS_WITH_TYPES_FIX = 1     # 1 once every request sends types:["episodic"]
PLAIN_VECTOR_SEARCHES = 1
PLAIN_POSTGRES_STATEMENTS = 2
PLAIN_LLM_CALLS = 0

AGENT_EMBEDS = 22
AGENT_VECTOR_SEARCHES = 22
AGENT_POSTGRES_STATEMENTS = 44
AGENT_LLM_CALLS_LOW = 1.0           # ESTIMATE: 1 to 2 language-model calls
AGENT_LLM_CALLS_HIGH = 2.0
AGENT_LLM_CALLS_PLANNING = 1.5      # ESTIMATE: the midpoint, used for sizing

# --- API servers ------------------------------------------------------------
# MEASURED 30 Aug 2026: 178.66 ops/s core API and 180.31 ops/s full platform,
# both at 8 workers and 128 concurrent requests, real OpenAI
# text-embedding-3-small at ~180-190 ms, 12,000-episode corpus, top_k 10,
# expand 0, rrf-hybrid reranker on, Qdrant and PostgreSQL each on their own host,
# every serving host an AWS c8a.4xlarge (16 vCPU, 32 GiB, AMD EPYC Turin).
API_SEARCHES_PER_S_PER_SERVER = 180.0
API_UTILIZATION_CEILING = 0.60      # ASSUMPTION: fill a server to at most 60%
API_WORKERS_PER_SERVER = 8          # MEASURED 30 Aug 2026: 8 is the knee
API_SERVER_VCPU = 16                # the machine class that was measured
API_SERVER_RAM_GB = 32
# ESTIMATE: one add costs at most one plain search of API work. Rounds up.
ADD_COST_IN_PLAIN_SEARCH_EQUIVALENTS = 1.0

# --- Embedding GPUs ---------------------------------------------------------
# ESTIMATE, never benchmarked on any card with the planned model.
EMBED_CARD_REQUESTS_PER_S_LOW = 300.0
EMBED_CARD_REQUESTS_PER_S_HIGH = 500.0
GPU_UTILIZATION_CEILING = 0.60      # ASSUMPTION
GPU_SPARE_CARDS = 1                 # ASSUMPTION: one spare on every GPU role

# --- Agent-model GPUs -------------------------------------------------------
# ESTIMATE: an 8B-class card serves 10-20 language-model calls/s; plan on 15.
AGENT_LLM_CALLS_PER_S_PER_CARD = 15.0

# --- Vector store (Qdrant) --------------------------------------------------
DEFAULT_VECTOR_DIMS = 1024          # ASSUMPTION, must be fixed before first ingest
DEFAULT_BYTES_PER_VALUE = 1         # ASSUMPTION: int8 quantization
QDRANT_INDEX_OVERHEAD_FACTOR = 1.5  # ASSUMPTION: index overhead on hot RAM
QDRANT_NODE_RAM_OPTIONS_GB = (256, 512, 768)
# How the report names the thing that forced the vector-store machine size.
# The reader of the web page never typed a command-line flag, so the page must
# not tell them they did.
NODE_GB_SOURCE_CLI = "--node-gb"
NODE_GB_SOURCE_WEB = "the RAM per vector-store machine box"
# How the report names what supplied the design peak and the traffic mix, when
# they were not given directly. A run sized from a caller population took both
# from that population, so the Inputs table must not label them as numbers
# somebody typed.
SIZED_FROM_POPULATION = "the caller population"
QDRANT_NODE_FILL_LIMIT = 0.70       # ASSUMPTION: at most 70% of a node's RAM
QDRANT_TIGHT_FIT_WARN_FRACTION = 0.95   # display only: warn when this full
SECONDS_PER_DAY = 86400
DEFAULT_RETENTION_DAYS = 90         # ASSUMPTION, placeholder - retention undecided
BYTES_PER_GB = 1_000_000_000        # GB means 10^9 bytes throughout
BYTES_PER_TB = 1_000_000_000_000

# --- Disk -------------------------------------------------------------------
# ESTIMATE. Qdrant keeps the full-precision original vector on disk even when the
# searchable copy in RAM is quantized.
ORIGINAL_VECTOR_BYTES_PER_VALUE = 4
QDRANT_DISK_PAYLOAD_BYTES_PER_EPISODE = 256
QDRANT_DISK_OVERHEAD_FACTOR = 1.3

# ESTIMATE. PostgreSQL holds the episode text.
EPISODE_TEXT_BYTES_LOW = 800
EPISODE_TEXT_BYTES_HIGH = 2400
POSTGRES_ROW_OVERHEAD_BYTES = 400
POSTGRES_INDEX_BYTES_PER_EPISODE = 300
POSTGRES_BLOAT_FACTOR = 1.4

# --- PostgreSQL connections -------------------------------------------------
# MEASURED 30 Aug 2026 (pool size 5 + max_overflow 10 per worker).
POSTGRES_POOL_SIZE = 5
POSTGRES_MAX_OVERFLOW = 10
POSTGRES_CONNECTIONS_PER_WORKER = POSTGRES_POOL_SIZE + POSTGRES_MAX_OVERFLOW
GATEWAY_CONNECTIONS_PER_API_SERVER = 20     # ASSUMPTION: gateway allowance
POSTGRES_CHART_DEFAULT_MAX_CONNECTIONS = 100    # MEASURED: the default that failed
POSTGRES_PROVEN_MAX_CONNECTIONS = 600           # MEASURED: cleared every error
POSTGRES_SERVERS_PER_TIER = 1               # ASSUMPTION, never benchmarked

# --- Network message sizes --------------------------------------------------
# ESTIMATE, every one. Named here so the network figures can be checked by hand.
NS_ADD_REQUEST_BYTES = 1200
NS_ADD_RESPONSE_BYTES = 300
NS_SEARCH_REQUEST_BYTES = 600
NS_RESPONSE_BYTES_PER_EPISODE = 900
PLAIN_SEARCH_EPISODES_RETURNED = 10     # top_k 10, the measured configuration
AGENT_SEARCH_EPISODES_RETURNED = 20     # ESTIMATE
NS_AGENT_ANSWER_BYTES = 2000            # ESTIMATE: the written answer

EMBED_REQUEST_BYTES = 1000
EMBED_RESPONSE_ENVELOPE_BYTES = 200
QDRANT_SEARCH_REQUEST_ENVELOPE_BYTES = 300
QDRANT_CANDIDATES_PER_SEARCH = 50       # vector_search_limit 50, measured config
QDRANT_BYTES_PER_CANDIDATE = 200
QDRANT_UPSERT_ENVELOPE_BYTES = 500
QDRANT_UPSERT_RESPONSE_BYTES = 200
POSTGRES_BYTES_PER_STATEMENT = 1800     # both directions
LLM_CALL_REQUEST_BYTES = 8000
LLM_CALL_RESPONSE_BYTES = 2000
NETWORK_PROTOCOL_OVERHEAD_FACTOR = 1.2  # TLS, HTTP and TCP framing
BITS_PER_BYTE = 8
BITS_PER_MBIT = 1_000_000               # Mbps means 10^6 bits per second

# --- Callers ----------------------------------------------------------------
# ESTIMATE, but no longer a guess: each of the two rates below now has a
# published measurement behind it, named in full so a reader can check it.
# These describe how fast a CALLER sends requests. Nothing here says anything
# about what kind of request it sends: that is the traffic mix above.
#
# BurstGPT is a public workload dataset: 110 consecutive days of one real
# Azure OpenAI deployment, released with the KDD '25 paper "BurstGPT: A
# Real-world Workload Dataset to Optimize LLM Serving Systems" (2025,
# https://arxiv.org/abs/2401.17644, dataset
# https://github.com/HPMLL/BurstGPT). Version 2 of the release added a session
# identifier, which is what makes a per-session rate recoverable at all.
# Across 55,295 conversation sessions and 176,466 gaps between one prompt and
# the next, the busiest five minutes of a session ran at a median of 0.0067
# prompts per second, a 90th percentile of 0.0167 and a 99th percentile of
# 0.030. At OPS_PER_HUMAN_PROMPT operations per prompt those are 0.013, 0.033
# and 0.060 operations per second. The band below therefore brackets the
# median to about the 90th percentile of a session's busiest five minutes.
# What BurstGPT is not: it is one regional deployment, and who its users were
# is not published. Meter your own API keys before trusting it.
BURSTGPT_CITATION = (
    "BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems "
    "(KDD '25, 2025), https://arxiv.org/abs/2401.17644, dataset "
    "https://github.com/HPMLL/BurstGPT")
HUMAN_SESSION_OPS_PER_S_LOW = 0.011
HUMAN_SESSION_OPS_PER_S_HIGH = 0.028
# The 99th percentile of the same measurement: a heavy session, one in a
# hundred. REPORTED ONLY. It appears as a headroom line in the report and it
# is never multiplied into a machine count, because sizing every session at
# the busiest one in a hundred would buy hardware for a population that does
# not exist.
HUMAN_SESSION_OPS_PER_S_HEAVY = 0.06

# An automated client is a program that sends requests in a loop rather than a
# person typing in a chat window. It is NOT the same thing as agent-mode
# search, which is a flag on one request.
#
# TraceLab instrumented about 4,300 real Claude Code and Codex coding-agent
# sessions - roughly 350,000 model steps and 430,000 tool calls from 43
# developers over about 8 months (2026, https://arxiv.org/abs/2606.30560). A
# step is one model generation plus the tool call it asks for. The MEDIAN step
# took 5.0 seconds (4.9 s of model generation, 0.1 s of tool execution), which
# at 2 operations per step is 0.40 operations per second - the figure below,
# and the same "5-second tool loop" this program has always described. The
# MEAN step took 28.3 seconds (11.5 s + 16.8 s), because a few tool calls are
# very slow, and that works out at 0.07 operations per second. Human thinking
# was 92.3% of session wall-clock time. Anthropic's own production Claude Code
# figures - about 10 actions per prompt and a median turn of about 45 seconds
# (2026, https://www.anthropic.com/research/claude-code-expertise) - imply
# about 0.22 actions per second inside a turn, between the two.
# What TraceLab is not: 43 developers using coding agents. It is not a
# measurement of automated clients in general.
TRACELAB_CITATION = (
    "TraceLab (2026), https://arxiv.org/abs/2606.30560")
# The planning figure, and a DESIGN PEAK: this model sizes for the worst five
# minutes it must sustain, and an agent that is actively working runs at the
# median step pace.
AUTOMATED_CLIENT_OPS_PER_S = 0.4
# The sustained rate over a whole session, from the mean step. REPORTED ONLY,
# for average-load planning; no machine count uses it. The two differ by about
# six times because an agent is idle most of the wall-clock time, waiting on
# the person. Sizing from the sustained figure would under-provision a burst;
# sizing from the design peak over-provisions a day.
AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED = 0.07
# Used only to describe where the human figures come from: about two
# operations per prompt, so 0.011 ops/s is about 20 prompts an hour and 0.028
# ops/s is about 50. The spread between a human chat session and an automated
# client is worked out from the constants above rather than written down, so it
# cannot drift.
OPS_PER_HUMAN_PROMPT = 2.0

# The flag --agents used to mean "how many callers are programs". It was one
# letter away from --agent, which is the agent-mode share of the traffic mix,
# and the two mean completely different things. It is refused by name now
# rather than being accepted quietly or read as the mix share.
AGENT_VERSUS_AUTOMATED_SENTENCE = (
    "An automated client is a CALLER - a program that sends requests in a "
    "loop - while agent-mode search is a property of one REQUEST, not of who "
    "sent it.")
RETIRED_AGENTS_FLAG_MESSAGE = (
    "--agents is no longer a flag. Use --automated for the number of "
    "automated clients, and --agent for the agent-mode share of the traffic "
    "mix. " + AGENT_VERSUS_AUTOMATED_SENTENCE)
RETIRED_AGENTS_SETTING_MESSAGE = (
    'the web address has a setting called "agents", which this calculator '
    'does not know. Use "automated" for the number of automated clients, and '
    '"agent" for the agent-mode share of the traffic mix. '
    + AGENT_VERSUS_AUTOMATED_SENTENCE)

# --- Users to concurrent sessions -------------------------------------------
# The model multiplies a count of CONCURRENT SESSIONS by a rate per session.
# Whoever commissions a deployment answers in USERS - "50,000 users" - and two
# conversions sit between the two, and they multiply: the share of the user
# base using the service at the busiest moment, and the number of sessions one
# active user holds at that moment.
#
# Read the counting rule below before either of them. It is the sentence that
# decides what a session is, and a reader who gets it wrong gets an answer
# wrong by a factor of ten whatever the conversion figures say.
COUNTING_RULE = (
    "The unit this model counts is a session sending requests at the same "
    "moment, whoever is behind it. One developer driving a ten-user load test "
    "is ten sessions, not one. If you already know how many sessions will be "
    "in flight at once, give that number directly as a count of concurrent "
    "sessions and skip the conversion. A user count and a share of users "
    "active at the busiest moment exist only for estimating that number from "
    "a population of people.")
# The same rule, in the names the command line uses. The web page must never
# be told it typed a flag, so the flag names live in their own sentence.
COUNTING_RULE_FLAGS = (
    "On the command line the two session counts are --humans and --automated.")

# Each of the four conversion figures below is an EXAMPLE DEFAULT. It exists
# so that a bare user count always gets an answer. It is labelled "example
# default" everywhere it is printed, and it is there to be replaced.
#
# Share of a user base active at the busiest moment: 10 per 100 users. This is
# a CONVENTION, not a measurement, and every place this program prints it says
# so. Microsoft's SharePoint capacity guidance states "A concurrency rate of
# 10 percent is assumed, with 1 percent of concurrent users making requests at
# a given moment" - note the words "is assumed". Teletraffic engineering
# offers a comparable figure as an explicit rule of thumb: 10 to 16% of
# telephone subscribers are busy during the busy hour. Two conventions from
# two different fields that happen to agree is weaker evidence than one
# measurement, which is why neither is called measured here. The plausible
# range is 5 to 20 per 100 users, so the figure can be wrong by a factor of
# two either way.
SHAREPOINT_CITATION = (
    "Microsoft SharePoint capacity planning guidance, "
    "https://learn.microsoft.com/en-us/previous-versions/office/"
    "sharepoint-2007-products-and-technologies/cc263100(v=office.12)")
SHAREPOINT_QUOTE = (
    "A concurrency rate of 10 percent is assumed, with 1 percent of "
    "concurrent users making requests at a given moment. For example, for "
    "10,000 users, 1,000 users are actively using the solution "
    "simultaneously")
TELETRAFFIC_CITATION = (
    "Iversen, Teletraffic Engineering and Network Planning, DTU")
DEFAULT_HUMAN_PEAK_ACTIVE_PER_100 = 10.0
# REPORTED ONLY. The two ends of the plausible range are printed so a reader
# can see how far the figure can move; no count reads either of them.
HUMAN_PEAK_ACTIVE_PER_100_LOW = 5.0
HUMAN_PEAK_ACTIVE_PER_100_HIGH = 20.0
# Sessions one active person holds: 1. NO PUBLISHED EVIDENCE EXISTS for this
# figure. That is not a shorthand for "we did not look" - it is the label, and
# it is printed in those words beside the number.
DEFAULT_HUMAN_SESSIONS_PER_ACTIVE_USER = 1.0
# The automated side converts by identity: 100 per 100 users, 1 session each.
# For a program the natural unit is the running client, and a client that is
# running IS a session, so there is nothing to convert. The 10-per-100
# convention above describes people, most of whom are asleep or busy at any
# one moment; applying it to load generators or to deployed clients would
# divide the load by ten.
DEFAULT_AUTOMATED_PEAK_ACTIVE_PER_100 = 100.0
DEFAULT_AUTOMATED_SESSIONS_PER_ACTIVE_USER = 1.0

SHARE_ACTIVE_MEANS = (
    "is the share of those users who are using the service at the busiest "
    "moment, written per 100 users: for most services that is a few per 100, "
    "and 1 rather than 10 moves the answer tenfold")
SESSIONS_PER_ACTIVE_USER_MEANS = (
    "is how many sessions one of those active users holds at that moment: "
    "about 1 for a person, but 10 or 50 for one person running automated "
    "clients")
AUTOMATED_IDENTITY_SENTENCE = (
    "For automated clients the example default is identity - "
    f"{DEFAULT_AUTOMATED_PEAK_ACTIVE_PER_100:g} per 100 users and "
    f"{DEFAULT_AUTOMATED_SESSIONS_PER_ACTIVE_USER:g} session each - because "
    "for a program the natural unit is the running client, and a client that "
    "is running is already a session. The "
    f"{DEFAULT_HUMAN_PEAK_ACTIVE_PER_100:g}-per-100 convention describes "
    "people, most of whom are asleep or busy at any one moment; applying it "
    "to load generators or to deployed clients would divide the load by ten.")

# What the report calls the two figures the reader has to supply. They are not
# this model's assumptions, and the label has to say whose they are.
LABEL_READER_ASSUMPTION = "assumption supplied by the reader, not by this model"
LABEL_READER_INPUT = "given by the reader"
LABEL_CONVERSION_DERIVED = (
    "derived: users x share active / 100 x sessions per active user")
# What one column of the conversion table says for a caller kind whose
# concurrent sessions were typed in directly rather than worked out from users.
CONVERSION_NOT_USED_CELL = "given as sessions"
# How the conversion table marks one figure, so that a reader can never
# mistake a figure this program chose for a figure they gave it.
CONVERSION_MARK_SUPPLIED = "supplied"
CONVERSION_MARK_DEFAULT = "example default"
# What the two caller kinds are called in that label.
CONVERSION_COLUMN_HUMAN = "people"
CONVERSION_COLUMN_AUTOMATED = "automated clients"
# Why each example default has the value it has, in the fewest honest words.
WHY_HUMAN_SHARE_DEFAULT = "a convention, not a measurement"
WHY_HUMAN_PER_USER_DEFAULT = "no published evidence exists"
WHY_AUTOMATED_IDENTITY = "a client that is running is already a session"

# --- Peak load against average load -----------------------------------------
# REPORTED ONLY. No machine count reads any of the four numbers below. They
# are here for a reader who knows their AVERAGE load and needs a multiplier to
# reach a design peak.
#
# Measured evidence, and it disagrees with itself by scale. A service for a
# few hundred users ran at 4.64 times its own mean, and Hotmail at 1.64 times;
# a Google production cell of 12,500 machines ran at 1.3 times. A small
# population is burstier because scale flattens the curve: one team going to
# lunch together is visible in a few hundred users and invisible in a few
# million. A MemMachine deployment inside one organisation is the small case,
# so 4x is the figure this program reports.
WANG_CITATION = "Wang et al. (2012), https://arxiv.org/abs/1207.6295"
GOOGLE_TRACE_CITATION = (
    "Reiss et al., SoCC '12, "
    "https://www.cs.virginia.edu/~cr4bd/papers/socc12.pdf")
PEAK_TO_AVERAGE_SINGLE_ORGANISATION = 4.0
PEAK_TO_AVERAGE_MEASURED_SMALL_SERVICE = 4.64
PEAK_TO_AVERAGE_LARGE_SERVICE_LOW = 1.3
PEAK_TO_AVERAGE_LARGE_SERVICE_HIGH = 1.64

# --- The five-minute window -------------------------------------------------
# Every tier rate in this program is a design peak: the worst rate the system
# must sustain for FIVE MINUTES. Five minutes is not this program's
# invention. ITU-T E.500, the international recommendation for measuring
# telephone traffic, independently requires measurement windows "greater than
# 5 minutes... so that resources are not dimensioned for infrequent small
# interval peak traffic levels".
ITU_E500_CITATION = (
    "ITU-T E.500 (1998), https://www.itu.int/rec/T-REC-E.500-199811-I/en")
ITU_E500_QUOTE = (
    "greater than 5 minutes... so that resources are not dimensioned for "
    "infrequent small interval peak traffic levels")

# What the report calls each of the four per-caller rates. Each names the
# published source it came from; the note above each table gives the full
# reference with its web address, because a label column that carried a URL
# would be wider than the table it sits in.
LABEL_HUMAN_RATE_BAND = (
    "estimate - BurstGPT 2025: median to about the 90th percentile of a "
    "session's busiest five minutes")
LABEL_HUMAN_RATE_HEAVY = (
    "estimate - BurstGPT 2025: the 99th percentile. Headroom only - no "
    "machine count uses it")
LABEL_AUTOMATED_RATE_PEAK = (
    "estimate - TraceLab 2026: the measured 5.0-second median step, a design "
    "peak")
LABEL_AUTOMATED_RATE_SUSTAINED = (
    "estimate - TraceLab 2026: the 28.3-second mean step. Average load only - "
    "no machine count uses it")

# --- Sensitivity ------------------------------------------------------------
# The agent-mode search rates the tier report always shows.
SENSITIVITY_AGENT_RATES = (0.0, 2.0, 10.0, 25.0)

# Where ``validate`` writes its JSON by default: a file in the current working
# directory, overridable with --out.
NUMBERS_FILE = "sizing-numbers.json"

# --- Web server -------------------------------------------------------------

# Seconds the server waits on one connection before dropping it. This is a
# local development server, not a service: the timeout is here so a client
# that connects and then goes quiet cannot hold a thread and a file descriptor
# open indefinitely. It changes no machine count.
SERVER_REQUEST_TIMEOUT_S = 10


# =============================================================================
# ERRORS AND SMALL HELPERS
# =============================================================================


class SizingError(ValueError):
    """Bad input. The command line turns this into a message and exit code 2."""


# Above this, a float can no longer hold every whole number, so its digits are
# an artefact of the format rather than anything anyone typed. Such a value is
# printed the short way, as 1e+307 rather than as 309 digits.
LARGEST_EXACTLY_COUNTABLE = 1e16


def as_given(value) -> str:
    """A number printed as it was given, with nothing rounded away.

    An ordinary whole number loses its ".0" and gains thousands separators;
    anything else keeps every digit it has. Both the report, which echoes its
    own inputs, and the errors, which quote the value they refuse, need this: a
    rounded copy describes something the reader did not ask for. ":.0f" turned
    half a byte per number into "0" above a table sized for half a byte, and
    ":g" turned 1,000,000,001 into "1e+09", which is the very limit the message
    says it exceeds.
    """
    if isinstance(value, int):
        return f"{value:,}"
    if (isinstance(value, float) and value.is_integer()
            and abs(value) < LARGEST_EXACTLY_COUNTABLE):
        return f"{int(value):,}"
    return repr(value)


def ceil_up(value: float, per_unit: float) -> int:
    """Divide and always round up to a whole machine.

    The tiny subtraction absorbs floating-point dust, so that a value that is
    mathematically exactly 3.0 does not come out as 4. It is a fraction OF THE
    ANSWER rather than a flat amount: a flat 1e-9 is dust next to 3 machines
    but is larger than the whole answer for a very small workload, and it used
    to turn any work below about 1.08e-7 searches per second into zero
    machines. Any work at all has to run somewhere, so the answer is never
    less than one machine once the work is greater than zero.
    """
    if not math.isfinite(per_unit) or per_unit <= 0:
        raise SizingError("cannot divide by a capacity of zero or less")
    if not math.isfinite(value):
        raise SizingError(
            f"cannot size for a quantity of {value:g} - every input must be a "
            "finite number")
    if value <= 0:
        return 0
    quotient = value / per_unit
    return max(1, math.ceil(quotient - abs(quotient) * FLOAT_DUST_FRACTION))


@dataclass(frozen=True)
class TrafficMix:
    """How 100 operations split between the three request types."""

    add: float = DEFAULT_MIX_ADD
    plain: float = DEFAULT_MIX_PLAIN
    agent: float = DEFAULT_MIX_AGENT

    def validate(self) -> None:
        for name, value in (("add", self.add), ("plain", self.plain),
                            ("agent", self.agent)):
            if not math.isfinite(value):
                raise SizingError(
                    f"traffic mix: {name} is {value:g}, but a share must be a "
                    "finite number")
            if value < 0:
                raise SizingError(
                    f"traffic mix: {name} is {value}, but a share cannot be "
                    "negative")
        total = self.add + self.plain + self.agent
        if abs(total - MIX_TOTAL) > 1e-6:
            raise SizingError(
                f"traffic mix must add up to {MIX_TOTAL:g} operations per 100, "
                f"but {self.add:g} adds + {self.plain:g} plain searches + "
                f"{self.agent:g} agent-mode searches = {total:g}")

    def as_dict(self) -> dict:
        return {"add": self.add, "plain": self.plain, "agent": self.agent}

    def as_words(self) -> str:
        """The mix as the report prints it, in the report's own wording."""
        return (f"{as_given(self.add)} adds, {as_given(self.plain)} plain "
                f"searches, {as_given(self.agent)} agent-mode searches")


def number_or_none(text: str):
    """One piece of text as a number, or None when it is not one."""
    try:
        return float(text)
    except (TypeError, ValueError, OverflowError):
        return None


def default_mix_text() -> str:
    """The default traffic mix, written the way a mix flag or box takes it."""
    return "/".join(as_given(share) for share in
                    (DEFAULT_MIX_ADD, DEFAULT_MIX_PLAIN, DEFAULT_MIX_AGENT))


def parse_mix_text(text: str, called: str) -> TrafficMix:
    """Read a traffic mix written as three numbers, adds/plain/agent-mode.

    Both 45/45/10 and 45,45,10 are accepted, because a reader who has just
    typed a mix on the command line should not have to remember which
    separator this program wanted. ``called`` is how the message names
    whatever holds the text - a flag on the command line, a box on the web
    form - so a refusal points at the thing the reader typed into.
    """
    typed = str(text).strip()
    parts = [part.strip()
             for part in typed.replace(",", "/").split("/")
             if part.strip() != ""]
    if len(parts) != 3:
        raise SizingError(
            f'{called} is "{typed}", but it must be three numbers written as '
            "adds/plain/agent-mode, such as 45/45/10 or 45,45,10 - "
            f"that is {len(parts)} number(s)")
    numbers = [number_or_none(part) for part in parts]
    if any(value is None for value in numbers):
        bad = parts[numbers.index(None)]
        raise SizingError(
            f'{called} is "{typed}", but "{bad}" in it is not a number - '
            f"write it as adds/plain/agent-mode, such as {MIX_TRIPLE_EXAMPLE}")
    mix = TrafficMix(*numbers)
    # The shares check names the mix but not what holds it, and a reader with
    # two mix flags in one command has to be told which of them is wrong.
    try:
        mix.validate()
    except SizingError as exc:
        raise SizingError(f'{called} is "{typed}": {exc}') from None
    return mix


# =============================================================================
# THE CALCULATION CORE
# Pure arithmetic. Nothing in this section prints anything.
# =============================================================================


def api_servers_for_work(work_per_s: float) -> int:
    """API servers needed for a given number of plain-search-equivalents/s."""
    return ceil_up(work_per_s, api_usable_searches_per_server())


def api_usable_searches_per_server() -> float:
    """Searches per second one API server is planned to carry (derived)."""
    return API_SEARCHES_PER_S_PER_SERVER * API_UTILIZATION_CEILING


def embed_gpu_cards_for_demand(embeds_per_s: float,
                               card_requests_per_s: float) -> int:
    """Embedding GPU cards for a given demand, including one spare card.

    A card is filled to at most GPU_UTILIZATION_CEILING of its rate. No
    embedding demand at all needs no card, and therefore no spare either.
    """
    bare = ceil_up(embeds_per_s, card_requests_per_s * GPU_UTILIZATION_CEILING)
    return bare + GPU_SPARE_CARDS if bare else 0


def agent_gpu_cards_for_demand(llm_calls_per_s: float) -> int:
    """Agent-model GPU cards for a given demand, including one spare card.

    No language-model calls at all needs no card, and therefore no spare either.
    """
    bare = ceil_up(llm_calls_per_s, AGENT_LLM_CALLS_PER_S_PER_CARD)
    return bare + GPU_SPARE_CARDS if bare else 0


def qdrant_node_plan(hot_ram_bytes: float, node_gb=None,
                     least_nodes: int = 0) -> dict:
    """Pick a Qdrant node size and count.

    Every node is filled to at most QDRANT_NODE_FILL_LIMIT of its RAM. With no
    node size given, the program works out the count for all three offered
    sizes and recommends the one that buys the least total RAM, breaking a tie
    towards fewer machines. Give node_gb and that size is used instead, however
    much total RAM it buys; a size that is not one of the three offered is
    added to the table so the comparison still shows it.

    least_nodes is the count no size may go below. The caller sets it to 1 when
    the deployment searches or writes vectors at all, because that work has to
    happen on a machine even when the stored bytes come to nothing: a search-
    only traffic mix stores nothing, and so does a retention of zero days, and
    both used to order zero vector-store machines next to hundreds of vector
    searches a second.
    """
    sizes = list(QDRANT_NODE_RAM_OPTIONS_GB)
    if node_gb is not None:
        if not math.isfinite(node_gb) or node_gb <= 0:
            raise SizingError(
                f"RAM per vector-store machine is {as_given(node_gb)} GB, but "
                "it must be greater than zero")
        if not any(abs(node_gb - size) <= 1e-9 for size in sizes):
            sizes.append(node_gb)
        sizes.sort()
    options = []
    for ram_gb in sizes:
        usable_bytes = ram_gb * BYTES_PER_GB * QDRANT_NODE_FILL_LIMIT
        count = max(ceil_up(hot_ram_bytes, usable_bytes), least_nodes)
        total_ram_gb = count * ram_gb
        fill = (hot_ram_bytes / (count * usable_bytes)) if count else 0.0
        options.append({
            "node_ram_gb": ram_gb,
            "usable_gb_per_node": usable_bytes / BYTES_PER_GB,
            "nodes": count,
            "total_ram_gb": total_ram_gb,
            "fill_of_allowance": fill,
            "share_of_node_ram": fill * QDRANT_NODE_FILL_LIMIT,
        })
    if node_gb is not None:
        chosen = next(o for o in options
                      if abs(o["node_ram_gb"] - node_gb) <= 1e-9)
    else:
        usable_options = [o for o in options if o["nodes"] > 0]
        if usable_options:
            chosen = min(usable_options,
                         key=lambda o: (o["total_ram_gb"], o["nodes"],
                                        o["node_ram_gb"]))
        else:
            chosen = dict(options[0])
    return {
        "options": options,
        "node_ram_gb_forced": node_gb is not None,
        "nodes": chosen["nodes"],
        "node_ram_gb": chosen["node_ram_gb"],
        "usable_gb_per_node": chosen["usable_gb_per_node"],
        "total_ram_gb": chosen["total_ram_gb"],
        "fill_of_allowance": chosen["fill_of_allowance"],
        "tight_fit": chosen["fill_of_allowance"] >= QDRANT_TIGHT_FIT_WARN_FRACTION,
    }


def sensitivity_rates(agent_searches_per_s: float,
                      base=SENSITIVITY_AGENT_RATES) -> tuple:
    """The agent-mode rates the sensitivity table shows, in ascending order.

    The fixed rates always appear. The deployment's own agent-mode rate is
    added when it is not already one of them, so that the table always contains
    the row that matches the headline machine count. Without this the scale
    tier printed a table whose worst row was 25 agent-mode searches/s next to a
    headline sized for 100.
    """
    rates = list(base)
    if (math.isfinite(agent_searches_per_s) and agent_searches_per_s > 0
            and not any(abs(agent_searches_per_s - r) <= 1e-9 for r in rates)):
        rates.append(agent_searches_per_s)
    return tuple(sorted(rates))


def agent_sensitivity(adds_per_s: float, plain_per_s: float,
                      agent_rates=SENSITIVITY_AGENT_RATES,
                      this_run_rate=None) -> list:
    """API server count against the agent-mode search rate.

    Adds and plain searches are held fixed; only the agent-mode rate varies.
    This is the table that shows why the agent-mode quota is a hardware
    decision and not a product detail.

    this_run_rate is the deployment's own agent-mode rate. The row at that rate
    is flagged, because two rates a rounded label cannot tell apart - 2.0 and
    2.04 both print as "2.0" - are still two different sizings, and the reader
    has to be able to see which one is the traffic mix they asked about.
    """
    rows = []
    for rate in agent_rates:
        vector_searches = (plain_per_s * PLAIN_VECTOR_SEARCHES
                           + rate * AGENT_VECTOR_SEARCHES)
        work = vector_searches + adds_per_s * ADD_COST_IN_PLAIN_SEARCH_EQUIVALENTS
        rows.append({
            "agent_searches_per_s": rate,
            "total_ops_per_s": adds_per_s + plain_per_s + rate,
            "vector_searches_per_s": vector_searches,
            "api_work_per_s": work,
            "api_servers": api_servers_for_work(work),
            "llm_calls_per_s_low": rate * AGENT_LLM_CALLS_LOW,
            "llm_calls_per_s_high": rate * AGENT_LLM_CALLS_HIGH,
            "is_this_run": (this_run_rate is not None
                            and abs(rate - this_run_rate) <= 1e-9),
        })
    return rows


def size_deployment(ops_per_s: float,
                    mix: TrafficMix | None = None,
                    retention_days: float = DEFAULT_RETENTION_DAYS,
                    dims: int = DEFAULT_VECTOR_DIMS,
                    bytes_per_value: float = DEFAULT_BYTES_PER_VALUE,
                    node_gb=None,
                    node_gb_source: str = NODE_GB_SOURCE_CLI,
                    run_name: str = "custom",
                    sized_from: str | None = None) -> dict:
    """Size one deployment. Returns a plain dictionary; prints nothing.

    ops_per_s       design peak, operations per second
    mix             how 100 operations split between adds, plain and agent-mode
    retention_days  how long an episode is kept before deletion
    dims            vector dimensions (numbers per vector)
    bytes_per_value bytes stored per number (1 means int8 quantized)
    node_gb         RAM of one vector-store machine in GB, or None to let the
                    program choose the size that buys the least total RAM
    node_gb_source  how the report should name whatever set node_gb, so that a
                    web page does not tell its reader they typed a flag
    run_name        what to call this run in the result - a tier name, or
                    "custom" or "web". It is not one of the four provenance
                    labels; the result dictionary carries no provenance.
    sized_from      what supplied the design peak and the mix, when they were
                    not given directly - SIZED_FROM_POPULATION for a run sized
                    from a caller population. None means they were given as
                    they stand, and the report reads exactly as it always has.
    """
    mix = mix or TrafficMix()
    mix.validate()

    # One name per input, taken from the label on the web form, so that a box
    # is never called one thing when it is empty and another when it is too
    # large.
    for field, value, limit, unit in (
            ("the design peak", ops_per_s, MAX_OPS_PER_S, "operations/s"),
            ("retention", retention_days, MAX_RETENTION_DAYS, "days"),
            ("vector dimensions", dims, MAX_VECTOR_DIMS, "dimensions"),
            ("bytes per number", bytes_per_value, MAX_BYTES_PER_VALUE,
             "bytes")):
        if not math.isfinite(value):
            raise SizingError(
                f"{field} is {as_given(value)} {unit}, but every input must be "
                "a finite number - infinity and not-a-number are not "
                "deployments to size")
        if value > limit:
            raise SizingError(
                f"{field} is {as_given(value)} {unit}, which is larger than "
                "this calculator will size - the most it accepts is "
                f"{as_given(limit)} {unit}")
    if ops_per_s <= 0:
        raise SizingError(
            f"the design peak is {as_given(ops_per_s)} operations/s, but it "
            "must be greater than zero - there is no deployment to size at no "
            "traffic")
    if retention_days < 0:
        raise SizingError(
            f"retention is {as_given(retention_days)} days, but it cannot be "
            "negative")
    # A vector holds a whole number of numbers. Cutting 1024.7 down to 1024
    # would size the deployment for a shape nobody asked for, and the report
    # itself warns that the dimension count must be fixed before the first
    # episode is ingested.
    if not float(dims).is_integer():
        raise SizingError(
            f"vector dimensions is {as_given(dims)}, but it must be a whole "
            "number of dimensions - a vector cannot hold part of a number")
    dims = int(dims)
    if dims <= 0:
        raise SizingError(
            f"vector dimensions is {as_given(dims)}, but it must be a positive "
            "whole number")
    if bytes_per_value <= 0:
        raise SizingError(
            f"bytes per number is {as_given(bytes_per_value)}, but it must be "
            "greater than zero")
    if node_gb is not None:
        if not math.isfinite(node_gb):
            raise SizingError(
                f"RAM per vector-store machine is {as_given(node_gb)} GB, but "
                "every input must be a finite number - infinity and "
                "not-a-number are not machines to order")
        if node_gb <= 0:
            raise SizingError(
                f"RAM per vector-store machine is {as_given(node_gb)} GB, but "
                "it must be greater than zero")
        if node_gb > MAX_NODE_GB:
            raise SizingError(
                f"RAM per vector-store machine is {as_given(node_gb)} GB, "
                "which is larger than this calculator will size - the most it "
                f"accepts is {as_given(MAX_NODE_GB)} GB")
        # A whole number of GB stays a whole number, so that the reports and
        # the JSON keys read "512 GB" and not "512.0 GB".
        if float(node_gb).is_integer():
            node_gb = int(node_gb)

    # ---- request rates by type ---------------------------------------------
    adds = ops_per_s * mix.add / MIX_TOTAL
    plains = ops_per_s * mix.plain / MIX_TOTAL
    agents = ops_per_s * mix.agent / MIX_TOTAL

    # ---- demand -------------------------------------------------------------
    embeds = (adds * ADD_EMBEDS + plains * PLAIN_EMBEDS + agents * AGENT_EMBEDS)
    embeds_with_fix = (adds * ADD_EMBEDS + plains * PLAIN_EMBEDS_WITH_TYPES_FIX
                       + agents * AGENT_EMBEDS)
    vector_searches = (plains * PLAIN_VECTOR_SEARCHES
                       + agents * AGENT_VECTOR_SEARCHES)
    vector_writes = adds * ADD_VECTOR_WRITES
    pg_statements = (adds * ADD_POSTGRES_STATEMENTS
                     + plains * PLAIN_POSTGRES_STATEMENTS
                     + agents * AGENT_POSTGRES_STATEMENTS)
    llm_low = agents * AGENT_LLM_CALLS_LOW
    llm_high = agents * AGENT_LLM_CALLS_HIGH
    llm_planning = agents * AGENT_LLM_CALLS_PLANNING

    demand = {
        "adds_per_s": adds,
        "plain_searches_per_s": plains,
        "agent_searches_per_s": agents,
        "embeds_per_s": embeds,
        "embeds_per_s_with_types_fix": embeds_with_fix,
        "vector_searches_per_s": vector_searches,
        "vector_writes_per_s": vector_writes,
        "postgres_statements_per_s": pg_statements,
        "agent_llm_calls_per_s_low": llm_low,
        "agent_llm_calls_per_s_high": llm_high,
        "agent_llm_calls_per_s_planning": llm_planning,
    }

    # ---- API servers --------------------------------------------------------
    api_work = vector_searches + adds * ADD_COST_IN_PLAIN_SEARCH_EQUIVALENTS
    api_servers = api_servers_for_work(api_work)

    # ---- embedding GPU cards ------------------------------------------------
    # Sized on the demand WITHOUT the types fix: the larger, safer figure.
    usable_low = EMBED_CARD_REQUESTS_PER_S_LOW * GPU_UTILIZATION_CEILING
    usable_high = EMBED_CARD_REQUESTS_PER_S_HIGH * GPU_UTILIZATION_CEILING
    cards_at_low_rate = ceil_up(embeds, usable_low)     # pessimistic card rate
    embed_cards_low = embed_gpu_cards_for_demand(
        embeds, EMBED_CARD_REQUESTS_PER_S_HIGH)
    embed_cards_high = embed_gpu_cards_for_demand(
        embeds, EMBED_CARD_REQUESTS_PER_S_LOW)

    # ---- agent-model GPU cards ---------------------------------------------
    agent_cards_needed = ceil_up(llm_planning, AGENT_LLM_CALLS_PER_S_PER_CARD)
    agent_cards = agent_gpu_cards_for_demand(llm_planning)

    # ---- storage ------------------------------------------------------------
    episodes = adds * SECONDS_PER_DAY * retention_days
    episodes_per_day = adds * SECONDS_PER_DAY
    episodes_per_year = episodes_per_day * 365

    # Bytes per stored episode. Each figure below is a count of episodes
    # multiplied by one of these, so the retained figures and the
    # one-year-with-no-deletion figures are worked out the same way and cannot
    # drift apart.
    hot_ram_bytes_per_episode = (dims * bytes_per_value
                                 * QDRANT_INDEX_OVERHEAD_FACTOR)
    nvme_bytes_per_episode = ((dims * ORIGINAL_VECTOR_BYTES_PER_VALUE
                               + QDRANT_DISK_PAYLOAD_BYTES_PER_EPISODE)
                              * QDRANT_DISK_OVERHEAD_FACTOR)
    pg_bytes_per_episode_low = ((EPISODE_TEXT_BYTES_LOW
                                 + POSTGRES_ROW_OVERHEAD_BYTES
                                 + POSTGRES_INDEX_BYTES_PER_EPISODE)
                                * POSTGRES_BLOAT_FACTOR)
    pg_bytes_per_episode_high = ((EPISODE_TEXT_BYTES_HIGH
                                  + POSTGRES_ROW_OVERHEAD_BYTES
                                  + POSTGRES_INDEX_BYTES_PER_EPISODE)
                                 * POSTGRES_BLOAT_FACTOR)

    hot_ram_bytes = episodes * hot_ram_bytes_per_episode
    nvme_bytes = episodes * nvme_bytes_per_episode
    pg_bytes_low = episodes * pg_bytes_per_episode_low
    pg_bytes_high = episodes * pg_bytes_per_episode_high
    # Vector work has to run somewhere, so it costs a machine even when the
    # stored bytes come to nothing.
    does_vector_work = vector_searches > 0 or vector_writes > 0
    qdrant = qdrant_node_plan(hot_ram_bytes, node_gb,
                              least_nodes=1 if does_vector_work else 0)

    # One year with nothing ever deleted. This is simply a year of adds, so it
    # is worked out from episodes_per_year and does not move with the retention
    # setting - at retention 0 it is still a full year of stored episodes.
    year_hot_ram_bytes = episodes_per_year * hot_ram_bytes_per_episode
    year_nvme_bytes = episodes_per_year * nvme_bytes_per_episode
    year_pg_bytes_low = episodes_per_year * pg_bytes_per_episode_low
    year_pg_bytes_high = episodes_per_year * pg_bytes_per_episode_high

    storage = {
        "retention_days": retention_days,
        "vector_dims": dims,
        "bytes_per_value": bytes_per_value,
        "episodes_per_day": episodes_per_day,
        "episodes_per_year": episodes_per_year,
        "episodes_retained": episodes,
        "hot_vector_ram_bytes": hot_ram_bytes,
        "hot_vector_ram_gb": hot_ram_bytes / BYTES_PER_GB,
        "qdrant_nvme_bytes": nvme_bytes,
        "qdrant_nvme_gb": nvme_bytes / BYTES_PER_GB,
        "postgres_bytes_low": pg_bytes_low,
        "postgres_bytes_high": pg_bytes_high,
        "postgres_gb_low": pg_bytes_low / BYTES_PER_GB,
        "postgres_gb_high": pg_bytes_high / BYTES_PER_GB,
        # One year with no deletion at all - the figure that makes retention a
        # requirement rather than an option.
        "unbounded_year_hot_vector_ram_bytes": year_hot_ram_bytes,
        "unbounded_year_hot_vector_ram_gb": year_hot_ram_bytes / BYTES_PER_GB,
        "unbounded_year_qdrant_nvme_bytes": year_nvme_bytes,
        "unbounded_year_qdrant_nvme_gb": year_nvme_bytes / BYTES_PER_GB,
        "unbounded_year_postgres_bytes_low": year_pg_bytes_low,
        "unbounded_year_postgres_bytes_high": year_pg_bytes_high,
        "unbounded_year_postgres_gb_low": year_pg_bytes_low / BYTES_PER_GB,
        "unbounded_year_postgres_gb_high": year_pg_bytes_high / BYTES_PER_GB,
    }

    # ---- PostgreSQL connections --------------------------------------------
    core_connections = (api_servers * API_WORKERS_PER_SERVER
                        * POSTGRES_CONNECTIONS_PER_WORKER)
    gateway_connections = api_servers * GATEWAY_CONNECTIONS_PER_API_SERVER
    total_connections = core_connections + gateway_connections
    postgres = {
        "servers": POSTGRES_SERVERS_PER_TIER,
        "statements_per_s": pg_statements,
        "workers_per_api_server": API_WORKERS_PER_SERVER,
        "connections_per_worker": POSTGRES_CONNECTIONS_PER_WORKER,
        "core_connections": core_connections,
        "gateway_connections": gateway_connections,
        "total_connections": total_connections,
        "max_connections_required": total_connections,
        "chart_default_max_connections": POSTGRES_CHART_DEFAULT_MAX_CONNECTIONS,
        "proven_max_connections": POSTGRES_PROVEN_MAX_CONNECTIONS,
        "exceeds_chart_default": total_connections >
        POSTGRES_CHART_DEFAULT_MAX_CONNECTIONS,
        "exceeds_proven_setting": total_connections >
        POSTGRES_PROVEN_MAX_CONNECTIONS,
        "needs_connection_pooler": total_connections >
        POSTGRES_PROVEN_MAX_CONNECTIONS,
    }

    # ---- network ------------------------------------------------------------
    ns_bytes_per_s = (
        adds * (NS_ADD_REQUEST_BYTES + NS_ADD_RESPONSE_BYTES)
        + plains * (NS_SEARCH_REQUEST_BYTES
                    + PLAIN_SEARCH_EPISODES_RETURNED * NS_RESPONSE_BYTES_PER_EPISODE)
        + agents * (NS_SEARCH_REQUEST_BYTES + NS_AGENT_ANSWER_BYTES
                    + AGENT_SEARCH_EPISODES_RETURNED * NS_RESPONSE_BYTES_PER_EPISODE)
    ) * NETWORK_PROTOCOL_OVERHEAD_FACTOR

    embed_call_bytes = (EMBED_REQUEST_BYTES
                        + dims * ORIGINAL_VECTOR_BYTES_PER_VALUE
                        + EMBED_RESPONSE_ENVELOPE_BYTES)
    vector_search_bytes = (dims * ORIGINAL_VECTOR_BYTES_PER_VALUE
                           + QDRANT_SEARCH_REQUEST_ENVELOPE_BYTES
                           + QDRANT_CANDIDATES_PER_SEARCH
                           * QDRANT_BYTES_PER_CANDIDATE)
    vector_write_bytes = (dims * ORIGINAL_VECTOR_BYTES_PER_VALUE
                          + QDRANT_UPSERT_ENVELOPE_BYTES
                          + QDRANT_UPSERT_RESPONSE_BYTES)
    llm_call_bytes = LLM_CALL_REQUEST_BYTES + LLM_CALL_RESPONSE_BYTES

    ew_bytes_per_s = (
        embeds * embed_call_bytes
        + vector_searches * vector_search_bytes
        + vector_writes * vector_write_bytes
        + pg_statements * POSTGRES_BYTES_PER_STATEMENT
        + llm_planning * llm_call_bytes
    ) * NETWORK_PROTOCOL_OVERHEAD_FACTOR

    def to_mbps(bytes_per_s: float) -> float:
        return bytes_per_s * BITS_PER_BYTE / BITS_PER_MBIT

    busiest_mbps = max(to_mbps(ns_bytes_per_s), to_mbps(ew_bytes_per_s))
    network = {
        "north_south_bytes_per_s": ns_bytes_per_s,
        "east_west_bytes_per_s": ew_bytes_per_s,
        "north_south_mbps": to_mbps(ns_bytes_per_s),
        "east_west_mbps": to_mbps(ew_bytes_per_s),
        "embed_bytes_per_call": embed_call_bytes,
        "vector_search_bytes_per_call": vector_search_bytes,
        "vector_write_bytes_per_call": vector_write_bytes,
        "llm_bytes_per_call": llm_call_bytes,
        "busiest_link_mbps": busiest_mbps,
        # Headroom is measured against the busiest of the two directions, which
        # is what the report says it is. East-west is the busier one with
        # today's constants, but nothing in the model forces that to stay true.
        # None, never a floating-point infinity: json.dumps writes an infinity
        # as the bare token Infinity, which most JSON parsers reject.
        "headroom_on_10gbe": (10000.0 / busiest_mbps
                              if busiest_mbps > 0 else None),
    }

    # ---- users --------------------------------------------------------------
    # ops_per_s is always greater than zero here, so none of these can divide
    # by zero and none of them can be infinite.
    # The two "_heavy" and "_sustained" entries are reported and nothing else.
    # No machine count reads them: they are here so the report can show what a
    # heavy session and an average-load agent would mean for the same capacity.
    users = {
        "human_sessions_low": ops_per_s / HUMAN_SESSION_OPS_PER_S_HIGH,
        "human_sessions_high": ops_per_s / HUMAN_SESSION_OPS_PER_S_LOW,
        "human_sessions_heavy": ops_per_s / HUMAN_SESSION_OPS_PER_S_HEAVY,
        "automated_client_sessions": ops_per_s / AUTOMATED_CLIENT_OPS_PER_S,
        "automated_client_sessions_sustained": (
            ops_per_s / AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED),
        "human_ops_per_s_low": HUMAN_SESSION_OPS_PER_S_LOW,
        "human_ops_per_s_high": HUMAN_SESSION_OPS_PER_S_HIGH,
        "human_ops_per_s_heavy": HUMAN_SESSION_OPS_PER_S_HEAVY,
        "automated_client_ops_per_s": AUTOMATED_CLIENT_OPS_PER_S,
        "automated_client_ops_per_s_sustained": (
            AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED),
    }

    machines = {
        "api_servers": api_servers,
        "api_server_spec": f"{API_SERVER_VCPU} vCPU, {API_SERVER_RAM_GB} GB",
        "api_work_per_s": api_work,
        "api_usable_searches_per_server": api_usable_searches_per_server(),
        "postgres_servers": POSTGRES_SERVERS_PER_TIER,
        "qdrant_servers": qdrant["nodes"],
        "qdrant_node_ram_gb": qdrant["node_ram_gb"],
        "qdrant_usable_gb_per_node": qdrant["usable_gb_per_node"],
        "qdrant_total_ram_gb": qdrant["total_ram_gb"],
        "qdrant_fill_of_allowance": qdrant["fill_of_allowance"],
        "qdrant_node_ram_gb_forced": qdrant["node_ram_gb_forced"],
        "qdrant_tight_fit": qdrant["tight_fit"],
        "qdrant_options": qdrant["options"],
        "embed_gpu_cards_low": embed_cards_low,
        "embed_gpu_cards_high": embed_cards_high,
        "embed_gpu_spare": GPU_SPARE_CARDS if cards_at_low_rate else 0,
        "embed_usable_per_card_low": usable_low,
        "embed_usable_per_card_high": usable_high,
        "agent_gpu_cards": agent_cards,
        "agent_gpu_spare": GPU_SPARE_CARDS if agent_cards_needed else 0,
        "total_cpu_servers": (api_servers + POSTGRES_SERVERS_PER_TIER
                              + qdrant["nodes"]),
    }

    return {
        "run_name": run_name,
        "inputs": {
            "ops_per_s": ops_per_s,
            "mix": mix.as_dict(),
            "retention_days": retention_days,
            "dims": dims,
            "bytes_per_value": bytes_per_value,
            "node_gb": node_gb,
            "node_gb_source": node_gb_source,
            "sized_from": sized_from,
        },
        "demand": demand,
        "machines": machines,
        "storage": storage,
        "postgres": postgres,
        "network": network,
        "users": users,
        "sensitivity": agent_sensitivity(adds, plains,
                                         sensitivity_rates(agents),
                                         this_run_rate=agents),
    }


@dataclass(frozen=True)
class ConversionNames:
    """What one caller kind's four inputs are called, in one door's words.

    The command line calls them --humans, --human-users, --human-peak-share
    and --human-sessions-per-active-user; the web form calls them boxes with
    names on them. The refusals below are written once and take these names,
    so the two doors give the same message about the same mistake and neither
    ever tells a reader to type something the other door offers.
    """

    sessions: str
    users: str
    share: str
    per_user: str


CLI_HUMAN_NAMES = ConversionNames(
    sessions="--humans", users="--human-users", share="--human-peak-share",
    per_user="--human-sessions-per-active-user")
CLI_AUTOMATED_NAMES = ConversionNames(
    sessions="--automated", users="--automated-users",
    share="--automated-peak-share",
    per_user="--automated-sessions-per-active-user")


@dataclass(frozen=True)
class ConversionDefaults:
    """One caller kind's example conversion figures, and why they are those.

    share       how many of every 100 users are active at the busiest moment
    per_user    how many sessions one of those active users holds
    column      what this caller kind is called in the table's label column
    why_share   why the share default is the number it is, in a few words
    why_per_user  the same for the sessions-per-active-user default
    phrase_share  how the warning note names this kind's share figure
    phrase_per_user  how the warning note names this kind's sessions figure

    Every one of these figures is an EXAMPLE. It exists so a bare user count
    always gets an answer, it is labelled "example default" wherever it is
    printed, and it is meant to be replaced with a metered number.
    """

    share: float
    per_user: float
    column: str
    why_share: str
    why_per_user: str
    phrase_share: str
    phrase_per_user: str


HUMAN_CONVERSION_DEFAULTS = ConversionDefaults(
    share=DEFAULT_HUMAN_PEAK_ACTIVE_PER_100,
    per_user=DEFAULT_HUMAN_SESSIONS_PER_ACTIVE_USER,
    column=CONVERSION_COLUMN_HUMAN,
    why_share=WHY_HUMAN_SHARE_DEFAULT,
    why_per_user=WHY_HUMAN_PER_USER_DEFAULT,
    phrase_share="the share of people active at the busiest moment",
    phrase_per_user="the sessions one active person holds")
AUTOMATED_CONVERSION_DEFAULTS = ConversionDefaults(
    share=DEFAULT_AUTOMATED_PEAK_ACTIVE_PER_100,
    per_user=DEFAULT_AUTOMATED_SESSIONS_PER_ACTIVE_USER,
    column=CONVERSION_COLUMN_AUTOMATED,
    why_share=WHY_AUTOMATED_IDENTITY,
    why_per_user=WHY_AUTOMATED_IDENTITY,
    phrase_share=("the share of automated-client users active at the busiest "
                  "moment"),
    phrase_per_user="the automated client sessions one active user holds")

# The users subcommand has always insisted on being told how many human
# callers there are, and it still does. There are two ways to answer now, and
# the refusal names both rather than only the older one.
HUMAN_SIDE_UNANSWERED_MESSAGE = (
    "the users subcommand needs to know how many human callers there are. "
    "Give --humans for the concurrent human chat sessions - people typing at "
    "the same moment - or give --human-users to work those sessions out from "
    "the size of the user base. If there are no human callers at all, pass "
    "--humans 0.")


def start_sentence(name: str) -> str:
    """A name at the start of a sentence.

    A flag has no capital to give, so --human-peak-share is left alone; a box
    name is a phrase, and "the sessions per active person box" has to open a
    sentence as "The sessions per active person box".
    """
    return name[:1].upper() + name[1:]


def and_list(names) -> tuple:
    """("a and b", "are") or ("a", "is") - a list, and the verb that fits it.

    Three or more are separated by commas, because four phrases joined by
    "and" read as one long phrase rather than as four things.
    """
    names = list(names)
    if len(names) == 1:
        return names[0], "is"
    if len(names) == 2:
        return " and ".join(names), "are"
    return ", ".join(names[:-1]) + " and " + names[-1], "are"


def defaulted_figures_note(conversion: dict | None) -> str:
    """The warning printed with the conversion table when a default was used.

    This used to be a refusal: a user count that arrived without both
    conversion figures was rejected with exit code 2. The figures now have
    example defaults, so the calculator always answers - and the refusal is
    demoted to this note rather than dropped, because everything it said is
    still true. It names which figures were defaulted, says they are examples,
    says how far the share active can be wrong, and says what to replace them
    with.

    Returns "" when every figure in the table was supplied by the reader, so a
    reader who gave their own numbers is not warned about numbers they did not
    use.
    """
    if conversion is None:
        return ""
    defaulted, automated_defaulted = [], False
    for key, defaults in (("human", HUMAN_CONVERSION_DEFAULTS),
                          ("automated", AUTOMATED_CONVERSION_DEFAULTS)):
        one = conversion.get(key)
        if one is None:
            continue
        for flag, phrase in (
                ("peak_active_per_100_is_default", defaults.phrase_share),
                ("sessions_per_active_user_is_default",
                 defaults.phrase_per_user)):
            if one.get(flag):
                defaulted.append(phrase)
                automated_defaulted |= key == "automated"
    if not defaulted:
        return ""
    joined, verb = and_list(defaulted)
    one_only = verb == "is"
    thing = "a figure you gave" if one_only else "figures you gave"
    them = "it" if one_only else "them"
    marked = "it is" if one_only else "they are"
    note = (
        f"EXAMPLE DEFAULTS WERE USED HERE. {start_sentence(joined)} {verb} "
        f"not {thing}: this calculator chose {them} so that a user count "
        f"always gets an answer, and {marked} marked "
        f'"{CONVERSION_MARK_DEFAULT}" in the table above. A user count is '
        "not a count of concurrent sessions, and this calculator cannot know "
        "the difference between them for your service. The share of users "
        "active at the busiest moment is the figure that hurts: "
        f"{DEFAULT_HUMAN_PEAK_ACTIVE_PER_100:g} per 100 users is a convention "
        "rather than a measurement, the plausible range is "
        f"{HUMAN_PEAK_ACTIVE_PER_100_LOW:g} to "
        f"{HUMAN_PEAK_ACTIVE_PER_100_HIGH:g} per 100, so it can be wrong by a "
        "factor of two either way - and every machine count below moves with "
        "it. Replace these examples with your own numbers: meter the "
        "concurrent sessions the deployment actually holds - callers sending "
        "requests at the same moment - from the first day of running, and "
        "give those session counts instead of a user count.")
    if automated_defaulted:
        note += " " + AUTOMATED_IDENTITY_SENTENCE
    return note


def conversion_without_users_message(names: ConversionNames, given) -> str:
    """Refuse a conversion figure that arrived with no user count to convert."""
    joined, verb = and_list(given)
    return (
        f"{joined} {verb} given, but {names.users} is missing. A share of the "
        "users active at the busiest moment and a count of sessions per "
        "active user are the two figures that turn a user count into "
        f"concurrent sessions. On their own they say nothing. Give "
        f"{names.users} as well, or drop them and give {names.sessions} if "
        "you already know how many sessions run at the same moment.")


def both_ways_message(names: ConversionNames) -> str:
    """Refuse a caller kind counted twice, once as sessions and once as users."""
    return (
        f"{names.sessions} and {names.users} are both given, and they answer "
        "the same question two different ways. "
        f"{start_sentence(names.sessions)} is how many sessions run at the "
        f"same moment. {start_sentence(names.users)} is how many users "
        "there are, which this calculator turns into concurrent sessions "
        f"with {names.share} and {names.per_user}. Give one or the other, "
        "not both.")


@dataclass(frozen=True)
class UsersToSessions:
    """One caller kind's user count, and the two figures that convert it.

    users                     how many users there are of this kind
    peak_active_per_100       how many of every 100 of them are using the
                              service at the busiest moment
    sessions_per_active_user  how many sessions one of those active users
                              holds at that moment

    Multiplying the three gives the concurrent sessions the rest of the model
    works in.

    The last two flags say whether each conversion figure was supplied by the
    reader or filled in from this program's example defaults. Nothing about
    the arithmetic reads them; they exist so that the report can mark every
    figure as supplied or as an example default, and so that the warning note
    can name the ones the reader did not choose.
    """

    users: float
    peak_active_per_100: float
    sessions_per_active_user: float
    peak_active_per_100_is_default: bool = False
    sessions_per_active_user_is_default: bool = False

    def sessions(self) -> float:
        return (self.users * self.peak_active_per_100
                / MAX_PEAK_ACTIVE_PER_100 * self.sessions_per_active_user)

    def validate(self, names: ConversionNames) -> None:
        """Refuse a figure by the name the reader typed it under."""
        for name, value in ((names.users, self.users),
                            (names.share, self.peak_active_per_100),
                            (names.per_user, self.sessions_per_active_user)):
            if not math.isfinite(value):
                raise SizingError(
                    f"{name} is {value:g}, but it must be a finite number")
            if value < 0:
                raise SizingError(
                    f"{name} is {as_given(value)}, but it cannot be negative")
        if self.peak_active_per_100 > MAX_PEAK_ACTIVE_PER_100:
            raise SizingError(
                f"{names.share} is {as_given(self.peak_active_per_100)}, but "
                "it is a share of the user base written per 100 users, so it "
                f"cannot be more than {MAX_PEAK_ACTIVE_PER_100:g}")
        if self.users > MAX_USERS:
            raise SizingError(
                f"{names.users} is {as_given(self.users)}, but the most this "
                f"calculator accepts is {as_given(MAX_USERS)}")
        if self.sessions_per_active_user > MAX_SESSIONS_PER_ACTIVE_USER:
            raise SizingError(
                f"{names.per_user} is "
                f"{as_given(self.sessions_per_active_user)}, but the most "
                f"this calculator accepts is "
                f"{as_given(MAX_SESSIONS_PER_ACTIVE_USER)}")

    def as_dict(self) -> dict:
        return {
            "users": self.users,
            "peak_active_per_100": self.peak_active_per_100,
            "peak_active_per_100_is_default":
                self.peak_active_per_100_is_default,
            "sessions_per_active_user": self.sessions_per_active_user,
            "sessions_per_active_user_is_default":
                self.sessions_per_active_user_is_default,
            "concurrent_sessions": self.sessions(),
        }


def sessions_for_caller_kind(names: ConversionNames,
                             defaults: ConversionDefaults, sessions, users,
                             share, per_user) -> tuple:
    """One caller kind's concurrent sessions, and how they were arrived at.

    Every argument after ``defaults`` is a number or None, where None means
    "not given". Returns (concurrent sessions or None, the UsersToSessions
    behind them or None).

    A reader who already knows their concurrent sessions gives them and
    nothing else changes: this returns what they typed and no conversion. A
    reader who knows only a user count gives that count, and either conversion
    figure they leave out is filled in from ``defaults`` - an example value,
    marked as one in the report and warned about in a note under the table.
    A conversion figure with no user count to convert is still refused, and so
    is a caller kind counted both ways at once.
    """
    conversion_given = [name for name, value in ((names.share, share),
                                                 (names.per_user, per_user))
                        if value is not None]
    if users is None:
        if conversion_given:
            raise SizingError(
                conversion_without_users_message(names, conversion_given))
        return sessions, None
    if sessions is not None:
        raise SizingError(both_ways_message(names))
    conversion = UsersToSessions(
        users,
        defaults.share if share is None else share,
        defaults.per_user if per_user is None else per_user,
        peak_active_per_100_is_default=share is None,
        sessions_per_active_user_is_default=per_user is None)
    conversion.validate(names)
    return conversion.sessions(), conversion


def users_conversion_dict(human: UsersToSessions | None,
                          automated: UsersToSessions | None):
    """Both kinds' conversions as one block, or None when neither used one.

    None is the answer for every run that names its concurrent sessions
    directly, and the report then prints no conversion table at all - so
    nothing about the older way of asking moves.
    """
    if human is None and automated is None:
        return None
    return {"human": human.as_dict() if human is not None else None,
            "automated": (automated.as_dict()
                          if automated is not None else None)}


def blend_mixes(human_ops_per_s: float, human_mix: TrafficMix,
                automated_ops_per_s: float,
                automated_mix: TrafficMix) -> TrafficMix:
    """One traffic mix for a population made of two kinds of caller.

    Each share is the two populations' shares averaged, weighted by the
    operations each population demands. A population that sends nothing carries
    no weight. With no traffic at all there is nothing to blend, so the human
    mix is returned unchanged rather than a division by zero.
    """
    total = human_ops_per_s + automated_ops_per_s
    if total <= 0:
        return human_mix

    def blended(human_share: float, automated_share: float) -> float:
        return (human_ops_per_s * human_share
                + automated_ops_per_s * automated_share) / total

    return TrafficMix(
        add=blended(human_mix.add, automated_mix.add),
        plain=blended(human_mix.plain, automated_mix.plain),
        agent=blended(human_mix.agent, automated_mix.agent))


def ops_for_population(humans: float, automated: float,
                       human_mix: TrafficMix | None = None,
                       automated_mix: TrafficMix | None = None,
                       retention_days: float = DEFAULT_RETENTION_DAYS,
                       dims: int = DEFAULT_VECTOR_DIMS,
                       bytes_per_value: float = DEFAULT_BYTES_PER_VALUE,
                       node_gb=None,
                       node_gb_source: str = NODE_GB_SOURCE_CLI,
                       run_name: str = "population",
                       human_conversion: UsersToSessions | None = None,
                       automated_conversion: UsersToSessions | None = None
                       ) -> dict:
    """Convert a population of callers into the capacity it demands.

    humans          concurrent human chat sessions - people typing at the same
                    moment, not registered accounts and not visitors in a day
    automated       concurrent automated client sessions - programs sending
                    requests in a loop at the same moment. This is a kind of
                    CALLER. It is not agent-mode search, which is a flag on
                    one request.
    human_mix       how the human sessions' operations split between adds,
                    plain searches and agent-mode searches
    automated_mix   the same for the automated client sessions

    The last two settings are the working behind those two counts, when the
    counts were not typed in but worked out from a user count. They change no
    arithmetic here - the sessions have already been multiplied out by then -
    and they exist so that the report can show the multiplication and a reader
    can argue with it. Leave them off and the report says nothing about users,
    which is what happens for a reader who gives concurrent sessions directly.

    The four settings after the mixes describe the shape of the store rather
    than the traffic - how long an episode is kept, how big a vector is and how
    much RAM one vector-store machine has - and they are passed straight
    through to size_deployment. They default to the same values the rest of the
    program defaults to, so a caller that leaves them off gets exactly what it
    got before they existed.

    Each population gets its own mix because the kind of caller and the kind of
    request are correlated: automated clients may use agent-mode search on
    nearly every call while people rarely do. Both mixes default to the model's
    own default mix, so leaving them off changes nothing.

    The blended mix is weighted at the busy end of the human rate, which is the
    rate this report tells you to plan for, and it is what sizes the
    deployment - the global default mix is not used once a population is given.
    """
    for field, value in (("concurrent human chat sessions", humans),
                         ("concurrent automated client sessions", automated)):
        if not math.isfinite(value):
            raise SizingError(
                f"the count of {field} is {value:g}, but a caller count must "
                "be a finite number")
    if humans < 0 or automated < 0:
        raise SizingError("a caller count cannot be negative")
    human_mix = human_mix or TrafficMix()
    automated_mix = automated_mix or TrafficMix()
    human_mix.validate()
    automated_mix.validate()

    human_low = humans * HUMAN_SESSION_OPS_PER_S_LOW
    human_high = humans * HUMAN_SESSION_OPS_PER_S_HIGH
    automated_ops = automated * AUTOMATED_CLIENT_OPS_PER_S
    # Reported, never sized on. These two are the headroom check and the
    # average-load check, and neither is added into low, high or the blend, so
    # neither can move a machine count.
    human_heavy = humans * HUMAN_SESSION_OPS_PER_S_HEAVY
    automated_sustained = automated * AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED
    low = human_low + automated_ops
    high = human_high + automated_ops
    blended = blend_mixes(human_high, human_mix, automated_ops, automated_mix)

    # The deployment this population needs, sized at the rate the report tells
    # you to plan for and with the population's own blended mix. A population
    # that makes no requests is not a deployment to size, and size_deployment
    # refuses a design peak of zero for that reason, so there is nothing here.
    sizing = None
    if high > 0:
        sizing = size_deployment(high, blended, retention_days, dims,
                                 bytes_per_value, node_gb,
                                 node_gb_source=node_gb_source,
                                 run_name=run_name,
                                 sized_from=SIZED_FROM_POPULATION)

    return {
        "humans": humans,
        "automated": automated,
        "conversion": users_conversion_dict(human_conversion,
                                            automated_conversion),
        "human_mix": human_mix.as_dict(),
        "automated_mix": automated_mix.as_dict(),
        "human_ops_per_s_low": human_low,
        "human_ops_per_s_high": human_high,
        "human_ops_per_s_heavy": human_heavy,
        "automated_ops_per_s": automated_ops,
        "automated_ops_per_s_sustained": automated_sustained,
        "ops_per_s_low": low,
        "ops_per_s_high": high,
        "blended_mix": blended.as_dict(),
        "tier_for_low": smallest_tier_holding(low),
        "tier_for_high": smallest_tier_holding(high),
        "sizing": sizing,
    }


def smallest_tier_holding(ops_per_s: float):
    """Name the smallest tier whose design peak covers this rate, or None."""
    for name in TIER_ORDER:
        if TIER_OPS_PER_S[name] >= ops_per_s:
            return name
    return None


# =============================================================================
# PRESENTATION
# Shared by the text report and the web page, so the two can never disagree.
# =============================================================================


def num(value: float, dp: int = 0) -> str:
    return f"{value:,.{dp}f}"


def gb(value_bytes: float) -> str:
    """Format a byte count. GB means 10^9 bytes; TB means 10^12 bytes."""
    if value_bytes >= BYTES_PER_TB:
        return (f"{value_bytes / BYTES_PER_TB:,.2f} TB "
                f"({value_bytes / BYTES_PER_GB:,.0f} GB)")
    return f"{value_bytes / BYTES_PER_GB:,.2f} GB"


def report_sections(r: dict) -> list:
    """Build every section of the report as (title, note, headers, rows)."""
    d = r["demand"]
    m = r["machines"]
    s = r["storage"]
    p = r["postgres"]
    n = r["network"]
    u = r["users"]
    i = r["inputs"]

    sections = []

    # A run sized from a caller population did not get its design peak or its
    # traffic mix from anybody's typing, so those two rows must not be labelled
    # as assumptions somebody made. They are worked out from the population,
    # and the labels say which population table to read them against.
    sized_from = i.get("sized_from")
    if sized_from:
        peak_label = (f"derived from {sized_from} - the demand it makes at "
                      "the busy end of the human rate")
        mix_label = (f"derived from {sized_from} - the two population mixes "
                     "blended, weighted by the demand each one makes")
    else:
        peak_label = "assumption (worst rate sustained 5 minutes)"
        mix_label = ("assumption - never measured; agent-mode search is a "
                     "request flag, not a kind of caller")

    sections.append({
        "title": "Inputs",
        "note": "Everything the answer depends on. Change any of these and the "
                "numbers below change. " + design_peak_note(),
        # Every input is echoed exactly as it was given. Rounding these to whole
        # numbers described a deployment the tables below were not sized for:
        # half a byte per number, which is int4 quantization, read as "0".
        "headers": ["Item", "Value", "Label"],
        "rows": [
            ["Design peak", f"{as_given(i['ops_per_s'])} operations/s",
             peak_label],
            ["Traffic mix per 100 operations",
             TrafficMix(**i["mix"]).as_words(), mix_label],
            ["Retention", f"{as_given(i['retention_days'])} days",
             "assumption - undecided, placeholder"],
            ["Vector dimensions", as_given(i["dims"]),
             "assumption - fix before first ingest"],
            ["Bytes stored per number", as_given(i["bytes_per_value"]),
             "assumption - 1 means int8 quantized"],
            ["RAM per vector-store machine",
             (f"{as_given(i['node_gb'])} GB, forced"
              if i.get("node_gb") else "chosen automatically"),
             ("assumption - set by hand" if i.get("node_gb") else
              "assumption - the automatic choice buys the least total RAM")],
        ],
    })

    sections.append({
        "title": "Demand per second",
        "note": "From the fan-out counts read from the code on 30 Aug 2026. "
                "All derived.",
        "headers": ["Item", "Per second", "Label"],
        "rows": [
            ["Adds", num(d["adds_per_s"], 1), "derived"],
            ["Plain searches", num(d["plain_searches_per_s"], 1), "derived"],
            ["Agent-mode searches", num(d["agent_searches_per_s"], 1), "derived"],
            ["Embedding calls (today)", num(d["embeds_per_s"], 1), "derived"],
            ["Embedding calls (with the types fix)",
             num(d["embeds_per_s_with_types_fix"], 1), "derived"],
            ["Vector searches", num(d["vector_searches_per_s"], 1), "derived"],
            ["Vector writes", num(d["vector_writes_per_s"], 1), "derived"],
            ["PostgreSQL statements", num(d["postgres_statements_per_s"], 1),
             "derived"],
            ["Agent language-model calls",
             (f"{num(d['agent_llm_calls_per_s_low'], 1)} to "
              f"{num(d['agent_llm_calls_per_s_high'], 1)} "
              f"(planning on {num(d['agent_llm_calls_per_s_planning'], 1)})"),
             "estimate - 1 to 2 calls per agent search"],
        ],
    })

    embed_cards = (f"{m['embed_gpu_cards_low']}"
                   if m["embed_gpu_cards_low"] == m["embed_gpu_cards_high"]
                   else f"{m['embed_gpu_cards_low']} to {m['embed_gpu_cards_high']}")
    sections.append({
        "title": "Machines",
        "note": "Counts always round up to a whole machine. GPU counts include "
                "one spare card.",
        "headers": ["Machine", "Count", "Spec each", "Basis"],
        "rows": [
            ["API server (gateway + MemMachine core)", num(m["api_servers"]),
             m["api_server_spec"],
             "derived from the 180/s anchor measured 30 Aug 2026"],
            # Only the API server has a measured machine class. Naming its
            # vCPU and RAM here as well would state a spec for two machines
            # nobody has sized.
            ["PostgreSQL server", num(m["postgres_servers"]),
             "NVMe disk; vCPU and RAM undecided",
             "assumption - never benchmarked at this statement rate"],
            ["Qdrant server", num(m["qdrant_servers"]),
             (f"{num(m['qdrant_node_ram_gb'])} GB RAM, NVMe disk; "
              "vCPU undecided"),
             "derived from retention and vector size"],
            ["Embedding GPU card", embed_cards,
             "H100-class (includes 1 spare)"
             if m["embed_gpu_cards_high"] else "not needed at this rate",
             "estimate - card rate never benchmarked"],
            ["Agent-model GPU card", num(m["agent_gpu_cards"]),
             "8B-class model (includes 1 spare)"
             if m["agent_gpu_cards"] else "not needed at this mix (no "
             "agent-mode traffic)",
             "estimate - card rate never benchmarked"],
            ["Total ordinary servers (not GPU)", num(m["total_cpu_servers"]),
             "-", "derived"],
        ],
    })

    sections.append({
        "title": "How the API server count was reached",
        "note": "Work is counted in plain-search-equivalents. One add is counted "
                "as one plain-search-equivalent, which is an estimate that "
                "rounds up: the 30 Aug run measured search only.",
        "headers": ["Step", "Value", "Label"],
        "rows": [
            ["Vector searches/s", num(d["vector_searches_per_s"], 1), "derived"],
            ["Plus adds/s counted as search-equivalents",
             num(d["adds_per_s"], 1), "estimate"],
            ["Work per second", num(m["api_work_per_s"], 1), "derived"],
            ["Measured capacity per server",
             f"{num(API_SEARCHES_PER_S_PER_SERVER, 0)} searches/s",
             "measured 30 Aug 2026, 8 workers, 128 concurrent"],
            ["Utilization ceiling",
             f"{num(API_UTILIZATION_CEILING * 100, 0)}%", "assumption"],
            ["Planned capacity per server",
             f"{num(m['api_usable_searches_per_server'], 0)} searches/s",
             "derived"],
            ["Servers, rounded up", num(m["api_servers"]), "derived"],
        ],
    })

    sections.append({
        "title": "Storage",
        "note": f"At {as_given(i['retention_days'])} days of retention, "
                f"{as_given(i['dims'])}-number vectors, "
                f"{as_given(i['bytes_per_value'])} byte(s) per number. "
                "GB means 10^9 bytes and TB means 10^12 bytes throughout.",
        "headers": ["Item", "Value", "Label"],
        "rows": [
            ["Episodes stored per day", num(s["episodes_per_day"], 0), "derived"],
            ["Episodes stored per year", num(s["episodes_per_year"], 0), "derived"],
            ["Episodes held at this retention", num(s["episodes_retained"], 0),
             "derived"],
            ["Hot vector RAM in Qdrant", gb(s["hot_vector_ram_bytes"]),
             "derived (includes 1.5x index overhead)"],
            ["Qdrant NVMe disk", gb(s["qdrant_nvme_bytes"]),
             "estimate - from declared per-episode byte sizes"],
            ["PostgreSQL disk",
             f"{gb(s['postgres_bytes_low'])} to {gb(s['postgres_bytes_high'])}",
             "estimate - from declared per-episode byte sizes"],
            ["One year, nothing ever deleted: hot vector RAM",
             gb(s["unbounded_year_hot_vector_ram_bytes"]), "derived"],
            ["One year, nothing ever deleted: Qdrant NVMe",
             gb(s["unbounded_year_qdrant_nvme_bytes"]), "estimate"],
            ["One year, nothing ever deleted: PostgreSQL disk",
             (f"{gb(s['unbounded_year_postgres_bytes_low'])} to "
              f"{gb(s['unbounded_year_postgres_bytes_high'])}"), "estimate"],
        ],
    })

    qdrant_rows = []
    for opt in m["qdrant_options"]:
        chosen = "  <- chosen" if opt["node_ram_gb"] == m["qdrant_node_ram_gb"] else ""
        qdrant_rows.append([
            f"{num(opt['node_ram_gb'])} GB",
            f"{num(opt['usable_gb_per_node'], 1)} GB",
            f"{opt['nodes']}{chosen}",
            f"{num(opt['total_ram_gb'])} GB",
            f"{num(opt['fill_of_allowance'] * 100, 2)}%",
            f"{num(opt['share_of_node_ram'] * 100, 2)}%",
        ])
    sections.append({
        "title": "Qdrant node choice",
        "note": "Every row here is a what-if, not a finding, so the rows carry "
                "no label of their own: one row is the order, the others are "
                "what the other machine sizes would have cost. All of them are "
                "derived from the hot vector RAM above and the two assumptions "
                "in this note. "
                "A node is filled to at most "
                f"{num(QDRANT_NODE_FILL_LIMIT * 100, 0)}% of its RAM, leaving "
                "room for the operating system, Qdrant's own metadata and shards "
                "that come out uneven. "
                + ("The size was forced with "
                   f"{i.get('node_gb_source', NODE_GB_SOURCE_CLI)}."
                   if m["qdrant_node_ram_gb_forced"] else
                   "The chosen size is the one that buys the least total RAM; "
                   "a tie goes to fewer machines.")
                + (" WARNING: the chosen size is more than "
                   f"{num(QDRANT_TIGHT_FIT_WARN_FRACTION * 100, 0)}% full within "
                   "that allowance, so a small growth in retention adds a whole "
                   "machine." if m["qdrant_tight_fit"] else ""),
        "headers": ["Node RAM", "Usable per node", "Nodes needed",
                    "Total RAM bought", "Fill of allowance",
                    "Share of node RAM used"],
        "rows": qdrant_rows,
    })

    pooler = ("YES - more connections than have ever been proven to work; put "
              "PgBouncer or an equivalent connection pooler in front of "
              "PostgreSQL"
              if p["needs_connection_pooler"] else "no")
    sections.append({
        "title": "PostgreSQL",
        "note": "Connections, not compute, are what failed on 30 Aug 2026: the "
                "core filled the connection table, the gateway could then not "
                "get a connection to check API keys, and it returned HTTP 401 "
                "on valid keys.",
        "headers": ["Item", "Value", "Label"],
        "rows": [
            ["Statements per second", num(p["statements_per_s"], 1), "derived"],
            ["Workers per API server", num(p["workers_per_api_server"]),
             "measured 30 Aug 2026 - 8 is the knee"],
            ["Connections per worker", num(p["connections_per_worker"]),
             (f"measured - pool {POSTGRES_POOL_SIZE} + overflow "
              f"{POSTGRES_MAX_OVERFLOW}")],
            ["Core connections", num(p["core_connections"]), "derived"],
            ["Gateway connections", num(p["gateway_connections"]),
             "assumption - 20 per API server"],
            ["max_connections this tier needs",
             num(p["max_connections_required"]), "derived"],
            ["Chart default", num(p["chart_default_max_connections"]),
             "measured - this default failed on 30 Aug 2026"],
            ["Largest setting ever proven to work",
             num(p["proven_max_connections"]),
             "measured 30 Aug 2026 - cleared every error"],
            ["Needs a connection pooler", pooler, "derived"],
        ],
    })

    sections.append({
        "title": "Network",
        "note": "Every byte size behind these figures is an estimate declared as "
                "a named constant in this program, so the numbers can be checked "
                "by hand. Mbps means 10^6 bits per second.",
        "headers": ["Item", "Value", "Label"],
        "rows": [
            ["North-south peak (clients to the service)",
             f"{num(n['north_south_mbps'], 1)} Mbps", "estimate"],
            ["East-west peak (between servers inside the data center)",
             f"{num(n['east_west_mbps'], 1)} Mbps", "estimate"],
            ["Bytes per embedding call",
             f"{num(n['embed_bytes_per_call'])} bytes", "estimate"],
            ["Bytes per vector search",
             f"{num(n['vector_search_bytes_per_call'])} bytes", "estimate"],
            ["Bytes per vector write",
             f"{num(n['vector_write_bytes_per_call'])} bytes", "estimate"],
            ["Headroom on a 10 GbE link (busiest direction)",
             (f"{num(n['headroom_on_10gbe'], 1)}x"
              if n["headroom_on_10gbe"] is not None else "no traffic"),
             "derived from estimates - no measured number enters it"],
        ],
    })

    sections.append({
        "title": "Callers this capacity holds",
        "note": caller_rates_note()
                + " Each row below is that kind of caller on its own, sized "
                  "at this run's traffic mix. The first two rows are the "
                  "rates this deployment is sized on. The last two are "
                  "reported for comparison and enter no machine count: the "
                  "heavy-session row is the headroom check, and the "
                  "sustained-rate row is the average-load check.",
        "headers": ["Kind of caller", "Sessions held", "Label"],
        "rows": [
            [("Concurrent human chat sessions "
              f"({u['human_ops_per_s_low']:g}-"
              f"{u['human_ops_per_s_high']:g} ops/s each)"),
             (f"{num(u['human_sessions_low'], 0)} to "
              f"{num(u['human_sessions_high'], 0)}"),
             LABEL_HUMAN_RATE_BAND],
            [("Concurrent automated client sessions in a 5-second tool "
              "loop "
              f"({u['automated_client_ops_per_s']:g} ops/s each)"),
             num(u["automated_client_sessions"], 0),
             LABEL_AUTOMATED_RATE_PEAK],
            [("Concurrent heavy human chat sessions - what the busiest one "
              f"in a hundred demands ({u['human_ops_per_s_heavy']:g} ops/s "
              "each)"),
             num(u["human_sessions_heavy"], 0),
             LABEL_HUMAN_RATE_HEAVY],
            [("Concurrent automated client sessions at the sustained rate, "
              "counting the time they sit idle "
              f"({u['automated_client_ops_per_s_sustained']:g} ops/s each)"),
             num(u["automated_client_sessions_sustained"], 0),
             LABEL_AUTOMATED_RATE_SUSTAINED],
        ],
    })

    # The agent rate takes one decimal place, matching the demand section. One
    # decimal place is not enough to separate every pair of rates the table can
    # hold - a mix that makes the run's own rate 2.04 puts it beside the fixed
    # rate 2.0 and both print as "2.0" - so the run's own row says so in words.
    sens_rows = [
        [num(row["agent_searches_per_s"], 1)
         + ("  <- this run" if row["is_this_run"] else ""),
         num(row["total_ops_per_s"], 1),
         num(row["vector_searches_per_s"], 1),
         num(row["api_work_per_s"], 1),
         num(row["api_servers"]),
         (f"{num(row['llm_calls_per_s_low'], 0)} to "
          f"{num(row['llm_calls_per_s_high'], 0)}")]
        for row in r["sensitivity"]
    ]
    sections.append({
        "title": "Sensitivity: what the agent-mode quota costs",
        "note": "Every row here is a what-if, not a finding, so the rows carry "
                'no label of their own: the one marked "this run" is the '
                "traffic mix you asked about. All of them are derived from the "
                "same "
                "fan-out counts and the same 180/s anchor as the report above. "
                f"Adds are held fixed at {num(r['demand']['adds_per_s'], 1)}/s and "
                f"plain searches at {num(r['demand']['plain_searches_per_s'], 1)}/s. "
                "Only the agent-mode rate varies. Agent-mode search is a flag "
                "on a request, not a kind of caller. One agent-mode search "
                f"costs about {AGENT_VECTOR_SEARCHES} plain searches, so this "
                "one product decision moves the hardware order more than any "
                "tuning does.",
        "headers": ["Agent-mode searches/s", "Total ops/s", "Vector searches/s",
                    "API work/s", "API servers", "Language-model calls/s"],
        "rows": sens_rows,
    })

    return sections


def render_table(headers: list, rows: list, indent: str = "  ") -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    lines = []
    lines.append(indent + "  ".join(h.ljust(widths[idx])
                                    for idx, h in enumerate(headers)).rstrip())
    lines.append(indent + "  ".join("-" * widths[idx]
                                    for idx in range(len(headers))))
    lines.extend(
        indent + "  ".join(str(cell).ljust(widths[idx])
                           for idx, cell in enumerate(row)).rstrip()
        for row in rows)
    return "\n".join(lines)


def wrap(text: str, width: int = 78, indent: str = "  ") -> str:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(indent + current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(indent + current)
    return "\n".join(lines)


def caller_rates_note() -> str:
    """Where the four per-caller rates come from, with both sources named.

    Shared by the tier report and the population report, so a reader meets the
    same citation whichever door they came in by.
    """
    spread = AUTOMATED_CLIENT_OPS_PER_S / AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED
    return (
        "Every per-caller rate here is an estimate, but each now has a "
        "published measurement behind it. The human figures come from "
        f"{BURSTGPT_CITATION}: 110 consecutive days of one real Azure OpenAI "
        "deployment, 55,295 conversation sessions, 176,466 gaps between one "
        "prompt and the next. Measured over a session's busiest five minutes, "
        f"{HUMAN_SESSION_OPS_PER_S_LOW:g} operations per second is about the "
        f"median and {HUMAN_SESSION_OPS_PER_S_HIGH:g} is about the 90th "
        f"percentile - 9 sessions in 10 are slower than that. "
        f"{HUMAN_SESSION_OPS_PER_S_HEAVY:g} is the 99th percentile: the "
        "busiest 1 session in 100. BurstGPT is one regional deployment and "
        "who its users were is not published, so it is a guide, not your "
        "traffic. The automated-client figures come from "
        f"{TRACELAB_CITATION}: about 4,300 real Claude Code and Codex "
        "coding-agent sessions, roughly 350,000 model steps and 430,000 tool "
        "calls from 43 developers over about 8 months. A step is one model "
        "generation plus the tool call it asks for. The median step took 5.0 "
        "seconds, which at two operations per step is "
        f"{AUTOMATED_CLIENT_OPS_PER_S:g} operations per second - the design "
        "peak this deployment is sized on, and the same 5-second tool loop "
        "this program has always described. The mean step took 28.3 seconds, "
        f"which is {AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED:g} operations per "
        f"second sustained. The two differ by about {spread:.0f} times "
        "because an agent is idle most of the wall-clock time, waiting on the "
        "person: TraceLab measured human thinking at 92.3% of session "
        "wall-clock time. Size from the sustained figure and a burst is "
        "under-provisioned; size from the design peak and a day is "
        "over-provisioned. This model sizes for the worst five minutes, so it "
        "uses the design peak. TraceLab is 43 developers using coding agents, "
        "not automated clients in general. Meter real operations per second "
        "per API key from the first day of the pilot and check both rates "
        "against your own traffic. An automated client is a kind of CALLER - "
        "a program sending requests in a loop - and has nothing to do with "
        "agent-mode search, which is a flag on one REQUEST.")


def design_peak_note() -> str:
    """Why five minutes, and how to get a peak when you only know an average.

    Two things a reader needs and neither of them moves a machine count. The
    five-minute window has an independent citation, so that it reads as an
    engineering convention rather than a number this program picked. The
    peak-to-average multiplier is REPORTED ONLY: nothing in this program
    multiplies by it, and a reader who knows their average load has to do the
    multiplication themselves and pass the result in.
    """
    return (
        "A design peak is the worst rate the system must sustain for five "
        "minutes, not an average. Five minutes is not this program's "
        "invention. ITU-T E.500, the international recommendation for "
        f'measuring telephone traffic, requires windows "{ITU_E500_QUOTE}" '
        f"({ITU_E500_CITATION}). If what you know is your AVERAGE load rather "
        "than your peak, multiply it by "
        f"{PEAK_TO_AVERAGE_SINGLE_ORGANISATION:g} to get a design peak for a "
        "deployment inside one organisation. That multiplier is REPORTED "
        "ONLY: no machine count in this program reads it, and nothing here "
        "multiplies by it for you. It comes from measured traces, which "
        "disagree by scale. A service for a few hundred users ran at "
        f"{PEAK_TO_AVERAGE_MEASURED_SMALL_SERVICE:g} times its own mean and "
        f"Hotmail at {PEAK_TO_AVERAGE_LARGE_SERVICE_HIGH:g} times "
        f"({WANG_CITATION}), while a Google production cell of 12,500 "
        f"machines ran at {PEAK_TO_AVERAGE_LARGE_SERVICE_LOW:g} times "
        f"({GOOGLE_TRACE_CITATION}). A small population is burstier because "
        "scale flattens the curve: one team going to lunch together is "
        "visible in a few hundred users and invisible in a few million.")


def population_note(with_counting_rule: bool = True) -> str:
    """The paragraph above the population tables, in the model's own numbers.

    The counting rule is dropped when the conversion table is printed above
    this one, because that table already opens with it and a reader should not
    meet the same paragraph twice in a row.
    """
    spread_low = AUTOMATED_CLIENT_OPS_PER_S / HUMAN_SESSION_OPS_PER_S_HIGH
    spread_high = AUTOMATED_CLIENT_OPS_PER_S / HUMAN_SESSION_OPS_PER_S_LOW
    prompts_low = HUMAN_SESSION_OPS_PER_S_LOW * 3600 / OPS_PER_HUMAN_PROMPT
    prompts_high = HUMAN_SESSION_OPS_PER_S_HIGH * 3600 / OPS_PER_HUMAN_PROMPT
    return (
        (f"{COUNTING_RULE} " if with_counting_rule else "")
        + "There are two kinds of caller here. A human chat session is a person "
        f"typing, assumed to make {HUMAN_SESSION_OPS_PER_S_LOW:g} to "
        f"{HUMAN_SESSION_OPS_PER_S_HIGH:g} operations per second. That is about "
        f"{prompts_low:.0f} prompts an hour at the low end and about "
        f"{prompts_high:.0f} prompts an hour at the high end, at roughly "
        f"{OPS_PER_HUMAN_PROMPT:g} operations per prompt. An automated client "
        "is a program that sends requests in a loop rather than a person; in a "
        f"5-second tool loop it is assumed to make {AUTOMATED_CLIENT_OPS_PER_S:g} "
        f"operations per second, which is {spread_low:.0f} to "
        f"{spread_high:.0f} times a human. Callers of either kind can send "
        "requests of either kind, which is why each population below carries "
        "its own traffic mix. The last two rows of the table are checks, not "
        "demand: they are reported and no machine count uses them. "
        + caller_rates_note() + " " + design_peak_note())


def tier_phrase(name, ops_per_s: float) -> str:
    """How the report names the smallest tier that holds a rate."""
    if name is None:
        biggest = TIER_OPS_PER_S[TIER_ORDER[-1]]
        return (f"above the scale tier ({num(biggest, 0)} ops/s) by "
                f"{ops_per_s / biggest:.1f}x - no named tier holds it")
    return f"{name} ({num(TIER_OPS_PER_S[name], 0)} ops/s design peak)"


def conversion_column(one) -> list:
    """The first three cells of one caller kind's column, top to bottom.

    A kind whose concurrent sessions were typed in directly has no user count
    and no conversion figures, and the column says so rather than printing a
    dash a reader has to interpret.
    """
    if one is None:
        return [CONVERSION_NOT_USED_CELL] * 3
    return [num(one["users"], 0),
            f"{as_given(one['peak_active_per_100'])} per 100 users",
            as_given(one["sessions_per_active_user"])]


def conversion_row_label(conversion: dict, flag: str, why: str) -> str:
    """The label column for one conversion row, marking each figure it holds.

    A row holds one figure per caller kind, and the two can differ: the reader
    may give a share for people and let the automated side take the example
    default. So the label names each kind that has a figure in the row and
    marks it supplied or example default.

    When every figure in the row was supplied the label is exactly what it has
    always been, so a reader who gives their own numbers sees no change.
    """
    parts, all_supplied = [], True
    for key, defaults in (("human", HUMAN_CONVERSION_DEFAULTS),
                          ("automated", AUTOMATED_CONVERSION_DEFAULTS)):
        one = conversion.get(key)
        if one is None:
            continue
        if one.get(flag):
            all_supplied = False
            reason = defaults.why_share if why == "share" else (
                defaults.why_per_user)
            parts.append(f"{defaults.column}: {CONVERSION_MARK_DEFAULT} - "
                         f"{reason}")
        else:
            parts.append(f"{defaults.column}: {CONVERSION_MARK_SUPPLIED}")
    if all_supplied or not parts:
        return LABEL_READER_ASSUMPTION
    return "; ".join(parts)


def conversion_sections(pop: dict) -> list:
    """The users-to-concurrent-sessions table, or nothing at all.

    Nothing at all when the concurrent sessions were given directly, which is
    every run that existed before this table did.

    The table is here so the multiplication can be argued with. Every figure
    in its two middle rows is marked in the label column as either supplied by
    the reader or an example default this program chose, so the two can never
    be mistaken for each other, and a note under the table warns about any
    example default that was used.
    """
    conversion = pop.get("conversion")
    if conversion is None:
        return []
    human = conversion_column(conversion["human"])
    automated = conversion_column(conversion["automated"])
    note = (f"{COUNTING_RULE} The rest of this report counts those "
            "concurrent sessions. A user count is a "
            "different number, and two figures turn one into the other. "
            "The share active at "
            "the busiest moment is a few per 100 users for most services, "
            "so the same user base can be 500 concurrent sessions or "
            "5,000 - a tenfold spread that moves the answer two whole "
            "tiers. The sessions one active user holds is about 1 for a "
            "person, but 10 or 50 for one person running automated "
            "clients. Multiply the three rows together to get the last "
            "one, and change either assumption to see the whole answer "
            "move.")
    warning = defaulted_figures_note(conversion)
    if warning:
        note = f"{note} {warning}"
    return [{
        "title": "From users to concurrent sessions",
        "note": note,
        "headers": ["Step", "People", "Automated clients", "Label"],
        "rows": [
            ["Users", human[0], automated[0], LABEL_READER_INPUT],
            ["Share of those users active at the busiest moment",
             human[1], automated[1],
             conversion_row_label(conversion, "peak_active_per_100_is_default",
                                  "share")],
            ["Sessions per active user", human[2], automated[2],
             conversion_row_label(
                 conversion, "sessions_per_active_user_is_default",
                 "per_user")],
            ["Concurrent sessions", num(pop["humans"], 0),
             num(pop["automated"], 0), LABEL_CONVERSION_DERIVED],
        ],
    }]


def population_sections(pop: dict) -> list:
    """Every table of the population report, as (title, note, headers, rows).

    Shared by the text report and the web page, so the two can never disagree.
    """
    human_mix = TrafficMix(**pop["human_mix"])
    automated_mix = TrafficMix(**pop["automated_mix"])
    blended = TrafficMix(**pop["blended_mix"])
    sections = conversion_sections(pop)
    sections += [{
        "title": "Demand from this population",
        "note": population_note(pop.get("conversion") is None),
        "headers": ["Kind of caller", "Count", "Rate each", "Demand", "Label"],
        "rows": [
            ["Concurrent human chat sessions", num(pop["humans"], 0),
             (f"{HUMAN_SESSION_OPS_PER_S_LOW:g}-"
              f"{HUMAN_SESSION_OPS_PER_S_HIGH:g} ops/s each"),
             (f"{num(pop['human_ops_per_s_low'], 2)} to "
              f"{num(pop['human_ops_per_s_high'], 2)} ops/s"),
             LABEL_HUMAN_RATE_BAND],
            ["Concurrent automated client sessions",
             num(pop["automated"], 0),
             f"{AUTOMATED_CLIENT_OPS_PER_S:g} ops/s each",
             f"{num(pop['automated_ops_per_s'], 2)} ops/s",
             LABEL_AUTOMATED_RATE_PEAK],
            ["Total", num(pop["humans"] + pop["automated"], 0), "-",
             (f"{num(pop['ops_per_s_low'], 2)} to "
              f"{num(pop['ops_per_s_high'], 2)} ops/s"),
             "derived from the two estimates above"],
            ["Headroom check: every human session a heavy one",
             num(pop["humans"], 0),
             f"{HUMAN_SESSION_OPS_PER_S_HEAVY:g} ops/s each",
             f"{num(pop['human_ops_per_s_heavy'], 2)} ops/s",
             LABEL_HUMAN_RATE_HEAVY],
            ["Average-load check: automated clients over a whole session",
             num(pop["automated"], 0),
             f"{AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED:g} ops/s each",
             f"{num(pop['automated_ops_per_s_sustained'], 2)} ops/s",
             LABEL_AUTOMATED_RATE_SUSTAINED],
        ],
    }, {
        "title": "Traffic mix, per caller and blended",
        "note": "Each population has its own mix, because the kind of caller "
                "and the kind of request are correlated: automated clients may "
                "use agent-mode search on nearly every call while people rarely "
                "do. The blended row is the two mixes averaged, each weighted "
                "by the operations that population demands at the busy end of "
                "the human rate - the rate this report tells you to plan for. "
                "The blended row, and not this program's default mix, is what "
                "sizes the deployment below.",
        "headers": ["Whose mix", "Per 100 operations", "Label"],
        "rows": [
            ["Concurrent human chat sessions", human_mix.as_words(),
             "assumption - never measured"],
            ["Concurrent automated client sessions",
             automated_mix.as_words(),
             "assumption - never measured"],
            ["Blended across the whole population", blended.as_words(),
             "derived from the two mixes above and the demand table"],
        ],
    }]
    # A population that makes no requests is not a deployment to size, so it
    # gets no tier and no machines. Naming the pilot tier for it would be a
    # hardware recommendation for nobody.
    if pop["ops_per_s_high"] <= 0:
        return sections

    sections.append({
        "title": "Smallest tier that holds this population",
        "note": "The same headcount can need a pilot tier or a scale tier "
                "depending on how many of the callers are automated clients "
                "rather than people, and on how much agent-mode search each "
                "population does.",
        "headers": ["Case", "Demand", "Smallest tier that holds it", "Label"],
        "rows": [
            ["If humans are at the low rate",
             f"{num(pop['ops_per_s_low'], 2)} ops/s",
             tier_phrase(pop["tier_for_low"], pop["ops_per_s_low"]),
             "derived"],
            ["If humans are at the high rate",
             f"{num(pop['ops_per_s_high'], 2)} ops/s",
             tier_phrase(pop["tier_for_high"], pop["ops_per_s_high"]),
             "derived"],
        ],
    })

    m = pop["sizing"]["machines"]
    embed = (f"{m['embed_gpu_cards_low']}"
             if m["embed_gpu_cards_low"] == m["embed_gpu_cards_high"]
             else f"{m['embed_gpu_cards_low']} to {m['embed_gpu_cards_high']}")
    sections.append({
        "title": "Machines this population needs",
        "note": f"Sized at {num(pop['ops_per_s_high'], 2)} operations/s - the "
                "high rate, which is the one to plan for - and with the "
                "blended mix above rather than this program's default mix. "
                "Run tier or calc for the full report behind these counts.",
        "headers": ["Machine", "Count", "Label"],
        "rows": [
            ["API server (gateway + MemMachine core)", num(m["api_servers"]),
             "derived from the 180/s anchor measured 30 Aug 2026"],
            ["PostgreSQL server", num(m["postgres_servers"]),
             "assumption - never benchmarked at this statement rate"],
            [f"Qdrant server ({num(m['qdrant_node_ram_gb'])} GB RAM each)",
             num(m["qdrant_servers"]),
             "derived from retention and vector size"],
            ["Embedding GPU card", embed,
             "estimate - card rate never benchmarked"],
            ["Agent-model GPU card", num(m["agent_gpu_cards"]),
             "estimate - card rate never benchmarked"],
        ],
    })
    return sections


def render_report(r: dict, title: str) -> str:
    out = []
    out.append("=" * 80)
    out.append(title)
    out.append("=" * 80)
    for section in report_sections(r):
        out.append("")
        out.append(section["title"])
        out.append("-" * len(section["title"]))
        if section["note"]:
            out.append(wrap(section["note"]))
            out.append("")
        out.append(render_table(section["headers"], section["rows"]))
    out.append("")
    out.append("Labels: measured = from a real test, named by its date, its "
               "configuration, or both.")
    out.append("        derived  = computed by this program from measured "
               "numbers.")
    out.append("        estimate = not measured on this system; where a "
               "published source stands behind")
    out.append("                   it, the label names it. Benchmark before "
               "ordering hardware.")
    out.append("        assumption = a planning choice, not a finding.")
    out.append("        The Qdrant node choice table and the sensitivity table "
               "show what-ifs rather")
    out.append("        than findings, so their rows carry no label; the note "
               "above each says")
    out.append("        where its numbers come from.")
    return "\n".join(out)


# =============================================================================
# VALIDATE - the published figures and the model's constants, as
# "name: value" lines. It is a fixed, named list that the test suite pins, not
# a dump of every value the model computes on the way to an answer.
# =============================================================================


def round_out(value):
    """Round floats so the JSON file is stable and diffable."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    if abs(value) >= 1e12:
        return round(value, 0)
    return round(value, 4)


def published_numbers(name: str, r: dict) -> list:
    """Flat (key, value) pairs for one tier, in a fixed order."""
    d, m, s, p, n, u, i = (r["demand"], r["machines"], r["storage"],
                           r["postgres"], r["network"], r["users"], r["inputs"])
    pairs = [
        (f"{name}.ops_per_s", i["ops_per_s"]),
        (f"{name}.mix_add", i["mix"]["add"]),
        (f"{name}.mix_plain", i["mix"]["plain"]),
        (f"{name}.mix_agent", i["mix"]["agent"]),
        (f"{name}.retention_days", i["retention_days"]),
        (f"{name}.vector_dims", i["dims"]),
        (f"{name}.bytes_per_value", i["bytes_per_value"]),
        # The node-size input under its own name. The size itself is
        # qdrant_node_ram_gb below: when this flag is true that is the size
        # that was forced, and when it is false it is the size the program
        # chose. Exporting the raw input would put a null in the file on every
        # ordinary run, which is harder to read than the flag.
        (f"{name}.node_gb_forced", i["node_gb"] is not None),

        (f"{name}.adds_per_s", d["adds_per_s"]),
        (f"{name}.plain_searches_per_s", d["plain_searches_per_s"]),
        (f"{name}.agent_searches_per_s", d["agent_searches_per_s"]),
        (f"{name}.embeds_per_s", d["embeds_per_s"]),
        (f"{name}.embeds_per_s_with_types_fix", d["embeds_per_s_with_types_fix"]),
        (f"{name}.vector_searches_per_s", d["vector_searches_per_s"]),
        (f"{name}.vector_writes_per_s", d["vector_writes_per_s"]),
        (f"{name}.postgres_statements_per_s", d["postgres_statements_per_s"]),
        (f"{name}.agent_llm_calls_per_s_low", d["agent_llm_calls_per_s_low"]),
        (f"{name}.agent_llm_calls_per_s_high", d["agent_llm_calls_per_s_high"]),
        (f"{name}.agent_llm_calls_per_s_planning",
         d["agent_llm_calls_per_s_planning"]),

        (f"{name}.api_work_per_s", m["api_work_per_s"]),
        (f"{name}.api_usable_searches_per_server",
         m["api_usable_searches_per_server"]),
        (f"{name}.api_servers", m["api_servers"]),
        (f"{name}.api_server_spec", m["api_server_spec"]),
        (f"{name}.postgres_servers", m["postgres_servers"]),
        (f"{name}.qdrant_servers", m["qdrant_servers"]),
        (f"{name}.qdrant_node_ram_gb", m["qdrant_node_ram_gb"]),
        (f"{name}.qdrant_usable_gb_per_node", m["qdrant_usable_gb_per_node"]),
        (f"{name}.qdrant_total_ram_gb", m["qdrant_total_ram_gb"]),
        (f"{name}.qdrant_fill_of_allowance_pct",
         m["qdrant_fill_of_allowance"] * 100),
        (f"{name}.qdrant_tight_fit", m["qdrant_tight_fit"]),
        (f"{name}.embed_gpu_cards_low", m["embed_gpu_cards_low"]),
        (f"{name}.embed_gpu_cards_high", m["embed_gpu_cards_high"]),
        (f"{name}.embed_gpu_spare", m["embed_gpu_spare"]),
        (f"{name}.agent_gpu_cards", m["agent_gpu_cards"]),
        (f"{name}.agent_gpu_spare", m["agent_gpu_spare"]),
        (f"{name}.total_cpu_servers", m["total_cpu_servers"]),
    ]
    pairs.extend(
        (f"{name}.qdrant_nodes_at_{opt['node_ram_gb']}gb", opt["nodes"])
        for opt in m["qdrant_options"])
    pairs += [
        (f"{name}.episodes_per_day", s["episodes_per_day"]),
        (f"{name}.episodes_per_year", s["episodes_per_year"]),
        (f"{name}.episodes_retained", s["episodes_retained"]),
        (f"{name}.hot_vector_ram_gb", s["hot_vector_ram_gb"]),
        (f"{name}.qdrant_nvme_gb", s["qdrant_nvme_gb"]),
        (f"{name}.postgres_gb_low", s["postgres_gb_low"]),
        (f"{name}.postgres_gb_high", s["postgres_gb_high"]),
        (f"{name}.unbounded_year_hot_vector_ram_gb",
         s["unbounded_year_hot_vector_ram_gb"]),
        (f"{name}.unbounded_year_qdrant_nvme_gb",
         s["unbounded_year_qdrant_nvme_gb"]),
        (f"{name}.unbounded_year_postgres_gb_low",
         s["unbounded_year_postgres_gb_low"]),
        (f"{name}.unbounded_year_postgres_gb_high",
         s["unbounded_year_postgres_gb_high"]),

        (f"{name}.postgres_core_connections", p["core_connections"]),
        (f"{name}.postgres_gateway_connections", p["gateway_connections"]),
        (f"{name}.postgres_total_connections", p["total_connections"]),
        (f"{name}.postgres_max_connections_required",
         p["max_connections_required"]),
        (f"{name}.postgres_exceeds_chart_default", p["exceeds_chart_default"]),
        (f"{name}.postgres_exceeds_proven_setting", p["exceeds_proven_setting"]),
        (f"{name}.postgres_needs_connection_pooler", p["needs_connection_pooler"]),

        (f"{name}.network_north_south_mbps", n["north_south_mbps"]),
        (f"{name}.network_east_west_mbps", n["east_west_mbps"]),
        (f"{name}.network_busiest_link_mbps", n["busiest_link_mbps"]),
        (f"{name}.network_embed_bytes_per_call", n["embed_bytes_per_call"]),
        (f"{name}.network_vector_search_bytes_per_call",
         n["vector_search_bytes_per_call"]),
        (f"{name}.network_vector_write_bytes_per_call",
         n["vector_write_bytes_per_call"]),
        (f"{name}.network_llm_bytes_per_call", n["llm_bytes_per_call"]),
        (f"{name}.network_headroom_on_10gbe", n["headroom_on_10gbe"]),

        (f"{name}.human_sessions_low", u["human_sessions_low"]),
        (f"{name}.human_sessions_high", u["human_sessions_high"]),
        # Reported only, like the two constants behind them.
        (f"{name}.human_sessions_heavy", u["human_sessions_heavy"]),
        (f"{name}.automated_client_sessions", u["automated_client_sessions"]),
        (f"{name}.automated_client_sessions_sustained",
         u["automated_client_sessions_sustained"]),
    ]
    # Numbered rows, not a key built from the agent rate. The rate is a float
    # that the traffic mix can make fractional, so a key like "agent2" put the
    # 2.0 row and the 2.5 row in the same place and the second silently
    # overwrote the first. Each row now carries its own rate as a value.
    for position, row in enumerate(r["sensitivity"], start=1):
        stem = f"{name}.sensitivity.row{position}"
        pairs += [
            (f"{stem}.agent_searches_per_s", row["agent_searches_per_s"]),
            (f"{stem}.total_ops_per_s", row["total_ops_per_s"]),
            (f"{stem}.vector_searches_per_s", row["vector_searches_per_s"]),
            (f"{stem}.api_work_per_s", row["api_work_per_s"]),
            (f"{stem}.api_servers", row["api_servers"]),
            (f"{stem}.llm_calls_per_s_low", row["llm_calls_per_s_low"]),
            (f"{stem}.llm_calls_per_s_high", row["llm_calls_per_s_high"]),
        ]
    return pairs


def constant_numbers() -> list:
    """The model's own inputs, so they can be quoted elsewhere and checked.

    Every named constant in the README's "Every input, and what it is set to"
    table appears here, under its own name in lower case. A test reads that
    table and fails if any of them is missing, so the two cannot drift apart.
    """
    return [
        ("constants.tier_ops_per_s_pilot", TIER_OPS_PER_S["pilot"]),
        ("constants.tier_ops_per_s_target", TIER_OPS_PER_S["target"]),
        ("constants.tier_ops_per_s_scale", TIER_OPS_PER_S["scale"]),
        ("constants.sensitivity_agent_rates", list(SENSITIVITY_AGENT_RATES)),
        ("constants.api_searches_per_s_per_server",
         API_SEARCHES_PER_S_PER_SERVER),
        ("constants.api_utilization_ceiling", API_UTILIZATION_CEILING),
        ("constants.api_usable_searches_per_server",
         api_usable_searches_per_server()),
        ("constants.api_workers_per_server", API_WORKERS_PER_SERVER),
        ("constants.api_server_vcpu", API_SERVER_VCPU),
        ("constants.api_server_ram_gb", API_SERVER_RAM_GB),
        ("constants.add_embeds", ADD_EMBEDS),
        ("constants.add_vector_writes", ADD_VECTOR_WRITES),
        ("constants.add_postgres_statements", ADD_POSTGRES_STATEMENTS),
        ("constants.add_llm_calls", ADD_LLM_CALLS),
        ("constants.add_cost_in_plain_search_equivalents",
         ADD_COST_IN_PLAIN_SEARCH_EQUIVALENTS),
        ("constants.plain_embeds", PLAIN_EMBEDS),
        ("constants.plain_embeds_with_types_fix", PLAIN_EMBEDS_WITH_TYPES_FIX),
        ("constants.plain_vector_searches", PLAIN_VECTOR_SEARCHES),
        ("constants.plain_postgres_statements", PLAIN_POSTGRES_STATEMENTS),
        ("constants.plain_llm_calls", PLAIN_LLM_CALLS),
        ("constants.agent_embeds", AGENT_EMBEDS),
        ("constants.agent_vector_searches", AGENT_VECTOR_SEARCHES),
        ("constants.agent_postgres_statements", AGENT_POSTGRES_STATEMENTS),
        ("constants.agent_llm_calls_low", AGENT_LLM_CALLS_LOW),
        ("constants.agent_llm_calls_high", AGENT_LLM_CALLS_HIGH),
        ("constants.agent_llm_calls_planning", AGENT_LLM_CALLS_PLANNING),
        ("constants.embed_card_requests_per_s_low",
         EMBED_CARD_REQUESTS_PER_S_LOW),
        ("constants.embed_card_requests_per_s_high",
         EMBED_CARD_REQUESTS_PER_S_HIGH),
        ("constants.gpu_utilization_ceiling", GPU_UTILIZATION_CEILING),
        ("constants.embed_usable_per_card_low",
         EMBED_CARD_REQUESTS_PER_S_LOW * GPU_UTILIZATION_CEILING),
        ("constants.embed_usable_per_card_high",
         EMBED_CARD_REQUESTS_PER_S_HIGH * GPU_UTILIZATION_CEILING),
        ("constants.gpu_spare_cards", GPU_SPARE_CARDS),
        ("constants.agent_llm_calls_per_s_per_card",
         AGENT_LLM_CALLS_PER_S_PER_CARD),
        ("constants.qdrant_index_overhead_factor", QDRANT_INDEX_OVERHEAD_FACTOR),
        ("constants.qdrant_node_fill_limit", QDRANT_NODE_FILL_LIMIT),
        ("constants.qdrant_node_ram_options_gb",
         list(QDRANT_NODE_RAM_OPTIONS_GB)),
        ("constants.qdrant_tight_fit_warn_fraction",
         QDRANT_TIGHT_FIT_WARN_FRACTION),
        ("constants.bytes_per_gb", BYTES_PER_GB),

        ("constants.max_ops_per_s", MAX_OPS_PER_S),
        ("constants.max_retention_days", MAX_RETENTION_DAYS),
        ("constants.max_vector_dims", MAX_VECTOR_DIMS),
        ("constants.max_bytes_per_value", MAX_BYTES_PER_VALUE),
        ("constants.max_node_gb", MAX_NODE_GB),
        ("constants.max_users", MAX_USERS),
        ("constants.max_peak_active_per_100", MAX_PEAK_ACTIVE_PER_100),
        ("constants.max_sessions_per_active_user",
         MAX_SESSIONS_PER_ACTIVE_USER),

        ("constants.original_vector_bytes_per_value",
         ORIGINAL_VECTOR_BYTES_PER_VALUE),
        ("constants.qdrant_disk_payload_bytes_per_episode",
         QDRANT_DISK_PAYLOAD_BYTES_PER_EPISODE),
        ("constants.qdrant_disk_overhead_factor", QDRANT_DISK_OVERHEAD_FACTOR),
        ("constants.episode_text_bytes_low", EPISODE_TEXT_BYTES_LOW),
        ("constants.episode_text_bytes_high", EPISODE_TEXT_BYTES_HIGH),
        ("constants.postgres_row_overhead_bytes", POSTGRES_ROW_OVERHEAD_BYTES),
        ("constants.postgres_index_bytes_per_episode",
         POSTGRES_INDEX_BYTES_PER_EPISODE),
        ("constants.postgres_bloat_factor", POSTGRES_BLOAT_FACTOR),

        ("constants.postgres_pool_size", POSTGRES_POOL_SIZE),
        ("constants.postgres_max_overflow", POSTGRES_MAX_OVERFLOW),
        ("constants.postgres_connections_per_worker",
         POSTGRES_CONNECTIONS_PER_WORKER),
        ("constants.gateway_connections_per_api_server",
         GATEWAY_CONNECTIONS_PER_API_SERVER),
        ("constants.postgres_chart_default_max_connections",
         POSTGRES_CHART_DEFAULT_MAX_CONNECTIONS),
        ("constants.postgres_proven_max_connections",
         POSTGRES_PROVEN_MAX_CONNECTIONS),
        ("constants.postgres_servers_per_tier", POSTGRES_SERVERS_PER_TIER),

        ("constants.ns_add_request_bytes", NS_ADD_REQUEST_BYTES),
        ("constants.ns_add_response_bytes", NS_ADD_RESPONSE_BYTES),
        ("constants.ns_search_request_bytes", NS_SEARCH_REQUEST_BYTES),
        ("constants.ns_response_bytes_per_episode",
         NS_RESPONSE_BYTES_PER_EPISODE),
        ("constants.ns_agent_answer_bytes", NS_AGENT_ANSWER_BYTES),
        ("constants.plain_search_episodes_returned",
         PLAIN_SEARCH_EPISODES_RETURNED),
        ("constants.agent_search_episodes_returned",
         AGENT_SEARCH_EPISODES_RETURNED),
        ("constants.embed_request_bytes", EMBED_REQUEST_BYTES),
        ("constants.embed_response_envelope_bytes",
         EMBED_RESPONSE_ENVELOPE_BYTES),
        ("constants.qdrant_search_request_envelope_bytes",
         QDRANT_SEARCH_REQUEST_ENVELOPE_BYTES),
        ("constants.qdrant_candidates_per_search", QDRANT_CANDIDATES_PER_SEARCH),
        ("constants.qdrant_bytes_per_candidate", QDRANT_BYTES_PER_CANDIDATE),
        ("constants.qdrant_upsert_envelope_bytes", QDRANT_UPSERT_ENVELOPE_BYTES),
        ("constants.qdrant_upsert_response_bytes", QDRANT_UPSERT_RESPONSE_BYTES),
        ("constants.postgres_bytes_per_statement", POSTGRES_BYTES_PER_STATEMENT),
        ("constants.llm_call_request_bytes", LLM_CALL_REQUEST_BYTES),
        ("constants.llm_call_response_bytes", LLM_CALL_RESPONSE_BYTES),
        ("constants.network_protocol_overhead_factor",
         NETWORK_PROTOCOL_OVERHEAD_FACTOR),

        ("constants.human_session_ops_per_s_low", HUMAN_SESSION_OPS_PER_S_LOW),
        ("constants.human_session_ops_per_s_high", HUMAN_SESSION_OPS_PER_S_HIGH),
        # Reported only. Nothing in the machine counts reads either of these.
        ("constants.human_session_ops_per_s_heavy",
         HUMAN_SESSION_OPS_PER_S_HEAVY),
        ("constants.automated_client_ops_per_s", AUTOMATED_CLIENT_OPS_PER_S),
        ("constants.automated_client_ops_per_s_sustained",
         AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED),
        ("constants.ops_per_human_prompt", OPS_PER_HUMAN_PROMPT),

        # The four example defaults that turn a user count into concurrent
        # sessions, and the plausible range of the one that matters. None of
        # them is read unless a user count is given without its own figure.
        ("constants.default_human_peak_active_per_100",
         DEFAULT_HUMAN_PEAK_ACTIVE_PER_100),
        ("constants.human_peak_active_per_100_low",
         HUMAN_PEAK_ACTIVE_PER_100_LOW),
        ("constants.human_peak_active_per_100_high",
         HUMAN_PEAK_ACTIVE_PER_100_HIGH),
        ("constants.default_human_sessions_per_active_user",
         DEFAULT_HUMAN_SESSIONS_PER_ACTIVE_USER),
        ("constants.default_automated_peak_active_per_100",
         DEFAULT_AUTOMATED_PEAK_ACTIVE_PER_100),
        ("constants.default_automated_sessions_per_active_user",
         DEFAULT_AUTOMATED_SESSIONS_PER_ACTIVE_USER),

        # Reported only. Nothing in the machine counts reads any of these.
        ("constants.peak_to_average_single_organisation",
         PEAK_TO_AVERAGE_SINGLE_ORGANISATION),
        ("constants.peak_to_average_measured_small_service",
         PEAK_TO_AVERAGE_MEASURED_SMALL_SERVICE),
        ("constants.peak_to_average_large_service_low",
         PEAK_TO_AVERAGE_LARGE_SERVICE_LOW),
        ("constants.peak_to_average_large_service_high",
         PEAK_TO_AVERAGE_LARGE_SERVICE_HIGH),

        ("constants.server_request_timeout_s", SERVER_REQUEST_TIMEOUT_S),
    ]


def run_validate(mix: TrafficMix, retention_days: float, dims: int,
                 bytes_per_value: float, node_gb=None,
                 out_path: str = NUMBERS_FILE) -> int:
    pairs = list(constant_numbers())
    for name in TIER_ORDER:
        r = size_deployment(TIER_OPS_PER_S[name], mix, retention_days, dims,
                            bytes_per_value, node_gb, run_name=name)
        pairs += published_numbers(name, r)

    def show(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{round_out(value)}"
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    flat = {}
    for key, value in pairs:
        flat[key] = round_out(value)
        print(f"{key}: {show(value)}")

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(flat, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print()
    print(f"wrote {len(flat)} numbers to {out_path}")
    return 0


# =============================================================================
# WEB SERVER
# One HTML form at / and one JSON endpoint at /api/calc. No internet access is
# needed: the page carries its own styling and uses no external files.
# =============================================================================

PAGE_STYLE = """
:root { color-scheme: light; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
       Arial, sans-serif; margin: 0; background: #f6f6f4; color: #16181d; }
main { max-width: 1000px; margin: 0 auto; padding: 24px 20px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 6px; padding-bottom: 4px;
     border-bottom: 2px solid #16181d; }
p.lede, p.note { font-size: 13px; line-height: 1.55; color: #4a4f57;
                 margin: 0 0 12px; }
form { background: #fff; border: 1px solid #d8d8d2; border-radius: 8px;
       padding: 16px; margin: 16px 0 8px; }
.fields { display: flex; flex-wrap: wrap; gap: 14px; }
label { display: block; font-size: 12px; font-weight: 600; margin-bottom: 4px; }
label span { display: block; font-weight: 400; color: #6b7079; font-size: 11px; }
input { font: inherit; font-size: 14px; padding: 6px 8px; width: 110px;
        border: 1px solid #c5c5bd; border-radius: 5px; background: #fff; }
button { font: inherit; font-size: 14px; font-weight: 600; margin-top: 14px;
         padding: 8px 18px; border: 0; border-radius: 5px; background: #16181d;
         color: #fff; cursor: pointer; }
.tablewrap { overflow-x: auto; margin-bottom: 4px; }
table { border-collapse: collapse; width: 100%; background: #fff;
        font-size: 13px; border: 1px solid #d8d8d2; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #ececE6;
         vertical-align: top; }
th { background: #eeeee8; font-weight: 600; }
tr:last-child td { border-bottom: 0; }
td.num { font-variant-numeric: tabular-nums; }
.err { background: #fff1f0; border: 1px solid #e0b4ae; border-radius: 8px;
       padding: 14px 16px; font-size: 14px; color: #8a2a1c; margin: 16px 0; }
footer { margin-top: 40px; font-size: 12px; color: #6b7079; line-height: 1.6; }
code { background: #eeeee8; padding: 1px 5px; border-radius: 4px;
       font-size: 12px; overflow-wrap: anywhere; }
"""

# Every label carries the unit the number is in, because a box labelled with
# a bare noun is read as whatever unit the reader has in mind: "retention" as
# months, "vector dimensions" as vectors, a caller count as accounts.
SIZING_FORM_FIELDS = [
    ("ops", "Design peak, operations/s", "worst rate sustained 5 minutes"),
    ("add", "Adds per 100 operations", "planning assumption"),
    ("plain", "Plain searches per 100 operations", "planning assumption"),
    ("agent", "Agent-mode searches per 100 operations",
     "a request flag, not a caller"),
    ("retention_days", "Retention, days", "undecided - placeholder"),
    ("dims", "Vector dimensions, numbers per vector",
     "fix before first ingest"),
    ("bytes_per_value", "Bytes per number", "1 means int8 quantized"),
    ("node_gb", "RAM per vector-store machine, GB",
     "blank or 'automatic' chooses it"),
]

# The caller population, which is a separate question from the sizing above:
# how many callers there are and how fast each sends, rather than a design
# peak somebody already knows. Both counts may be left blank, and then the
# page asks nothing about a population at all.
# Both counts say "concurrent" in the label and again in the hint, because the
# model multiplies the count by a rate per session: these are the sessions
# running at one moment, and a reader who types their total user count or
# their visitors a day gets an answer wrong by a factor of hundreds.
POPULATION_FORM_FIELDS = [
    ("humans", "Concurrent human chat sessions",
     ("people typing at the same moment - not accounts, not visitors a day; "
      "blank for none")),
    ("automated", "Concurrent automated client sessions",
     ("programs in a tool loop at the same moment - not programs installed; "
      "blank for none")),
    ("human_mix", "Human traffic mix", "adds/plain/agent-mode"),
    ("automated_mix", "Automated client traffic mix",
     "adds/plain/agent-mode"),
]

# The other way to answer the two count boxes above: a user count, and the two
# figures that turn it into concurrent sessions. Each of those four figures has
# an EXAMPLE DEFAULT, so a bare user count always gets an answer. Leave one
# blank beside a filled-in user count and the example is used, marked as an
# example in the table and warned about in a note under it. Every box is blank
# on a first view, and while the user-count boxes are blank the page asks
# nothing about a conversion at all.
USERS_FORM_FIELDS = [
    ("human_users", "People in the user base",
     "how many people could use it - not sessions; blank to skip"),
    ("human_peak_share",
     "Share of people active at the busiest moment, per 100 users",
     (f"blank uses {DEFAULT_HUMAN_PEAK_ACTIVE_PER_100:g}, an example - a "
      "convention, not a measurement")),
    ("human_sessions_per_active_user", "Sessions per active person",
     (f"blank uses {DEFAULT_HUMAN_SESSIONS_PER_ACTIVE_USER:g}, an example - "
      "no published evidence exists")),
    ("automated_users", "Users who run automated clients",
     "how many people run them - not sessions; blank to skip"),
    ("automated_peak_share",
     "Share of those users active at the busiest moment, per 100 users",
     (f"blank uses {DEFAULT_AUTOMATED_PEAK_ACTIVE_PER_100:g}, an example - a "
      "running client is already a session")),
    ("automated_sessions_per_active_user",
     "Automated client sessions per active user",
     (f"blank uses {DEFAULT_AUTOMATED_SESSIONS_PER_ACTIVE_USER:g}, an example "
      "- often 10 or 50 for a person running a framework")),
]

# Every box on the page, in the order it is drawn.
FORM_FIELDS = SIZING_FORM_FIELDS + POPULATION_FORM_FIELDS + USERS_FORM_FIELDS

# What a reader may type in the "RAM per vector-store machine" box to ask for
# the automatic choice, as well as leaving it empty.
NODE_GB_AUTOMATIC_WORDS = ("auto", "automatic")

# What each box is called in an error message, and what belongs in it. A
# message that says "every field must be a number" leaves the reader to hunt
# through eight boxes, so every message below names one box and quotes what was
# typed into it.
FORM_FIELD_HELP = {
    "ops": ("design peak",
            "the operations per second you need to serve"),
    "add": ("adds per 100 operations",
            "how many of every 100 operations are adds"),
    "plain": ("plain searches per 100 operations",
              "how many of every 100 operations are plain searches"),
    "agent": ("agent-mode searches per 100 operations",
              "how many of every 100 operations are agent-mode searches"),
    "retention_days": ("retention, days",
                       "how many days an episode is kept before it is deleted"),
    "dims": ("vector dimensions", "how many numbers one vector holds"),
    "bytes_per_value": ("bytes per number",
                        "how many bytes are stored for each number"),
    "node_gb": ("RAM per vector-store machine",
                ("the GB of RAM one machine has, or nothing at all to have "
                 "the size chosen for you")),
    "humans": ("concurrent human chat sessions",
               ("how many people are typing at the same moment - not how many "
                "accounts there are and not how many visit in a day - or "
                "nothing at all if you are not sizing from a population")),
    "automated": ("concurrent automated client sessions",
                  ("how many programs are sending requests in a loop at the "
                   "same moment, or nothing at all if you are not sizing "
                   "from a population")),
    "human_mix": ("human traffic mix",
                  ("how the human sessions' 100 operations split, as "
                   f"adds/plain/agent-mode, such as {MIX_TRIPLE_EXAMPLE}")),
    "automated_mix": ("automated client traffic mix",
                      ("how the automated clients' 100 operations split, "
                       "written the same way")),
    "human_users": ("people in the user base",
                    ("how many people could use the service - a user count, "
                     "not a session count - or nothing at all if you are "
                     "giving the concurrent sessions instead")),
    "human_peak_share": ("share of people active at the busiest moment",
                         ("how many of every 100 of those people are using "
                          "the service at the busiest moment")),
    "human_sessions_per_active_user":
        ("sessions per active person",
         ("how many sessions one of those active people holds at that "
          "moment, usually about 1")),
    "automated_users": ("users who run automated clients",
                        ("how many people run automated clients - a user "
                         "count, not a session count - or nothing at all if "
                         "you are giving the concurrent sessions instead")),
    "automated_peak_share":
        ("share of those users active at the busiest moment",
         ("how many of every 100 of them are running automated clients at "
          "the busiest moment")),
    "automated_sessions_per_active_user":
        ("automated client sessions per active user",
         ("how many automated client sessions one of them holds at that "
          "moment, often 10 or 50")),
}

# The four inputs of each caller kind, named the way the web form names them,
# so that a refusal on the page talks about boxes and the same refusal on the
# command line talks about flags.
WEB_HUMAN_NAMES = ConversionNames(
    sessions=f"the {FORM_FIELD_HELP['humans'][0]} box",
    users=f"the {FORM_FIELD_HELP['human_users'][0]} box",
    share=f"the {FORM_FIELD_HELP['human_peak_share'][0]} box",
    per_user=f"the {FORM_FIELD_HELP['human_sessions_per_active_user'][0]} box")
WEB_AUTOMATED_NAMES = ConversionNames(
    sessions=f"the {FORM_FIELD_HELP['automated'][0]} box",
    users=f"the {FORM_FIELD_HELP['automated_users'][0]} box",
    share=f"the {FORM_FIELD_HELP['automated_peak_share'][0]} box",
    per_user=(
        f"the {FORM_FIELD_HELP['automated_sessions_per_active_user'][0]} box"))

# What each box holds when it is not sent at all - a bare /api/calc call, or a
# first visit to the page. None in the node_gb box means "choose the size".
FORM_DEFAULTS = {
    "ops": TIER_OPS_PER_S["target"],
    "add": DEFAULT_MIX_ADD,
    "plain": DEFAULT_MIX_PLAIN,
    "agent": DEFAULT_MIX_AGENT,
    "retention_days": DEFAULT_RETENTION_DAYS,
    "dims": DEFAULT_VECTOR_DIMS,
    "bytes_per_value": DEFAULT_BYTES_PER_VALUE,
    "node_gb": None,
    # A population is an optional second question. Both counts start empty,
    # and while they are both empty the page says nothing about a population.
    "humans": None,
    "automated": None,
    "human_mix": default_mix_text(),
    "automated_mix": default_mix_text(),
    # The other way to answer the two counts above. All six start empty. The
    # two shares and the two sessions-per-user figures do have example
    # defaults now, but the BOX still starts empty: the default is applied
    # when a user count is given without it, and the report marks it as an
    # example rather than as something the reader typed.
    "human_users": None,
    "human_peak_share": None,
    "human_sessions_per_active_user": None,
    "automated_users": None,
    "automated_peak_share": None,
    "automated_sessions_per_active_user": None,
}

# The id of the error message, so that a box can point at it with
# aria-describedby and a screen reader reads the two together.
FORM_ERROR_ID = "form-error"


class FieldError(SizingError):
    """A bad input that knows which box on the form it came from."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def html_sections(sections: list) -> list:
    """One report section per heading and table, as HTML fragments.

    The sizing report and the population report are drawn by this one
    function, so a table can never look one way in the first and another way
    in the second.
    """
    parts = []
    for section in sections:
        parts.append(f"<h2>{escape(section['title'])}</h2>")
        if section["note"]:
            parts.append(f"<p class=\"note\">{escape(section['note'])}</p>")
        parts.append('<div class="tablewrap"><table><thead><tr>')
        parts.extend(f"<th>{escape(head)}</th>" for head in section["headers"])
        parts.append("</tr></thead><tbody>")
        for row in section["rows"]:
            parts.append("<tr>")
            for idx, cell in enumerate(row):
                css = ' class="num"' if idx > 0 else ""
                parts.append(f"<td{css}>{escape(str(cell))}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
    return parts


def sizing_source_note(values: dict, result: dict) -> str:
    """Which boxes produced the machine counts, in one plain paragraph.

    The form asks for the same traffic twice - a design peak with a mix, and a
    caller population - and only one of the two can size the machines. Nobody
    should have to guess which, or work out from the tables whether the number
    they typed into a box was used, so the page says it in words beside the
    answer.
    """
    population = result.get("population")
    peak = f"{as_given(result['inputs']['ops_per_s'])} operations/s"
    mix = TrafficMix(**result["inputs"]["mix"]).as_words()
    if population is None:
        return (f"Sized from the design peak box - {peak} - and the traffic "
                f"mix boxes: {mix} per 100 operations. Both caller-count "
                "boxes are empty, so no caller population was worked out. "
                "Fill in either count and the population sizes the deployment "
                "instead.")
    if population["sizing"] is None:
        # Zero callers is a question with an answer - "nobody demands nothing"
        # - and the tables below still print it. What it is not is a
        # deployment: sizing from it would order zero machines, which nobody
        # can buy. So the design peak box governs, and this says so rather
        # than leaving the reader to work out which of the two was used.
        return ("This population sends nothing at all: "
                f"{as_given(population['humans'])} concurrent human chat "
                f"sessions and {as_given(population['automated'])} concurrent "
                "automated client sessions demand 0 operations/s, and no "
                "machine count can be worked out from no traffic. The tables "
                "below are sized from the design peak box instead - "
                f"{peak} - with the traffic mix boxes: {mix} per 100 "
                "operations. The population tables report the zero demand and "
                "order nothing.")
    typed = str(values.get("ops", "")).strip()
    # Quoted as typed and escaped where it is drawn, never here: this sentence
    # is also carried in the JSON, where HTML entities would be noise.
    typed_says = f'says "{typed}"' if typed else "was left empty"
    # A population worked out from a user count has one more step behind it,
    # and the reader supplied the two figures that made it. Say so, so that
    # nobody reads the session counts as numbers they typed.
    converted = ""
    # Advice that fits the boxes the reader actually filled in: telling
    # somebody to clear a caller-count box they left empty helps nobody.
    clear_advice = "Clear both caller-count boxes to size from them instead."
    if population.get("conversion") is not None:
        clear_advice = ("Clear the two caller-count boxes and the six user "
                        "boxes to size from them instead.")
        converted = (
            " Those session counts were not typed in: they were worked out "
            "from the user counts, the share active at the busiest moment and "
            "the sessions per active user, all of which you gave. The first "
            "table below shows that multiplication.")
    return (
        "Sized from the caller population, not from the design peak box. "
        f"{as_given(population['humans'])} concurrent human chat sessions and "
        f"{as_given(population['automated'])} concurrent automated client "
        f"sessions demand {peak} at the busy end of the human rate, and their "
        f"blended mix is {mix} per 100 operations. Those are the numbers the "
        "tables below are sized for, and they are the same numbers the "
        "command line gives for the same population. The design peak box, "
        f"which {typed_says}, and the three traffic-mix boxes were not used. "
        + clear_advice + converted)


def render_html(values: dict, result: dict | None, error: str | None,
                bad_field: str | None = None) -> str:
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>MemMachine sizing calculator</title>",
        f"<style>{PAGE_STYLE}</style></head><body><main>",
        "<h1>MemMachine sizing calculator</h1>",
        ('<p class="lede">Set the design peak, the traffic mix and the '
         "retention period, and this page gives the machine counts, storage, "
         "PostgreSQL connections and network peaks. Every number in the tables "
         "of findings below is labelled measured, derived, estimate or "
         'assumption. Two tables are different - "Qdrant node choice" and the '
         "sensitivity table show what-ifs rather than findings, and the note "
         "above each says where its numbers come from. The traffic mix is a "
         "planning assumption that nobody has measured.</p>"
         '<p class="lede">Two things here sound alike and are not. '
         "<strong>Agent-mode search</strong> is how a request behaves: one "
         "search that fans out into about 22. <strong>Automated clients</strong> "
         "are callers: programs that send requests in a loop, about 0.4 "
         "operations a second each, where a person sends 0.011 to 0.028. One "
         "is how expensive a request is, the other is how fast a caller sends "
         "them, and a caller of either kind can send requests of either "
         "kind.</p>"),
    ]
    # The message goes above the form, carries role="alert" so a screen reader
    # announces it, and the box it blames takes the focus - so submitting with
    # the keyboard lands the reader on the box that has to change.
    if error:
        parts.append(f'<div class="err" id="{FORM_ERROR_ID}" role="alert">'
                     f"<strong>Cannot calculate.</strong> {escape(error)}</div>")
    parts.append('<form method="get" action="/">')

    def draw_boxes(fields):
        parts.append('<div class="fields">')
        for key, title, hint in fields:
            val = escape(str(values.get(key, "")))
            flags = ""
            if error and key == bad_field:
                flags = (' aria-invalid="true" '
                         f'aria-describedby="{FORM_ERROR_ID}" autofocus')
            parts.append(
                f'<div><label for="{key}">{escape(title)}'
                f"<span>{escape(hint)}</span>"
                f'</label><input id="{key}" name="{key}" value="{val}" '
                f'type="text" inputmode="decimal"{flags}></div>')
        parts.append("</div>")

    draw_boxes(SIZING_FORM_FIELDS)
    parts.append(
        '<p class="note">A caller population, if you would rather start from '
        "how many callers there are than from a rate. Both counts are "
        "sessions running at the same moment - not registered accounts, and "
        "not visitors in a day - because each one is multiplied by a rate per "
        "session. Fill in either count and the population sizes the "
        "deployment: the design peak box and the three mix boxes above are "
        "then not used, and the page says so under the button. Leave both "
        "empty and this part is skipped. An automated client is a caller: a "
        "program that sends requests in a loop. It is not the same thing as "
        "agent-mode search above, which is a flag on one request. Each "
        "population carries its own mix, written as adds/plain/agent-mode.")
    draw_boxes(POPULATION_FORM_FIELDS)
    parts.append(
        '<p class="note">Or start from a user count. The two count boxes '
        "above are sessions running at the same moment, and whoever asks for "
        "a deployment usually answers in users instead. Fill these in rather "
        "than the count boxes above, and the page works the sessions out. Two "
        "figures do that, and each has an <strong>example default</strong> "
        "you can leave blank, so a user count on its own always gets an "
        f"answer. {escape(COUNTING_RULE)} The share active at the busiest "
        f"moment defaults to {DEFAULT_HUMAN_PEAK_ACTIVE_PER_100:g} per 100 "
        "people, which is a convention rather than a measurement and can be "
        f"wrong by a factor of two either way; the sessions one active person "
        f"holds defaults to "
        f"{DEFAULT_HUMAN_SESSIONS_PER_ACTIVE_USER:g}, for which no published "
        "evidence exists. Both automated boxes default to identity, because a "
        "client that is running is already a session. Every default is marked "
        "as an example in the table and the page says what to replace it "
        "with. Give a user count or give the count box above; giving both for "
        "the same kind of caller is refused, because they answer the same "
        "question two different ways.")
    draw_boxes(USERS_FORM_FIELDS)
    parts.append('<button type="submit">Calculate</button></form>')
    # Directly under the form and above every table: which boxes sized the
    # machines, and which were not used.
    if result is not None:
        note = result.get("sized_from_note") or sizing_source_note(values,
                                                                   result)
        parts.append(f'<p class="note"><strong>What sized these machines.'
                     f"</strong> {escape(note)}</p>")
    if result is not None and result.get("population") is not None:
        parts.extend(html_sections(population_sections(result["population"])))
    if result is not None:
        parts.extend(html_sections(report_sections(result)))

    parts.append(
        "<footer><p><strong>Labels.</strong> measured = from a real test, named "
        "by its date, its configuration, or both. derived = computed by this "
        "program from measured "
        "numbers. estimate = not measured on this system; where a published "
        "source stands behind it, the label names it; benchmark before "
        "ordering hardware. assumption = a planning choice, not a finding. "
        "The Qdrant "
        "node choice table and the sensitivity table show what-ifs rather than "
        "findings, so their rows carry no label; the note above each says "
        "where its numbers come from.</p>"
        "<p>The JSON below carries the same numbers without labels. Its "
        "<code>run_name</code> field is the name of the run, not one of the "
        "four labels above.</p>"
        "<p>The same figures as JSON: "
        "<code>/api/calc?ops=100&amp;add=45&amp;plain=45&amp;agent=10"
        "&amp;retention_days=90&amp;dims=1024&amp;bytes_per_value=1</code>. "
        "Add <code>&amp;node_gb=512</code> to force the size of a "
        "vector-store machine; leave it out and the size is chosen "
        "automatically.</p>"
        "<p>To size from a caller population instead, add "
        "<code>&amp;humans=5000&amp;automated=40</code>, and "
        "<code>&amp;human_mix=48/50/2&amp;automated_mix=20/20/60</code> to "
        "give each population its own traffic mix. <code>automated</code> "
        "counts callers that are programs; <code>agent</code> above is the "
        "agent-mode share of the requests themselves.</p>"
        "<p>To start from a user count instead of a session count, add "
        "<code>&amp;human_users=50000&amp;human_peak_share=2"
        "&amp;human_sessions_per_active_user=1</code>, and the same three "
        "with <code>automated_</code> in front for automated clients. The "
        "share and the sessions per active user may be left out: each has an "
        "example default, and <code>&amp;human_users=50000</code> on its own "
        "answers.</p>"
        "<p>GB means 10<sup>9</sup> bytes, TB means 10<sup>12</sup> bytes and "
        "Mbps means 10<sup>6</sup> bits per second throughout.</p></footer>")
    parts.append("</main></body></html>")
    return "".join(parts)


def parse_node_gb(raw):
    """Read the "RAM per vector-store machine" box.

    Empty, or one of the words in NODE_GB_AUTOMATIC_WORDS, means the program
    chooses the size itself, exactly as it does when --node-gb is not given.
    This is the one box where empty is an answer rather than a blank, and the
    hint under it says so. Anything else must be a number; size_deployment
    then refuses it if it is not greater than zero, so a bad value becomes the
    same reported error as any other bad field rather than a silent fall back
    to the automatic choice.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in NODE_GB_AUTOMATIC_WORDS:
        return None
    return read_number(text, "node_gb")


def format_default(value) -> str:
    """A default as it should read in a form box.

    A whole number reads as a whole number: the design peak default shows as
    "100" and not "100.0", so the first view of the page matches every view
    after a submit. A default that is already text - a traffic mix written as
    45/45/10 - is shown as it stands.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):g}"


def read_number(text: str, key: str) -> float:
    """One box's text as a number, or a FieldError that names that box."""
    name, what = FORM_FIELD_HELP[key]
    typed = str(text).strip()
    if typed == "":
        raise FieldError(key, f"the {name} box is empty - type {what}")
    try:
        return float(typed)
    except (TypeError, ValueError, OverflowError):
        if "," in typed:
            raise FieldError(
                key,
                f'the {name} box says "{typed}", which is not a number - '
                "type digits only, with no comma between the thousands"
            ) from None
        raise FieldError(
            key,
            f'the {name} box says "{typed}", which is not a number - '
            f"type {what}") from None


def submitted_text(query: dict, key: str):
    """The text submitted for one box, or None when the box was not sent.

    An empty box and a box that was never sent are different things. A bare
    /api/calc call sends no boxes at all and takes every default, which is how
    the command line behaves when a flag is left off. A form submission always
    sends every box, so an empty one is a blank answer, and blanking the
    design peak must not quietly become a plan for 100 operations per second.
    """
    sent = query.get(key)
    if not sent:
        return None
    return sent[0]


def unknown_parameter_error(query: dict):
    """Complain about a web address this calculator cannot honour, or None.

    A misspelled parameter used to be ignored in silence, so
    /api/calc?ops=100&retention_day=1 answered confidently for the 90-day
    default - a hardware order for a question nobody asked. A parameter sent
    twice was quietly first-wins for the same reason.
    """
    for key in query:
        if key not in FORM_DEFAULTS:
            # "agents" is the one wrong spelling that must not be answered
            # with "did you mean agent?", because agent is the mix share and
            # the reader almost certainly meant the count of callers that are
            # programs. It is named against the setting they want instead.
            if key.lower() == "agents":
                return RETIRED_AGENTS_SETTING_MESSAGE
            known = ", ".join(sorted(FORM_DEFAULTS))
            # Matched in lower case: the settings are all lower case, and the
            # commonest miss is the right word in the wrong case. "Ops" scores
            # 0.67 against "ops" and "OPS" scores 0, both below the cutoff, so
            # neither used to get the one suggestion that would help.
            near = difflib.get_close_matches(key.lower(), FORM_DEFAULTS, n=1,
                                             cutoff=0.7)
            suggestion = (f' Did you mean "{near[0]}"?' if near else
                          f" The settings it accepts are: {known}.")
            return (f'the web address has a setting called "{key}", which '
                    f"this calculator does not know.{suggestion}")
    for key, sent in query.items():
        if len(sent) > 1:
            return (f'the web address sets "{key}" {len(sent)} times. Set it '
                    "once, so it is clear which value you meant.")
    return None


def box_number_or_none(values: dict, key: str):
    """One box as a number, or None when it is empty.

    Empty means "not given", which is what a flag left off the command line
    means. Every box this reads is one a run may legitimately leave blank.
    """
    text = str(values.get(key, "")).strip()
    if text == "":
        return None
    return read_number(text, key)


def caller_kind_from_boxes(values: dict, names: ConversionNames,
                           defaults: ConversionDefaults,
                           sessions_key: str, users_key: str, share_key: str,
                           per_user_key: str) -> tuple:
    """One caller kind's concurrent sessions from the page's own boxes.

    Two ways in and the same rules as the command line, because it is the
    same function underneath: give the sessions box, or give the user box
    with both conversion boxes. The refusals name boxes here and flags there.
    """
    return sessions_for_caller_kind(
        names,
        defaults,
        box_number_or_none(values, sessions_key),
        box_number_or_none(values, users_key),
        box_number_or_none(values, share_key),
        box_number_or_none(values, per_user_key))


def population_boxes(values: dict):
    """What the population boxes say, or None if no population was named.

    Returns (humans, automated, human_mix, automated_mix, human_conversion,
    automated_conversion) - the same six things --humans, --automated,
    --human-mix, --automated-mix and the two sets of user flags give the users
    subcommand, so the two interfaces read one population the same way.

    The two count boxes and the two user-count boxes are the switch. Leave all
    four empty and the page is only a sizing calculator, exactly as it was
    before it could take a population; fill in any of them and an empty one
    counts as none of that kind of caller, which is what --automated
    defaulting to 0 does on the command line. The two mix boxes are always
    answered, because each has a default on the page, and each is read the
    same way as --human-mix and --automated-mix.
    """
    # A conversion box counts as naming a population too, so that a share
    # active typed on its own is refused for having no user count rather than
    # quietly ignored.
    named = [str(values.get(key, "")).strip()
             for key in ("humans", "automated", "human_users",
                         "automated_users", "human_peak_share",
                         "human_sessions_per_active_user",
                         "automated_peak_share",
                         "automated_sessions_per_active_user")]
    if not any(named):
        return None
    humans, human_conversion = caller_kind_from_boxes(
        values, WEB_HUMAN_NAMES, HUMAN_CONVERSION_DEFAULTS, "humans",
        "human_users", "human_peak_share",
        "human_sessions_per_active_user")
    automated, automated_conversion = caller_kind_from_boxes(
        values, WEB_AUTOMATED_NAMES, AUTOMATED_CONVERSION_DEFAULTS,
        "automated", "automated_users",
        "automated_peak_share", "automated_sessions_per_active_user")
    human_mix = read_mix_box(values, "human_mix")
    automated_mix = read_mix_box(values, "automated_mix")
    return (0.0 if humans is None else humans,
            0.0 if automated is None else automated,
            human_mix, automated_mix, human_conversion, automated_conversion)


def store_shape_boxes(values: dict) -> tuple:
    """The four boxes that describe the store rather than the traffic.

    Retention, vector dimensions, bytes per number and the RAM of one
    vector-store machine. They are read the same way whether the deployment is
    sized from a design peak or from a caller population, because a population
    says how much traffic there is and nothing at all about how long an
    episode is kept or how wide a vector is.
    """
    return (read_number(values["retention_days"], "retention_days"),
            read_number(values["dims"], "dims"),
            read_number(values["bytes_per_value"], "bytes_per_value"),
            parse_node_gb(values["node_gb"]))


def design_peak_boxes(values: dict) -> tuple:
    """The design peak box and the three traffic-mix boxes, as (ops, mix)."""
    return (read_number(values["ops"], "ops"),
            TrafficMix(read_number(values["add"], "add"),
                       read_number(values["plain"], "plain"),
                       read_number(values["agent"], "agent")))


def read_mix_box(values: dict, key: str) -> TrafficMix:
    """One traffic-mix box as a mix, named by its own label when it is wrong."""
    name, what = FORM_FIELD_HELP[key]
    text = str(values.get(key, "")).strip()
    if text == "":
        raise FieldError(key, f"the {name} box is empty - type {what}")
    try:
        return parse_mix_text(text, f"the {name} box")
    except SizingError as exc:
        raise FieldError(key, str(exc)) from None


def result_from_query(query: dict) -> tuple:
    """Return (values_for_the_form, result, error, the_box_the_error_blames).

    result is None when there is an error, and error is None when there is a
    result. The fourth item names the box to mark invalid on the page, and is
    None when the fault is not one box's alone - a traffic mix that does not
    add up to 100, for instance.

    The form asks for the same traffic in two ways - a design peak with a
    traffic mix, and a caller population - and only one of them can size the
    machines. The rule, which sizing_source_note says on the page in so many
    words:

      no population named   the design peak box and the mix boxes size the
                            deployment, exactly as they always have
      a population named    the population sizes it: the design peak is the
                            demand it makes at the busy end of the human rate
                            and the mix is its blended mix, which is what the
                            users subcommand does with the same numbers. The
                            design peak box and the three mix boxes are not
                            read at all
      a population that
      makes no requests     nothing to size from it, so the design peak box
                            sizes the deployment after all

    The page used to work out the blended mix, print it, and then size the
    machines from the mix boxes anyway, so the same population ordered
    different hardware on the page and on the command line.
    """
    values = {}
    for key, default in FORM_DEFAULTS.items():
        raw = submitted_text(query, key)
        values[key] = format_default(default) if raw is None else raw
    # No box on the page is at fault for a bad web address, so nothing is
    # marked invalid: the fourth item stays None.
    bad_address = unknown_parameter_error(query)
    if bad_address is not None:
        return values, None, bad_address, None
    try:
        # The population boxes are read first because they decide what the
        # rest of the form means. When they name a population, the design peak
        # box and the mix boxes are never read, so blanking a box this run
        # does not use is not an error.
        boxes = population_boxes(values)
        if boxes is None:
            ops, mix = design_peak_boxes(values)
            retention, dims, bpv, node_gb = store_shape_boxes(values)
            result = size_deployment(ops, mix, retention, dims, bpv, node_gb,
                                     node_gb_source=NODE_GB_SOURCE_WEB,
                                     run_name="web")
            result["population"] = None
        else:
            humans, automated, hmix, amix, hconv, aconv = boxes
            retention, dims, bpv, node_gb = store_shape_boxes(values)
            population = ops_for_population(
                humans, automated, hmix, amix, retention_days=retention,
                dims=dims, bytes_per_value=bpv, node_gb=node_gb,
                node_gb_source=NODE_GB_SOURCE_WEB, run_name="web",
                human_conversion=hconv, automated_conversion=aconv)
            if population["sizing"] is None:
                # Nobody sending anything is not a deployment to size, so the
                # design peak box has work to do after all.
                ops, mix = design_peak_boxes(values)
                result = size_deployment(ops, mix, retention, dims, bpv,
                                         node_gb,
                                         node_gb_source=NODE_GB_SOURCE_WEB,
                                         run_name="web")
            else:
                # A copy, so that the result can carry the population that
                # produced it without the population carrying the result back:
                # a loop like that cannot be written out as JSON.
                result = dict(population["sizing"])
            result["population"] = population
        # Carried in the result rather than written on the page alone, so that
        # a reader of /api/calc is told which boxes were used just as plainly
        # as a reader of the page.
        result["sized_from_note"] = sizing_source_note(values, result)
    except FieldError as exc:
        return values, None, str(exc), exc.field
    except (SizingError, ValueError, OverflowError, ArithmeticError) as exc:
        return values, None, str(exc), None
    return values, result, None, None


class SizingHandler(BaseHTTPRequestHandler):
    server_version = "MemMachineSizing/1.0"

    # socketserver applies this to the connection with settimeout(), so a
    # client that opens a connection and then sends nothing is dropped instead
    # of holding a thread and a file descriptor for as long as it likes.
    # BaseHTTPRequestHandler already treats the timeout as "close and stop".
    timeout = SERVER_REQUEST_TIMEOUT_S

    def do_GET(self):
        # Belt and braces: a caller must always get an HTTP response, never a
        # dropped connection, whatever a future change to the model raises.
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionResetError):
            # The reader closed the tab or hit stop. That is ordinary, not an
            # internal error, and there is no longer a socket to answer on.
            pass
        except Exception as exc:
            sys.stderr.write(f"unhandled error serving {self.path}: {exc}\n")
            try:
                self._send(500, "text/plain; charset=utf-8",
                           b"internal error while sizing this request\n")
            except OSError:
                pass

    def _handle_get(self):
        parsed = urlparse(self.path)
        # keep_blank_values matters: without it "?ops=" would arrive looking
        # exactly like a request that never mentioned ops at all, and an empty
        # box on the form would silently take the default.
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path in ("/", "/index.html"):
            values, result, error, bad_field = result_from_query(query)
            body = render_html(values, result, error, bad_field).encode("utf-8")
            self._send(200 if error is None else 400, "text/html; charset=utf-8",
                       body)
        elif parsed.path == "/favicon.ico":
            # Every browser asks for a tab icon on every page load. Answering
            # 404 puts an error in the console of an otherwise clean page, so
            # say plainly that there is no icon and nothing went wrong.
            self._send_no_content()
        elif parsed.path == "/api/calc":
            _, result, error, _bad_field = result_from_query(query)
            if error is not None:
                payload = json.dumps({"error": error}, indent=2).encode("utf-8")
                self._send(400, "application/json; charset=utf-8", payload)
            else:
                payload = json.dumps(result, indent=2, default=str).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", payload)
        elif parsed.path == "/healthz":
            self._send(200, "text/plain; charset=utf-8", b"ok\n")
        else:
            self._send(404, "text/plain; charset=utf-8",
                       b"not found - try / or /api/calc\n")

    def _send_no_content(self) -> None:
        """204 No Content: a real answer with nothing in it, and no error."""
        self.send_response(204)
        self.end_headers()

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # fmt % args is http.server's own calling convention for this hook.
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")


def checked_port(port: int) -> int:
    """Refuse a port the operating system cannot bind, by name."""
    if not 0 < port < 65536:
        raise SizingError(f"port {port} is not between 1 and 65535")
    return port


def run_server(host: str, port: int) -> int:
    httpd = ThreadingHTTPServer((host, port), SizingHandler)
    print(f"sizing calculator on http://{host}:{port}/  "
          f"(JSON at http://{host}:{port}/api/calc)  press Ctrl-C to stop")
    print("this is a local development server: no authentication, no rate "
          "limiting, not for an address the public can reach")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


# =============================================================================
# COMMAND LINE
# =============================================================================


def render_users_report(pop: dict) -> str:
    out = []
    title = "Caller population to required capacity"
    out.append("=" * 80)
    out.append(title)
    out.append("=" * 80)
    for section in population_sections(pop):
        out.append("")
        out.append(section["title"])
        out.append("-" * len(section["title"]))
        if section["note"]:
            out.append(wrap(section["note"]))
            out.append("")
        out.append(render_table(section["headers"], section["rows"]))
    out.append("")
    if pop["ops_per_s_high"] <= 0:
        out.append(wrap(
            "This population makes no requests at all, so there is nothing to "
            "size and no tier to name. Count the callers who will actually be "
            "using it and ask again."))
        return "\n".join(out)
    recommended = pop["tier_for_high"] or "none - above the scale tier"
    out.append(f"  Plan for the high rate: {recommended}")
    out.append("")
    out.append(wrap(
        "The same population of callers can demand a pilot tier or a scale "
        "tier depending on how many of them are automated clients rather than "
        "people, and on how much agent-mode search each population does. Meter "
        "real operations per second per API key from the first day of the "
        "pilot and re-check this answer against the metered figure."))
    return "\n".join(out)


def render_tier_headline(r: dict) -> str:
    m = r["machines"]
    embed = (f"{m['embed_gpu_cards_low']}"
             if m["embed_gpu_cards_low"] == m["embed_gpu_cards_high"]
             else f"{m['embed_gpu_cards_low']} to {m['embed_gpu_cards_high']}")
    return wrap(
        f"In words: {m['api_servers']} API server(s), {m['postgres_servers']} "
        f"PostgreSQL server(s) and {m['qdrant_servers']} Qdrant server(s) of "
        f"{num(m['qdrant_node_ram_gb'])} GB RAM each, plus {embed} embedding GPU "
        f"card(s) and {m['agent_gpu_cards']} agent-model GPU card(s), both "
        "counts including one spare.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memmachine_sizing.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "MemMachine deployment sizing calculator. Turns a design peak in\n"
            "operations per second into machine counts, storage, PostgreSQL\n"
            "connections and network peaks. Read the module docstring for the\n"
            "full list of inputs and where each one came from."),
        epilog=(
            "Examples:\n"
            "  memmachine_sizing.py tier target\n"
            "  memmachine_sizing.py calc --ops 250 --agent 4 --plain 51 --json\n"
            "  memmachine_sizing.py users --humans 5000 --automated 40\n"
            "  memmachine_sizing.py users --humans 5000 --automated 200 "
            "--human-mix 48/50/2 --automated-mix 20/20/60\n"
            "  memmachine_sizing.py users --human-users 50000\n"
            "  memmachine_sizing.py users --human-users 50000 "
            "--human-peak-share 2 --human-sessions-per-active-user 1\n"
            "  memmachine_sizing.py validate --out sizing-numbers.json\n"
            "  memmachine_sizing.py serve --port 8899\n"))
    subs = parser.add_subparsers(dest="command", metavar="<subcommand>")

    def add_mix_options(sub, include_shape=True):
        sub.add_argument("--add", type=float, default=DEFAULT_MIX_ADD,
                         help="adds per 100 operations (default: %(default)s; "
                              "planning assumption, never measured)")
        sub.add_argument("--plain", type=float, default=DEFAULT_MIX_PLAIN,
                         help="plain searches per 100 operations "
                              "(default: %(default)s)")
        sub.add_argument("--agent", type=float, default=DEFAULT_MIX_AGENT,
                         help="agent-mode searches per 100 operations "
                              "(default: %(default)s; a request flag, not a "
                              "kind of caller, and the biggest single lever "
                              "on the hardware order)")
        # Refused by name in main. Hidden, because it is not a flag any more:
        # without it, --agents here would only ever be an unrecognized
        # argument, and the reader would never be told that the flag they
        # want is --automated on the users subcommand.
        sub.add_argument("--agents", dest="retired_agents", nargs="?",
                         const="", default=None, help=argparse.SUPPRESS)
        if include_shape:
            sub.add_argument("--retention-days", type=float,
                             default=DEFAULT_RETENTION_DAYS,
                             help="days an episode is kept before deletion "
                                  "(default: %(default)s; a placeholder, "
                                  "retention is undecided)")
            sub.add_argument("--dims", type=int, default=DEFAULT_VECTOR_DIMS,
                             help="numbers per vector (default: %(default)s)")
            sub.add_argument("--bytes-per-value", type=float,
                             default=DEFAULT_BYTES_PER_VALUE,
                             help="bytes stored per number (default: "
                                  "%(default)s, meaning int8 quantized)")
            sub.add_argument("--node-gb", type=float, default=None,
                             help="RAM per vector-store machine in GB "
                                  "(default: chosen automatically; pass 256, "
                                  "512 or 768 to force a shape)")

    tier = subs.add_parser(
        "tier", help="full report for one tier",
        description="Print the full sizing report for one named tier as an "
                    "aligned text table, including the agent-mode sensitivity "
                    "table.")
    tier.add_argument("name", choices=TIER_ORDER,
                      help="pilot (20 ops/s), target (100 ops/s) or scale "
                           "(1,000 ops/s)")
    add_mix_options(tier)
    tier.add_argument("--json", action="store_true",
                      help="print the raw result as JSON instead of a table")

    calc = subs.add_parser(
        "calc", help="full report for any operations-per-second rate",
        description="Size an arbitrary point: any design peak, any traffic mix, "
                    "any retention period and any vector shape.")
    calc.add_argument("--ops", type=float, required=True,
                      help="design peak in operations per second")
    add_mix_options(calc)
    calc.add_argument("--json", action="store_true",
                      help="print the raw result as JSON instead of a table")

    users = subs.add_parser(
        "users", help="convert a caller population into required capacity",
        description="Convert a population of human chat sessions and "
                    "automated clients into the operations per second they "
                    "demand, blend their two traffic mixes, name the smallest "
                    "tier that holds them and size the machines that mix "
                    "needs. An automated client is a caller - a program "
                    "sending requests in a loop. It is not agent-mode search, "
                    "which is a flag on one request. " + COUNTING_RULE + " "
                    + COUNTING_RULE_FLAGS + " Or give a user count with "
                    "--human-users or --automated-users, and the two figures "
                    "that turn it into concurrent sessions - the share of "
                    "those users active at the busiest moment and the "
                    "sessions one active user holds - are filled in from "
                    "example defaults if you leave them out. Those defaults "
                    "are examples, they are labelled as examples in the "
                    "report, and the report says what to replace them "
                    "with.")
    users.add_argument("--humans", type=float, default=None,
                       help="concurrent human chat sessions: people typing "
                            "at the same moment - not how many accounts "
                            "there are and not how many visit in a day. Give "
                            "this or --human-users, not both")
    users.add_argument("--automated", type=float, default=None,
                       help="concurrent automated client sessions: programs "
                            "sending requests in a 5-second tool loop at the "
                            "same moment (default: 0, meaning none). Give "
                            "this or --automated-users, not both")
    users.add_argument("--human-users", type=float, default=None,
                       help="how many people could use the service: a user "
                            "count, not a session count. Give "
                            "--human-peak-share and "
                            "--human-sessions-per-active-user with it, or "
                            "leave either out and take its example default")
    users.add_argument("--human-peak-share", type=float, default=None,
                       help="how many of every 100 of those people are using "
                            "the service at the busiest moment (default: "
                            f"{DEFAULT_HUMAN_PEAK_ACTIVE_PER_100:g}, an "
                            "EXAMPLE. It is a convention, not a measurement: "
                            "Microsoft's SharePoint capacity guidance assumes "
                            "10 percent concurrency, and teletraffic "
                            "engineering uses 10-16%% of subscribers busy "
                            "in "
                            "the busy hour as a rule of thumb. Two "
                            "conventions from two fields that agree is "
                            f"weaker evidence than one measurement. Plausible "
                            f"range {HUMAN_PEAK_ACTIVE_PER_100_LOW:g} to "
                            f"{HUMAN_PEAK_ACTIVE_PER_100_HIGH:g}, so it can "
                            "be wrong by a factor of two either way)")
    users.add_argument("--human-sessions-per-active-user", type=float,
                       default=None,
                       help="how many sessions one of those active people "
                            "holds at that moment (default: "
                            f"{DEFAULT_HUMAN_SESSIONS_PER_ACTIVE_USER:g}, an "
                            "EXAMPLE. No published evidence exists for this "
                            "figure)")
    users.add_argument("--automated-users", type=float, default=None,
                       help="how many people run automated clients: a user "
                            "count, not a session count. Give "
                            "--automated-peak-share and "
                            "--automated-sessions-per-active-user with it, or "
                            "leave either out and take its example default")
    users.add_argument("--automated-peak-share", type=float, default=None,
                       help="how many of every 100 of them are running "
                            "automated clients at the busiest moment "
                            "(default: "
                            f"{DEFAULT_AUTOMATED_PEAK_ACTIVE_PER_100:g}, an "
                            "EXAMPLE, and identity: for a program the natural "
                            "unit is the running client, and a client that is "
                            "running is already a session. The "
                            f"{DEFAULT_HUMAN_PEAK_ACTIVE_PER_100:g}-per-100 "
                            "convention describes people, most of whom are "
                            "asleep or busy; applying it to load generators "
                            "or deployed clients would divide the load by "
                            "ten)")
    users.add_argument("--automated-sessions-per-active-user", type=float,
                       default=None,
                       help="how many automated client sessions one of them "
                            "holds at that moment, often 10 or 50 for a "
                            "person running a framework (default: "
                            f"{DEFAULT_AUTOMATED_SESSIONS_PER_ACTIVE_USER:g}, "
                            "an EXAMPLE, and identity: a client that is "
                            "running is already a session)")
    users.add_argument("--human-mix", default=None,
                       help="traffic mix of the human chat sessions, three "
                            "numbers as adds/plain/agent-mode, such as "
                            f"{MIX_TRIPLE_EXAMPLE} or 45,45,10 "
                            f"(default: {default_mix_text()})")
    users.add_argument("--automated-mix", default=None,
                       help="traffic mix of the automated clients, written "
                            "the same way "
                            f"(default: {default_mix_text()})")
    # Refused by name in main. See the note in add_mix_options.
    users.add_argument("--agents", dest="retired_agents", nargs="?",
                       const="", default=None, help=argparse.SUPPRESS)

    validate = subs.add_parser(
        "validate", help="print the published figures for all three tiers, "
                          "and the model constants behind them",
        description="Print the figures this program publishes for the pilot, "
                    "target and scale tiers, together with every named "
                    "constant the model is built from, as 'name: value' lines, "
                    f"and write them to {NUMBERS_FILE} in the current "
                    "directory so any figures quoted elsewhere can be checked "
                    "against this program mechanically. It is a fixed, named "
                    "list, not a dump of the model's intermediate working.")
    add_mix_options(validate)
    validate.add_argument("--out", default=NUMBERS_FILE,
                          help="where to write the JSON file "
                               "(default: %(default)s)")

    serve = subs.add_parser(
        "serve", help="run the web form and the JSON endpoint",
        description="Serve an HTML form at / and a JSON endpoint at /api/calc. "
                    "The page is self-contained and needs no internet access. "
                    "Binds to localhost unless told otherwise. This is a local "
                    "development server for one person at a desk, not a "
                    "service: it has no authentication and no rate limiting, "
                    "so do not put it on an address the public can reach.")
    serve.add_argument("--host", default="127.0.0.1",
                       help="address to bind (default: %(default)s)")
    serve.add_argument("--port", type=int, default=8000,
                       help="port to bind (default: %(default)s)")

    return parser


def human_side_of_the_population(args) -> tuple:
    """The users subcommand's human sessions, and how they were arrived at.

    The subcommand has always insisted on being told how many human callers
    there are. There are two ways to answer now - a session count, or a user
    count with the two figures that convert it - and this refuses when
    neither has been given.
    """
    humans, conversion = sessions_for_caller_kind(
        CLI_HUMAN_NAMES, HUMAN_CONVERSION_DEFAULTS, args.humans,
        args.human_users, args.human_peak_share,
        args.human_sessions_per_active_user)
    if humans is None:
        raise SizingError(HUMAN_SIDE_UNANSWERED_MESSAGE)
    return humans, conversion


def refuse_the_retired_agents_flag(args) -> None:
    """Refuse --agents by name, on every subcommand that could take it.

    It used to mean "how many callers are programs", one letter from --agent,
    which is the agent-mode share of the traffic mix. Left as an unrecognized
    argument it would be refused without ever naming the flag the reader
    wants, and on a parser that has --agent there is a real risk of it being
    read as the mix share instead.
    """
    if getattr(args, "retired_agents", None) is not None:
        raise SizingError(RETIRED_AGENTS_FLAG_MESSAGE)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    try:
        # --agents used to mean "how many callers are programs", one letter
        # from --agent, which is the agent-mode share of the traffic mix.
        # It is refused by name on every subcommand that took it or that
        # takes --agent, so it can never be read as the mix share.
        refuse_the_retired_agents_flag(args)

        if args.command == "tier":
            mix = TrafficMix(args.add, args.plain, args.agent)
            result = size_deployment(TIER_OPS_PER_S[args.name], mix,
                                     args.retention_days, args.dims,
                                     args.bytes_per_value, args.node_gb,
                                     run_name=args.name)
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                title = (f"{args.name.upper()} TIER - design peak "
                         f"{as_given(TIER_OPS_PER_S[args.name])} operations/s")
                print(render_report(result, title))
                print()
                print(render_tier_headline(result))
            return 0

        if args.command == "calc":
            mix = TrafficMix(args.add, args.plain, args.agent)
            result = size_deployment(args.ops, mix, args.retention_days,
                                     args.dims, args.bytes_per_value,
                                     args.node_gb, run_name="custom")
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(render_report(
                    result,
                    f"DESIGN PEAK {as_given(args.ops)} operations/s"))
                print()
                print(render_tier_headline(result))
            return 0

        if args.command == "users":
            human_mix = (parse_mix_text(args.human_mix, "--human-mix")
                         if args.human_mix is not None else TrafficMix())
            automated_mix = (
                parse_mix_text(args.automated_mix, "--automated-mix")
                if args.automated_mix is not None else TrafficMix())
            humans, human_conversion = human_side_of_the_population(args)
            automated, automated_conversion = sessions_for_caller_kind(
                CLI_AUTOMATED_NAMES, AUTOMATED_CONVERSION_DEFAULTS,
                args.automated, args.automated_users,
                args.automated_peak_share,
                args.automated_sessions_per_active_user)
            print(render_users_report(ops_for_population(
                humans, 0.0 if automated is None else automated,
                human_mix, automated_mix,
                human_conversion=human_conversion,
                automated_conversion=automated_conversion)))
            return 0

        if args.command == "validate":
            mix = TrafficMix(args.add, args.plain, args.agent)
            mix.validate()
            return run_validate(mix, args.retention_days, args.dims,
                                args.bytes_per_value, args.node_gb, args.out)

        if args.command == "serve":
            return run_server(args.host, checked_port(args.port))

    except SizingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
  subcommand, and the flag that sets it is --automated.

  One says how expensive a request is. The other says how fast a caller sends
  requests. A caller of either kind can send requests of either kind, which is
  why each population carries its own traffic mix.

Everything the model uses is listed below. A reader should be able to audit the
whole model from this docstring alone, without reading the code.

Four labels are used throughout. Every number in the report's tables of
findings carries one, and so does every constant listed below:

  measured    - it came out of a real test, and the label names that test:
                its date, the configuration it ran with, or both.
  derived     - this program computed it from measured numbers and from the
                assumptions listed here.
  estimate    - nobody has measured it. Benchmark it before ordering hardware
                against it.
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
ESTIMATE - none of the following has ever been measured
--------------------------------------------------------------------------------

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

Callers to capacity. There are two kinds of caller. A human chat session is a
  person typing in a chat window, estimated at 0.011 to 0.028 operations per
  second. At roughly two operations per prompt that is about 20 prompts an hour
  at the low end (40 operations an hour, 0.011 ops/s) and about 50 prompts an
  hour at the high end (about 101 operations an hour, 0.028 ops/s); neither
  prompt rate has been measured. The rate counts operations of all three types,
  split by that population's own traffic mix, so about one operation in ten of
  a default session's traffic is an agent-mode search - without that, dividing
  a tier's rate by a session rate made only of adds and plain searches would
  count the agent-mode share twice. An automated client is a program that sends
  requests in a loop rather than a person; one running a 5-second tool loop is
  estimated at 0.4 operations per second. How many times a human that is comes
  out of the constants themselves (0.4 / 0.028 = 14 and 0.4 / 0.011 = 36, so 14
  to 36 times), rather than being written down where it could drift. The gap
  between one request and the next has never been
  measured for either. Meter real operations per second per API key from the
  first day of the pilot and re-check the tier choice against it. A population
  of nobody demands no operations, and the report says there is nothing to size
  rather than naming the pilot tier for it.

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
  average. A design peak must be greater than zero: zero operations per second
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
  submission always sends all eight boxes, so an empty one means the reader
  cleared it, and the page says which box is blank instead of inventing a
  number. The "RAM per vector-store machine" box is the one exception, because
  empty there already means "choose the size for me" and the hint under the
  box says so.

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
# ESTIMATE. The gap between one request and the next has never been measured.
# These describe how fast a CALLER sends requests. Nothing here says anything
# about what kind of request it sends: that is the traffic mix above.
HUMAN_SESSION_OPS_PER_S_LOW = 0.011
HUMAN_SESSION_OPS_PER_S_HIGH = 0.028
# An automated client is a program that sends requests in a loop rather than a
# person typing in a chat window. It is NOT the same thing as agent-mode
# search, which is a flag on one request.
AUTOMATED_CLIENT_OPS_PER_S = 0.4
# Used only to describe where the two human figures come from: about two
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
                    run_name: str = "custom") -> dict:
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
    users = {
        "human_sessions_low": ops_per_s / HUMAN_SESSION_OPS_PER_S_HIGH,
        "human_sessions_high": ops_per_s / HUMAN_SESSION_OPS_PER_S_LOW,
        "automated_client_sessions": ops_per_s / AUTOMATED_CLIENT_OPS_PER_S,
        "human_ops_per_s_low": HUMAN_SESSION_OPS_PER_S_LOW,
        "human_ops_per_s_high": HUMAN_SESSION_OPS_PER_S_HIGH,
        "automated_client_ops_per_s": AUTOMATED_CLIENT_OPS_PER_S,
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
                       automated_mix: TrafficMix | None = None) -> dict:
    """Convert a population of callers into the capacity it demands.

    humans          concurrent human chat sessions - people typing
    automated       concurrent automated clients - programs sending requests
                    in a loop. This is a kind of CALLER. It is not agent-mode
                    search, which is a flag on one request.
    human_mix       how the human sessions' operations split between adds,
                    plain searches and agent-mode searches
    automated_mix   the same for the automated clients

    Each population gets its own mix because the kind of caller and the kind of
    request are correlated: automated clients may use agent-mode search on
    nearly every call while people rarely do. Both mixes default to the model's
    own default mix, so leaving them off changes nothing.

    The blended mix is weighted at the busy end of the human rate, which is the
    rate this report tells you to plan for, and it is what sizes the
    deployment - the global default mix is not used once a population is given.
    """
    for field, value in (("human chat sessions", humans),
                         ("automated clients", automated)):
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
    low = human_low + automated_ops
    high = human_high + automated_ops
    blended = blend_mixes(human_high, human_mix, automated_ops, automated_mix)

    # The deployment this population needs, sized at the rate the report tells
    # you to plan for and with the population's own blended mix. A population
    # that makes no requests is not a deployment to size, and size_deployment
    # refuses a design peak of zero for that reason, so there is nothing here.
    sizing = None
    if high > 0:
        sizing = size_deployment(high, blended, run_name="population")

    return {
        "humans": humans,
        "automated": automated,
        "human_mix": human_mix.as_dict(),
        "automated_mix": automated_mix.as_dict(),
        "human_ops_per_s_low": human_low,
        "human_ops_per_s_high": human_high,
        "automated_ops_per_s": automated_ops,
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

    sections.append({
        "title": "Inputs",
        "note": "Everything the answer depends on. Change any of these and the "
                "numbers below change.",
        # Every input is echoed exactly as it was given. Rounding these to whole
        # numbers described a deployment the tables below were not sized for:
        # half a byte per number, which is int4 quantization, read as "0".
        "headers": ["Item", "Value", "Label"],
        "rows": [
            ["Design peak", f"{as_given(i['ops_per_s'])} operations/s",
             "assumption (worst rate sustained 5 minutes)"],
            ["Traffic mix per 100 operations",
             TrafficMix(**i["mix"]).as_words(),
             ("assumption - never measured; agent-mode search is a "
              "request flag, not a kind of caller")],
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
        "note": "Both per-caller rates are estimates. The gap between one "
                "request and the next has never been measured. Meter real "
                "operations per second per API key from the first day of the "
                "pilot. An automated client is a kind of CALLER - a program "
                "sending requests in a loop - and has nothing to do with "
                "agent-mode search, which is a flag on one REQUEST. Each row "
                "below is that kind of caller on its own, sized at this run's "
                "traffic mix.",
        "headers": ["Kind of caller", "Sessions held", "Label"],
        "rows": [
            [("Human chat sessions "
              f"({u['human_ops_per_s_low']:g}-"
              f"{u['human_ops_per_s_high']:g} ops/s each)"),
             (f"{num(u['human_sessions_low'], 0)} to "
              f"{num(u['human_sessions_high'], 0)}"), "estimate"],
            [("Automated clients in a 5-second tool loop "
              f"({u['automated_client_ops_per_s']:g} ops/s each)"),
             num(u["automated_client_sessions"], 0), "estimate"],
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


def population_note() -> str:
    """The paragraph above the population tables, in the model's own numbers."""
    spread_low = AUTOMATED_CLIENT_OPS_PER_S / HUMAN_SESSION_OPS_PER_S_HIGH
    spread_high = AUTOMATED_CLIENT_OPS_PER_S / HUMAN_SESSION_OPS_PER_S_LOW
    prompts_low = HUMAN_SESSION_OPS_PER_S_LOW * 3600 / OPS_PER_HUMAN_PROMPT
    prompts_high = HUMAN_SESSION_OPS_PER_S_HIGH * 3600 / OPS_PER_HUMAN_PROMPT
    return (
        "There are two kinds of caller here, and both per-caller rates are "
        "estimates. A human chat session is a person typing, assumed to make "
        f"{HUMAN_SESSION_OPS_PER_S_LOW:g} to "
        f"{HUMAN_SESSION_OPS_PER_S_HIGH:g} operations per second. That is about "
        f"{prompts_low:.0f} prompts an hour at the low end and about "
        f"{prompts_high:.0f} prompts an hour at the high end, at roughly "
        f"{OPS_PER_HUMAN_PROMPT:g} operations per prompt. An automated client "
        "is a program that sends requests in a loop rather than a person; in a "
        f"5-second tool loop it is assumed to make {AUTOMATED_CLIENT_OPS_PER_S:g} "
        f"operations per second, which is {spread_low:.0f} to "
        f"{spread_high:.0f} times a human. An automated client is a kind of "
        "CALLER. It is not agent-mode search, which is a flag on one REQUEST; "
        "callers of either kind can send requests of either kind, which is why "
        "each population below carries its own traffic mix. The gap between one "
        "request and the next has never been measured for either caller, so "
        "this is the largest single uncertainty in this model.")


def tier_phrase(name, ops_per_s: float) -> str:
    """How the report names the smallest tier that holds a rate."""
    if name is None:
        biggest = TIER_OPS_PER_S[TIER_ORDER[-1]]
        return (f"above the scale tier ({num(biggest, 0)} ops/s) by "
                f"{ops_per_s / biggest:.1f}x - no named tier holds it")
    return f"{name} ({num(TIER_OPS_PER_S[name], 0)} ops/s design peak)"


def population_sections(pop: dict) -> list:
    """Every table of the population report, as (title, note, headers, rows).

    Shared by the text report and the web page, so the two can never disagree.
    """
    human_mix = TrafficMix(**pop["human_mix"])
    automated_mix = TrafficMix(**pop["automated_mix"])
    blended = TrafficMix(**pop["blended_mix"])
    sections = [{
        "title": "Demand from this population",
        "note": population_note(),
        "headers": ["Kind of caller", "Count", "Rate each", "Demand", "Label"],
        "rows": [
            ["Human chat sessions", num(pop["humans"], 0),
             (f"{HUMAN_SESSION_OPS_PER_S_LOW:g}-"
              f"{HUMAN_SESSION_OPS_PER_S_HIGH:g} ops/s each"),
             (f"{num(pop['human_ops_per_s_low'], 2)} to "
              f"{num(pop['human_ops_per_s_high'], 2)} ops/s"),
             "estimate - the per-caller rate has never been measured"],
            ["Automated clients", num(pop["automated"], 0),
             f"{AUTOMATED_CLIENT_OPS_PER_S:g} ops/s each",
             f"{num(pop['automated_ops_per_s'], 2)} ops/s",
             "estimate - the per-caller rate has never been measured"],
            ["Total", num(pop["humans"] + pop["automated"], 0), "-",
             (f"{num(pop['ops_per_s_low'], 2)} to "
              f"{num(pop['ops_per_s_high'], 2)} ops/s"),
             "derived from the two estimates above"],
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
            ["Human chat sessions", human_mix.as_words(),
             "assumption - never measured"],
            ["Automated clients", automated_mix.as_words(),
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
    out.append("        estimate = never measured. Benchmark before ordering "
               "hardware.")
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
        (f"{name}.automated_client_sessions", u["automated_client_sessions"]),
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
        ("constants.automated_client_ops_per_s", AUTOMATED_CLIENT_OPS_PER_S),
        ("constants.ops_per_human_prompt", OPS_PER_HUMAN_PROMPT),

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

SIZING_FORM_FIELDS = [
    ("ops", "Design peak, operations/s", "worst rate sustained 5 minutes"),
    ("add", "Adds per 100 ops", "planning assumption"),
    ("plain", "Plain searches per 100 ops", "planning assumption"),
    ("agent", "Agent-mode searches per 100 ops",
     "a request flag, not a caller"),
    ("retention_days", "Retention, days", "undecided - placeholder"),
    ("dims", "Vector dimensions", "fix before first ingest"),
    ("bytes_per_value", "Bytes per number", "1 means int8 quantized"),
    ("node_gb", "RAM per vector-store machine, GB",
     "blank or 'automatic' chooses it"),
]

# The caller population, which is a separate question from the sizing above:
# how many callers there are and how fast each sends, rather than a design
# peak somebody already knows. Both counts may be left blank, and then the
# page asks nothing about a population at all.
POPULATION_FORM_FIELDS = [
    ("humans", "Human chat sessions", "people typing; blank for none"),
    ("automated", "Automated clients",
     "programs in a tool loop; blank for none"),
    ("human_mix", "Human traffic mix", "adds/plain/agent-mode"),
    ("automated_mix", "Automated client traffic mix",
     "adds/plain/agent-mode"),
]

# Every box on the page, in the order it is drawn.
FORM_FIELDS = SIZING_FORM_FIELDS + POPULATION_FORM_FIELDS

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
    "humans": ("human chat sessions",
               ("how many people are typing at once, or nothing at all if "
                "you are not sizing from a population")),
    "automated": ("automated clients",
                  ("how many programs are sending requests in a loop, or "
                   "nothing at all if you are not sizing from a population")),
    "human_mix": ("human traffic mix",
                  ("how the human sessions' 100 operations split, as "
                   f"adds/plain/agent-mode, such as {MIX_TRIPLE_EXAMPLE}")),
    "automated_mix": ("automated client traffic mix",
                      ("how the automated clients' 100 operations split, "
                       "written the same way")),
}

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
        "how many callers there are than from a rate. Leave both counts empty "
        "and this part is skipped. An automated client is a caller: a program "
        "that sends requests in a loop. It is not the same thing as "
        "agent-mode search above, which is a flag on one request. Each "
        "population carries its own mix, written as adds/plain/agent-mode.")
    draw_boxes(POPULATION_FORM_FIELDS)
    parts.append('<button type="submit">Calculate</button></form>')
    if result is not None and result.get("population") is not None:
        parts.extend(html_sections(population_sections(result["population"])))
    if result is not None:
        parts.extend(html_sections(report_sections(result)))

    parts.append(
        "<footer><p><strong>Labels.</strong> measured = from a real test, named "
        "by its date, its configuration, or both. derived = computed by this "
        "program from measured "
        "numbers. estimate = never measured; benchmark before ordering "
        "hardware. assumption = a planning choice, not a finding. The Qdrant "
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
    sends all eight boxes, so an empty one is a blank answer, and blanking the
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


def population_from_form(values: dict):
    """The caller population the form asks about, or None if it asks about none.

    The two count boxes are the switch. Leave both empty and the page is only
    a sizing calculator, exactly as it was before it could take a population;
    fill in either one and the empty one counts as none of that kind of
    caller. The two mix boxes are always answered, because each has a default
    on the page, and each is read the same way as --human-mix and
    --automated-mix on the command line.
    """
    humans_text = str(values.get("humans", "")).strip()
    automated_text = str(values.get("automated", "")).strip()
    if humans_text == "" and automated_text == "":
        return None
    humans = 0.0 if humans_text == "" else read_number(humans_text, "humans")
    automated = (0.0 if automated_text == ""
                 else read_number(automated_text, "automated"))
    human_mix = read_mix_box(values, "human_mix")
    automated_mix = read_mix_box(values, "automated_mix")
    return ops_for_population(humans, automated, human_mix, automated_mix)


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
        ops = read_number(values["ops"], "ops")
        mix = TrafficMix(read_number(values["add"], "add"),
                         read_number(values["plain"], "plain"),
                         read_number(values["agent"], "agent"))
        retention = read_number(values["retention_days"], "retention_days")
        dims = read_number(values["dims"], "dims")
        bpv = read_number(values["bytes_per_value"], "bytes_per_value")
        node_gb = parse_node_gb(values["node_gb"])
        result = size_deployment(ops, mix, retention, dims, bpv, node_gb,
                                 node_gb_source=NODE_GB_SOURCE_WEB,
                                 run_name="web")
        result["population"] = population_from_form(values)
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
                    "which is a flag on one request.")
    users.add_argument("--humans", type=float, required=True,
                       help="concurrent human chat sessions - people typing")
    users.add_argument("--automated", type=float, default=0.0,
                       help="concurrent automated clients: programs sending "
                            "requests in a 5-second tool loop "
                            "(default: %(default)s)")
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
            print(render_users_report(ops_for_population(
                args.humans, args.automated, human_mix, automated_mix)))
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

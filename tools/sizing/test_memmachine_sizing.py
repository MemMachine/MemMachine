#!/usr/bin/env python3
"""Tests for memmachine_sizing.py, the MemMachine sizing calculator.

These tests are written from the agreed sizing model, NOT from what the
program happens to do. Every expected number below is worked out by hand from
the model and written as a literal, so that a test failure means the program
and the model disagree.

The model, restated so the tests can be audited without another document:

  Tiers (design peak, the worst rate held for five minutes):
      pilot 20 ops/s, target 100 ops/s, scale 1000 ops/s. A design peak must be
      greater than zero: zero operations per second is not a deployment to
      size, so it is refused rather than sized.

  Traffic mix (a planning assumption, never measured): per 100 operations,
      45 adds, 45 plain searches, 10 agent-mode searches. Adjustable.

  Fan-out per request (read from the code on 30 August 2026):
      add           1 embed,   1 vector write,   2 PostgreSQL statements, 0 LLM
      plain search  2 embeds,  1 vector search,  2 PostgreSQL statements, 0 LLM
      agent search 22 embeds, 22 vector searches, 44 PostgreSQL statements,
                   1 to 2 LLM calls (plan on 1.5)
      A plain search drops to 1 embed once every request sends
      types: ["episodic"].

  API servers: work = vector searches/s + adds/s (one add counted as one
      plain-search-equivalent). Anchor 180 plain searches/s per 16-vCPU server
      at 8 workers, measured 30 August 2026. Ceiling 0.60, so 108 usable/s per
      server, and servers = ceil(work / 108).

  Embedding GPU cards: demand = embeds/s WITHOUT the types fix. A card does
      300 to 500 requests/s (estimate). Ceiling 0.60, so 180 to 300 usable/s.
      cards = ceil(demand / usable) + 1 spare.

  Agent-model GPU cards: 15 LLM calls/s per 8B-class card (estimate),
      + 1 spare, sized on the 1.5-calls planning figure.

  Qdrant: episodes = adds/s x 86400 x retention days.
      hot RAM bytes = episodes x dims x bytes per value x 1.5.
      dims 1024 and 1 byte per value by default. GB means 10^9 bytes.
      Node RAM options 256 / 512 / 768 GB, each filled to at most 70%. The
      size is chosen automatically unless it is forced - with --node-gb, or
      with the "RAM per vector-store machine" box on the web form - and a
      forced size must be greater than zero. The report names whichever of the
      two forced it.

  Machine counts always round up, and any work at all costs at least one
      machine: a positive design peak never comes back as nothing to buy.

  Values are never quietly changed. Dimensions must be a whole number - 1024.7
      is refused, not cut down to 1024. Every input that scales the arithmetic
      has an upper bound far above any deployment, so a number too large is
      refused by name instead of overflowing to infinity and being reported as
      "inf", a value nobody typed. The bounds change no machine count.

  On the web form, an empty box is a blank answer and not a request for the
      default: the page says which box is blank. A parameter left out of the
      URL altogether is different and still takes the default, which is how
      the command line treats a flag that is left off. The "RAM per
      vector-store machine" box is the one exception, because empty there
      already means "choose the size for me".

  PostgreSQL: connections = api servers x 8 workers x 15 connections
      + 20 gateway connections per API server. Compare with the chart default
      of 100 and with the 600 that cleared every error on 30 August 2026.

  Network: from named per-call byte-size estimates, reported in Mbps.

  Callers: two kinds, and they are told apart from the two kinds of request.
      A human chat session is a person typing, 0.011 to 0.028 ops/s. An
      automated client is a program sending requests in a loop, 0.4 ops/s in a
      5-second tool loop. Both estimates. Neither is agent-mode search, which
      is a flag on one request and the third share of the traffic mix. Each
      population carries its own mix, and the `users` subcommand blends the
      two, weighted by the operations each population demands at the busy end
      of the human rate, and sizes the deployment from the blended mix.

  Users to concurrent sessions: a user count is not a session count, and two
      figures turn one into the other. concurrent sessions = users x share
      active at the busiest moment / 100 x sessions per active user. Both
      figures are the reader's and neither has a default, so a user count
      given without them is refused. A reader who gives concurrent sessions
      directly never meets any of this.
"""

from __future__ import annotations

import ast
import http.server
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from typing import ClassVar

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "memmachine_sizing.py")
# The tests run the calculator as a command with whatever interpreter is
# running them, so there is no virtual environment to set up and nothing to
# install: the calculator uses the standard library only.
PYTHON = sys.executable

sys.path.insert(0, HERE)

import memmachine_sizing as ps  # noqa: E402

# =============================================================================
# The model's own numbers, written out by hand.
# =============================================================================

USABLE_SEARCHES_PER_SERVER = 108.0          # 180 x 0.60
EMBED_USABLE_LOW = 180.0                    # 300 x 0.60, pessimistic card
EMBED_USABLE_HIGH = 300.0                   # 500 x 0.60, optimistic card
LLM_CALLS_PER_CARD = 15.0
SPARE_CARDS = 1
SECONDS_PER_DAY = 86400
INDEX_OVERHEAD = 1.5
NODE_FILL = 0.70
GB = 1_000_000_000

# Per tier, worked out by hand from the mix 45 / 45 / 10.
#   pilot  20 ops/s ->   9 adds,   9 plain,   2 agent
#   target 100      ->  45 adds,  45 plain,  10 agent
#   scale  1000     -> 450 adds, 450 plain, 100 agent
TIER_EXPECTATIONS = {
    "pilot": {
        "ops": 20.0,
        "adds": 9.0, "plain": 9.0, "agent": 2.0,
        # embeds = 9x1 + 9x2 + 2x22
        "embeds": 71.0,
        # with the types fix = 9x1 + 9x1 + 2x22
        "embeds_with_fix": 62.0,
        # vector searches = 9x1 + 2x22
        "vector_searches": 53.0,
        "vector_writes": 9.0,
        # PostgreSQL = 9x2 + 9x2 + 2x44
        "pg_statements": 124.0,
        "llm_low": 2.0, "llm_high": 4.0, "llm_planning": 3.0,
        # work = 53 + 9 = 62 -> ceil(62/108) = 1
        "api_work": 62.0, "api_servers": 1,
        # ceil(71/300)=1 (+1 spare); ceil(71/180)=1 (+1 spare)
        "embed_cards_low": 2, "embed_cards_high": 2,
        # ceil(3/15)=1 (+1 spare)
        "agent_cards": 2,
        # 1 server x 8 workers x 15 = 120, plus 1 x 20 gateway
        "pg_core_connections": 120, "pg_gateway_connections": 20,
        "pg_total_connections": 140,
        # 9 x 86400 x 90
        "episodes": 69_984_000.0,
        # 69,984,000 x 1024 x 1 x 1.5
        "hot_ram_bytes": 107_495_424_000.0,
        # 107.495 GB fits one 256 GB node (179.2 GB usable); 256 GB is the
        # least total RAM of the three options.
        "qdrant_nodes": 1, "qdrant_node_ram_gb": 256,
    },
    "target": {
        "ops": 100.0,
        "adds": 45.0, "plain": 45.0, "agent": 10.0,
        "embeds": 355.0,            # 45 + 90 + 220
        "embeds_with_fix": 310.0,   # 45 + 45 + 220
        "vector_searches": 265.0,   # 45 + 220
        "vector_writes": 45.0,
        "pg_statements": 620.0,     # 90 + 90 + 440
        "llm_low": 10.0, "llm_high": 20.0, "llm_planning": 15.0,
        # work = 265 + 45 = 310 -> ceil(310/108) = 3
        "api_work": 310.0, "api_servers": 3,
        # ceil(355/300)=2 (+1); ceil(355/180)=2 (+1)
        "embed_cards_low": 3, "embed_cards_high": 3,
        # ceil(15/15)=1 (+1)
        "agent_cards": 2,
        "pg_core_connections": 360, "pg_gateway_connections": 60,
        "pg_total_connections": 420,
        "episodes": 349_920_000.0,
        "hot_ram_bytes": 537_477_120_000.0,
        # 537.477 GB against 537.6 GB usable on one 768 GB node. The 256 GB
        # option needs 3 nodes and also buys 768 GB, so the tie goes to the
        # single machine.
        "qdrant_nodes": 1, "qdrant_node_ram_gb": 768,
    },
    "scale": {
        "ops": 1000.0,
        "adds": 450.0, "plain": 450.0, "agent": 100.0,
        "embeds": 3550.0,           # 450 + 900 + 2200
        "embeds_with_fix": 3100.0,  # 450 + 450 + 2200
        "vector_searches": 2650.0,  # 450 + 2200
        "vector_writes": 450.0,
        "pg_statements": 6200.0,    # 900 + 900 + 4400
        "llm_low": 100.0, "llm_high": 200.0, "llm_planning": 150.0,
        # work = 2650 + 450 = 3100 -> ceil(3100/108) = 29
        "api_work": 3100.0, "api_servers": 29,
        # ceil(3550/300)=12 (+1); ceil(3550/180)=20 (+1)
        "embed_cards_low": 13, "embed_cards_high": 21,
        # ceil(150/15)=10 (+1)
        "agent_cards": 11,
        "pg_core_connections": 3480, "pg_gateway_connections": 580,
        "pg_total_connections": 4060,
        "episodes": 3_499_200_000.0,
        "hot_ram_bytes": 5_374_771_200_000.0,
        # 5,374.77 GB. All three node sizes buy 7,680 GB in total
        # (30 x 256, 15 x 512, 10 x 768), so the tie goes to the fewest
        # machines.
        "qdrant_nodes": 10, "qdrant_node_ram_gb": 768,
    },
}


def free_port() -> int:
    """Ask the operating system for a port that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_cli(*args, timeout=120):
    """Run the calculator as a command and return the finished process."""
    return subprocess.run([PYTHON, MODULE_PATH, *args],
                          capture_output=True, text=True, timeout=timeout)


def report_section(r: dict, title_starts_with: str) -> dict:
    """One section of the printed report, by the start of its title."""
    for section in ps.report_sections(r):
        if section["title"].startswith(title_starts_with):
            return section
    raise AssertionError(f"the report has no {title_starts_with} section")


def sensitivity_rate_cells(r: dict) -> list:
    """The first column of the printed sensitivity table."""
    return [row[0] for row in report_section(r, "Sensitivity")["rows"]]


class ModelBaseTest(unittest.TestCase):

    def size(self, ops, **kwargs):
        return ps.size_deployment(ops, **kwargs)

    # Named the way unittest names its own assertion helpers, so it
    # reads beside assertEqual rather than against it.
    def assertClose(self, actual, expected, msg=None, places=6):  # noqa: N802
        if expected == 0:
            self.assertAlmostEqual(actual, 0.0, places=places, msg=msg)
        else:
            self.assertAlmostEqual(actual / expected, 1.0, places=places,
                                   msg=f"{msg}: expected {expected!r}, "
                                       f"got {actual!r}")


# =============================================================================
# 1. Fan-out arithmetic at all three tiers
# =============================================================================


class TestFanOutAtEveryTier(ModelBaseTest):
    """One request of each kind makes a known number of internal calls."""

    def test_fan_out_constants_match_the_code_reading(self):
        self.assertEqual(ps.ADD_EMBEDS, 1)
        self.assertEqual(ps.ADD_VECTOR_WRITES, 1)
        self.assertEqual(ps.ADD_POSTGRES_STATEMENTS, 2)
        self.assertEqual(ps.ADD_LLM_CALLS, 0)
        self.assertEqual(ps.PLAIN_EMBEDS, 2)
        self.assertEqual(ps.PLAIN_EMBEDS_WITH_TYPES_FIX, 1)
        self.assertEqual(ps.PLAIN_VECTOR_SEARCHES, 1)
        self.assertEqual(ps.PLAIN_POSTGRES_STATEMENTS, 2)
        self.assertEqual(ps.PLAIN_LLM_CALLS, 0)
        self.assertEqual(ps.AGENT_EMBEDS, 22)
        self.assertEqual(ps.AGENT_VECTOR_SEARCHES, 22)
        self.assertEqual(ps.AGENT_POSTGRES_STATEMENTS, 44)
        self.assertEqual(ps.AGENT_LLM_CALLS_LOW, 1.0)
        self.assertEqual(ps.AGENT_LLM_CALLS_HIGH, 2.0)
        self.assertEqual(ps.AGENT_LLM_CALLS_PLANNING, 1.5)

    def test_tier_design_peaks(self):
        self.assertEqual(ps.TIER_OPS_PER_S["pilot"], 20.0)
        self.assertEqual(ps.TIER_OPS_PER_S["target"], 100.0)
        self.assertEqual(ps.TIER_OPS_PER_S["scale"], 1000.0)

    def test_default_mix_is_45_45_10(self):
        mix = ps.TrafficMix()
        self.assertEqual((mix.add, mix.plain, mix.agent), (45.0, 45.0, 10.0))

    def test_request_rates_by_type(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                d = self.size(want["ops"])["demand"]
                self.assertClose(d["adds_per_s"], want["adds"], "adds/s")
                self.assertClose(d["plain_searches_per_s"], want["plain"],
                                 "plain searches/s")
                self.assertClose(d["agent_searches_per_s"], want["agent"],
                                 "agent searches/s")

    def test_embedding_demand_today_and_with_the_types_fix(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                d = self.size(want["ops"])["demand"]
                self.assertClose(d["embeds_per_s"], want["embeds"], "embeds/s")
                self.assertClose(d["embeds_per_s_with_types_fix"],
                                 want["embeds_with_fix"],
                                 "embeds/s with the types fix")
                self.assertLess(d["embeds_per_s_with_types_fix"],
                                d["embeds_per_s"],
                                "the types fix must lower the embedding demand")

    def test_vector_store_demand(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                d = self.size(want["ops"])["demand"]
                self.assertClose(d["vector_searches_per_s"],
                                 want["vector_searches"], "vector searches/s")
                self.assertClose(d["vector_writes_per_s"],
                                 want["vector_writes"], "vector writes/s")

    def test_postgres_statement_demand(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                d = self.size(want["ops"])["demand"]
                self.assertClose(d["postgres_statements_per_s"],
                                 want["pg_statements"],
                                 "PostgreSQL statements/s")

    def test_language_model_call_demand(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                d = self.size(want["ops"])["demand"]
                self.assertClose(d["agent_llm_calls_per_s_low"],
                                 want["llm_low"], "LLM calls/s low")
                self.assertClose(d["agent_llm_calls_per_s_high"],
                                 want["llm_high"], "LLM calls/s high")
                self.assertClose(d["agent_llm_calls_per_s_planning"],
                                 want["llm_planning"],
                                 "LLM calls/s planning figure")

    def test_only_agent_mode_searches_call_a_language_model(self):
        mix = ps.TrafficMix(add=50.0, plain=50.0, agent=0.0)
        d = self.size(100.0, mix=mix)["demand"]
        self.assertEqual(d["agent_llm_calls_per_s_low"], 0.0)
        self.assertEqual(d["agent_llm_calls_per_s_high"], 0.0)
        self.assertEqual(d["agent_llm_calls_per_s_planning"], 0.0)


# =============================================================================
# 2. API server counts, including the rounding boundary
# =============================================================================


class TestApiServers(ModelBaseTest):

    def test_anchor_and_ceiling(self):
        self.assertEqual(ps.API_SEARCHES_PER_S_PER_SERVER, 180.0)
        self.assertEqual(ps.API_UTILIZATION_CEILING, 0.60)
        self.assertEqual(ps.API_WORKERS_PER_SERVER, 8)
        self.assertClose(ps.api_usable_searches_per_server(),
                         USABLE_SEARCHES_PER_SERVER, "usable searches/s")

    def test_work_is_vector_searches_plus_adds(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                r = self.size(want["ops"])
                self.assertClose(r["machines"]["api_work_per_s"],
                                 want["api_work"], "API work/s")
                self.assertEqual(r["machines"]["api_servers"],
                                 want["api_servers"])

    def test_server_count_at_every_tier(self):
        self.assertEqual(self.size(20.0)["machines"]["api_servers"], 1)
        self.assertEqual(self.size(100.0)["machines"]["api_servers"], 3)
        self.assertEqual(self.size(1000.0)["machines"]["api_servers"], 29)

    def test_exactly_one_server_of_work(self):
        """108 work/s is exactly one server, not two."""
        self.assertEqual(ps.api_servers_for_work(108.0), 1)

    def test_just_above_one_server_of_work(self):
        """Anything above 108 work/s needs a second server."""
        self.assertEqual(ps.api_servers_for_work(108.0001), 2)
        self.assertEqual(ps.api_servers_for_work(108.5), 2)
        self.assertEqual(ps.api_servers_for_work(109.0), 2)

    def test_exact_multiples_do_not_round_up(self):
        self.assertEqual(ps.api_servers_for_work(216.0), 2)
        self.assertEqual(ps.api_servers_for_work(324.0), 3)
        self.assertEqual(ps.api_servers_for_work(1080.0), 10)

    def test_just_above_exact_multiples(self):
        self.assertEqual(ps.api_servers_for_work(216.5), 3)
        self.assertEqual(ps.api_servers_for_work(324.5), 4)

    def test_just_below_a_boundary(self):
        self.assertEqual(ps.api_servers_for_work(107.999), 1)
        self.assertEqual(ps.api_servers_for_work(215.999), 2)

    def test_no_work_needs_no_server(self):
        self.assertEqual(ps.api_servers_for_work(0.0), 0)

    def test_a_sliver_of_work_still_needs_one_whole_server(self):
        self.assertEqual(ps.api_servers_for_work(0.5), 1)
        self.assertEqual(ps.api_servers_for_work(1.0), 1)

    def test_server_count_is_a_whole_number(self):
        for ops in (1.0, 7.5, 20.0, 100.0, 333.0, 1000.0):
            with self.subTest(ops=ops):
                servers = self.size(ops)["machines"]["api_servers"]
                self.assertIsInstance(servers, int)


# =============================================================================
# 3. Embedding GPU cards
# =============================================================================


class TestEmbeddingGpuCards(ModelBaseTest):

    def test_card_rate_constants(self):
        self.assertEqual(ps.EMBED_CARD_REQUESTS_PER_S_LOW, 300.0)
        self.assertEqual(ps.EMBED_CARD_REQUESTS_PER_S_HIGH, 500.0)
        self.assertEqual(ps.GPU_UTILIZATION_CEILING, 0.60)
        self.assertEqual(ps.GPU_SPARE_CARDS, 1)

    def test_usable_requests_per_card(self):
        m = self.size(100.0)["machines"]
        self.assertClose(m["embed_usable_per_card_low"], EMBED_USABLE_LOW,
                         "usable per card at the pessimistic rate")
        self.assertClose(m["embed_usable_per_card_high"], EMBED_USABLE_HIGH,
                         "usable per card at the optimistic rate")

    def test_card_counts_at_every_tier(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                m = self.size(want["ops"])["machines"]
                self.assertEqual(m["embed_gpu_cards_low"],
                                 want["embed_cards_low"],
                                 "cards if a card really does 500/s")
                self.assertEqual(m["embed_gpu_cards_high"],
                                 want["embed_cards_high"],
                                 "cards if a card only does 300/s")

    def test_every_card_count_includes_one_spare(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                r = self.size(want["ops"])
                embeds = r["demand"]["embeds_per_s"]
                bare_low = math.ceil(embeds / EMBED_USABLE_HIGH)
                bare_high = math.ceil(embeds / EMBED_USABLE_LOW)
                self.assertEqual(r["machines"]["embed_gpu_cards_low"],
                                 bare_low + SPARE_CARDS)
                self.assertEqual(r["machines"]["embed_gpu_cards_high"],
                                 bare_high + SPARE_CARDS)
                self.assertEqual(r["machines"]["embed_gpu_spare"], SPARE_CARDS)

    def test_sized_on_the_demand_without_the_types_fix(self):
        """The conservative figure is the one that sets the order."""
        # A mix of plain searches only makes the two embedding figures differ
        # by a factor of two, so the choice is unmistakable.
        mix = ps.TrafficMix(add=0.0, plain=100.0, agent=0.0)
        r = self.size(100.0, mix=mix)
        self.assertClose(r["demand"]["embeds_per_s"], 200.0, "embeds/s today")
        self.assertClose(r["demand"]["embeds_per_s_with_types_fix"], 100.0,
                         "embeds/s with the types fix")
        # 200/300 -> 1 card, +1 spare = 2.  200/180 -> 2 cards, +1 spare = 3.
        self.assertEqual(r["machines"]["embed_gpu_cards_low"], 2)
        self.assertEqual(r["machines"]["embed_gpu_cards_high"], 3)

    def test_low_count_is_never_above_the_high_count(self):
        for ops in (1.0, 20.0, 100.0, 500.0, 1000.0, 5000.0):
            with self.subTest(ops=ops):
                m = self.size(ops)["machines"]
                self.assertLessEqual(m["embed_gpu_cards_low"],
                                     m["embed_gpu_cards_high"])

    def test_no_embedding_traffic_needs_no_card(self):
        """No embedding demand means no card, and so no spare card either.

        A design peak of zero operations per second is refused (see
        TestBadInputInTheLibrary), so this asks the card arithmetic directly
        instead of sizing a deployment that carries no traffic.
        """
        for rate in (ps.EMBED_CARD_REQUESTS_PER_S_LOW,
                     ps.EMBED_CARD_REQUESTS_PER_S_HIGH):
            with self.subTest(card_rate=rate):
                self.assertEqual(ps.embed_gpu_cards_for_demand(0.0, rate), 0)
        # A sliver of demand still buys a whole card plus the spare.
        self.assertEqual(
            ps.embed_gpu_cards_for_demand(0.5,
                                          ps.EMBED_CARD_REQUESTS_PER_S_LOW),
            1 + SPARE_CARDS)


# =============================================================================
# 4. Agent-model GPU cards
# =============================================================================


class TestAgentModelGpuCards(ModelBaseTest):

    def test_planning_rate_per_card(self):
        self.assertEqual(ps.AGENT_LLM_CALLS_PER_S_PER_CARD, LLM_CALLS_PER_CARD)

    def test_card_counts_at_every_tier(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                m = self.size(want["ops"])["machines"]
                self.assertEqual(m["agent_gpu_cards"], want["agent_cards"])

    def test_count_includes_one_spare(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                r = self.size(want["ops"])
                planning = r["demand"]["agent_llm_calls_per_s_planning"]
                bare = math.ceil(planning / LLM_CALLS_PER_CARD)
                self.assertEqual(r["machines"]["agent_gpu_cards"],
                                 bare + SPARE_CARDS)
                self.assertEqual(r["machines"]["agent_gpu_spare"], SPARE_CARDS)

    def test_sized_on_the_midpoint_of_one_to_two_calls(self):
        # 100 ops/s with 30 agent-mode searches per 100 ops -> 30 agent
        # searches/s -> 45 planning calls/s -> ceil(45/15) = 3, +1 spare = 4.
        mix = ps.TrafficMix(add=35.0, plain=35.0, agent=30.0)
        r = self.size(100.0, mix=mix)
        self.assertClose(r["demand"]["agent_llm_calls_per_s_planning"], 45.0,
                         "planning LLM calls/s")
        self.assertEqual(r["machines"]["agent_gpu_cards"], 4)

    def test_no_agent_traffic_needs_no_card(self):
        mix = ps.TrafficMix(add=50.0, plain=50.0, agent=0.0)
        m = self.size(100.0, mix=mix)["machines"]
        self.assertEqual(m["agent_gpu_cards"], 0)
        self.assertEqual(m["agent_gpu_spare"], 0)


# =============================================================================
# 5. Qdrant episodes, bytes and node counts at 70% fill
# =============================================================================


class TestQdrantStorage(ModelBaseTest):

    def test_storage_constants(self):
        self.assertEqual(ps.DEFAULT_VECTOR_DIMS, 1024)
        self.assertEqual(ps.DEFAULT_BYTES_PER_VALUE, 1)
        self.assertEqual(ps.QDRANT_INDEX_OVERHEAD_FACTOR, INDEX_OVERHEAD)
        self.assertEqual(ps.QDRANT_NODE_FILL_LIMIT, NODE_FILL)
        self.assertEqual(tuple(ps.QDRANT_NODE_RAM_OPTIONS_GB), (256, 512, 768))
        self.assertEqual(ps.SECONDS_PER_DAY, SECONDS_PER_DAY)
        self.assertEqual(ps.BYTES_PER_GB, GB, "GB must mean 10^9 bytes")
        self.assertEqual(ps.DEFAULT_RETENTION_DAYS, 90)

    def test_episode_counts_at_every_tier(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                s = self.size(want["ops"])["storage"]
                self.assertClose(s["episodes_retained"], want["episodes"],
                                 "episodes retained")

    def test_hot_vector_ram_bytes_at_every_tier(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                s = self.size(want["ops"])["storage"]
                self.assertClose(s["hot_vector_ram_bytes"],
                                 want["hot_ram_bytes"], "hot vector RAM bytes")
                self.assertClose(s["hot_vector_ram_gb"],
                                 want["hot_ram_bytes"] / GB,
                                 "hot vector RAM in GB")

    def test_node_counts_at_every_tier(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                m = self.size(want["ops"])["machines"]
                self.assertEqual(m["qdrant_servers"], want["qdrant_nodes"])
                self.assertEqual(m["qdrant_node_ram_gb"],
                                 want["qdrant_node_ram_gb"])

    def test_every_node_size_is_offered(self):
        m = self.size(100.0)["machines"]
        sizes = [o["node_ram_gb"] for o in m["qdrant_options"]]
        self.assertEqual(sizes, [256, 512, 768])

    def test_usable_ram_per_node_is_70_percent(self):
        m = self.size(100.0)["machines"]
        for opt in m["qdrant_options"]:
            with self.subTest(node_ram_gb=opt["node_ram_gb"]):
                self.assertClose(opt["usable_gb_per_node"],
                                 opt["node_ram_gb"] * NODE_FILL,
                                 "usable GB per node")

    def test_node_counts_for_each_size_at_the_scale_tier(self):
        # 5,374.7712 GB against 179.2 / 358.4 / 537.6 GB usable per node.
        m = self.size(1000.0)["machines"]
        by_size = {o["node_ram_gb"]: o["nodes"] for o in m["qdrant_options"]}
        self.assertEqual(by_size[256], 30)
        self.assertEqual(by_size[512], 15)
        self.assertEqual(by_size[768], 10)

    def test_no_node_is_asked_to_hold_more_than_70_percent(self):
        for ops in (20.0, 100.0, 250.0, 1000.0, 4000.0):
            with self.subTest(ops=ops):
                r = self.size(ops)
                m = r["machines"]
                hot = r["storage"]["hot_vector_ram_bytes"]
                bought = m["qdrant_servers"] * m["qdrant_node_ram_gb"] * GB
                self.assertLessEqual(hot, bought * NODE_FILL + 1.0,
                                     "the plan fills a node past 70% of its "
                                     "RAM, which leaves no headroom")

    def test_data_exactly_filling_one_node(self):
        """179.2 GB is exactly one 256 GB node's 70% allowance."""
        exact = 256 * GB * NODE_FILL
        plan = ps.qdrant_node_plan(exact)
        by_size = {o["node_ram_gb"]: o["nodes"] for o in plan["options"]}
        self.assertEqual(by_size[256], 1, "exactly one node's worth is one node")
        self.assertEqual(by_size[512], 1)
        self.assertEqual(by_size[768], 1)
        self.assertEqual(plan["nodes"], 1)
        self.assertEqual(plan["node_ram_gb"], 256)
        self.assertClose(plan["fill_of_allowance"], 1.0, "fill of allowance")

    def test_data_exactly_filling_several_nodes(self):
        exact = 3 * 256 * GB * NODE_FILL          # 537.6 GB
        plan = ps.qdrant_node_plan(exact)
        by_size = {o["node_ram_gb"]: o["nodes"] for o in plan["options"]}
        self.assertEqual(by_size[256], 3)
        self.assertEqual(by_size[512], 2)         # 537.6 / 358.4 = 1.5
        self.assertEqual(by_size[768], 1)         # 537.6 / 537.6 = 1.0

    def test_a_byte_over_a_node_adds_a_node(self):
        just_over = 256 * GB * NODE_FILL * 1.000001
        plan = ps.qdrant_node_plan(just_over)
        by_size = {o["node_ram_gb"]: o["nodes"] for o in plan["options"]}
        self.assertEqual(by_size[256], 2)

    def test_no_data_and_no_work_needs_no_node(self):
        plan = ps.qdrant_node_plan(0.0)
        self.assertEqual(plan["nodes"], 0)

    def test_no_data_but_some_work_still_needs_one_node(self):
        """Bytes are not the only reason to own a machine."""
        plan = ps.qdrant_node_plan(0.0, least_nodes=1)
        self.assertEqual(plan["nodes"], 1)
        self.assertTrue(all(o["nodes"] >= 1 for o in plan["options"]),
                        "every size in the comparison keeps the floor")
        # The cheapest way to own one machine is the smallest one offered.
        self.assertEqual(plan["node_ram_gb"], 256)

    def test_the_floor_never_shrinks_a_count_the_bytes_earned(self):
        exact = 3 * 256 * GB * NODE_FILL
        with_floor = ps.qdrant_node_plan(exact, node_gb=256, least_nodes=1)
        self.assertEqual(with_floor["nodes"], 3)

    def test_the_chosen_size_buys_the_least_total_ram(self):
        for ops in (20.0, 100.0, 400.0, 1000.0):
            with self.subTest(ops=ops):
                m = self.size(ops)["machines"]
                usable = [o for o in m["qdrant_options"] if o["nodes"] > 0]
                least = min(o["total_ram_gb"] for o in usable)
                self.assertEqual(m["qdrant_total_ram_gb"], least)

    def test_a_tie_on_total_ram_goes_to_fewer_machines(self):
        for ops in (100.0, 1000.0):
            with self.subTest(ops=ops):
                m = self.size(ops)["machines"]
                tied = [o for o in m["qdrant_options"]
                        if o["nodes"] > 0
                        and o["total_ram_gb"] == m["qdrant_total_ram_gb"]]
                fewest = min(o["nodes"] for o in tied)
                self.assertEqual(m["qdrant_servers"], fewest)

    def test_dimensions_and_bytes_per_value_scale_the_ram(self):
        base = self.size(100.0)["storage"]["hot_vector_ram_bytes"]
        wider = self.size(100.0, dims=2048)["storage"]["hot_vector_ram_bytes"]
        fatter = self.size(100.0,
                           bytes_per_value=4)["storage"]["hot_vector_ram_bytes"]
        self.assertClose(wider, base * 2, "doubling the dimensions")
        self.assertClose(fatter, base * 4, "four bytes per number")

    def test_retention_scales_the_episode_count(self):
        thirty = self.size(100.0, retention_days=30)["storage"]
        ninety = self.size(100.0, retention_days=90)["storage"]
        self.assertClose(ninety["episodes_retained"],
                         thirty["episodes_retained"] * 3, "three times as long")


# =============================================================================
# 6. Forcing the vector-store machine size with --node-gb
# =============================================================================


class TestForcedNodeSize(ModelBaseTest):
    """--node-gb overrides the automatic choice of vector-store machine.

    At the target tier, 100 operations per second with the default mix, the
    hot vector RAM is 45 x 86,400 x 90 x 1,024 x 1 x 1.5 = 537,477,120,000
    bytes. Filled to 70%:
        256 GB machine ->  179.2 GB usable -> 3 machines, 768 GB bought
        512 GB machine ->  358.4 GB usable -> 2 machines, 1,024 GB bought
        768 GB machine ->  537.6 GB usable -> 1 machine,  768 GB bought
    so the automatic choice is one 768 GB machine (the tie on total RAM goes
    to fewer machines), and forcing 512 GB must give two 512 GB machines.
    """

    TARGET_HOT_RAM_BYTES = 45.0 * SECONDS_PER_DAY * 90 * 1024 * 1 * INDEX_OVERHEAD

    def test_the_hand_worked_hot_ram_at_the_target_tier(self):
        s = self.size(100.0)["storage"]
        self.assertClose(s["hot_vector_ram_bytes"], self.TARGET_HOT_RAM_BYTES,
                         "target tier hot vector RAM")

    def test_the_default_is_the_automatic_choice(self):
        m = self.size(100.0)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 768)
        self.assertEqual(m["qdrant_servers"], 1)
        self.assertFalse(m["qdrant_node_ram_gb_forced"])

    def test_no_forced_size_leaves_the_input_empty(self):
        self.assertIsNone(self.size(100.0)["inputs"]["node_gb"])

    def test_forcing_512_gb_gives_two_512_gb_machines(self):
        m = self.size(100.0, node_gb=512)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 512)
        self.assertEqual(m["qdrant_servers"], 2)
        self.assertEqual(m["qdrant_total_ram_gb"], 1024)
        self.assertTrue(m["qdrant_node_ram_gb_forced"])

    def test_forcing_256_gb_gives_three_256_gb_machines(self):
        m = self.size(100.0, node_gb=256)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 256)
        self.assertEqual(m["qdrant_servers"], 3)
        self.assertEqual(m["qdrant_total_ram_gb"], 768)

    def test_forcing_768_gb_matches_the_automatic_choice(self):
        forced = self.size(100.0, node_gb=768)["machines"]
        auto = self.size(100.0)["machines"]
        self.assertEqual(forced["qdrant_servers"], auto["qdrant_servers"])
        self.assertEqual(forced["qdrant_node_ram_gb"],
                         auto["qdrant_node_ram_gb"])

    def test_a_forced_size_is_reported_back_in_the_inputs(self):
        self.assertEqual(self.size(100.0, node_gb=512)["inputs"]["node_gb"], 512)

    def test_a_whole_number_of_gb_stays_a_whole_number(self):
        node_gb = self.size(100.0, node_gb=512.0)["inputs"]["node_gb"]
        self.assertIsInstance(node_gb, int)
        self.assertEqual(node_gb, 512)

    def test_a_forced_size_that_is_not_one_of_the_three_is_still_offered(self):
        m = self.size(100.0, node_gb=1024)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 1024)
        self.assertEqual(m["qdrant_servers"], 1)
        self.assertEqual([o["node_ram_gb"] for o in m["qdrant_options"]],
                         [256, 512, 768, 1024])

    def test_a_forced_size_never_holds_more_than_70_percent(self):
        for node_gb in (256, 512, 768, 1024):
            with self.subTest(node_gb=node_gb):
                r = self.size(100.0, node_gb=node_gb)
                m, s = r["machines"], r["storage"]
                per_node = s["hot_vector_ram_bytes"] / m["qdrant_servers"]
                self.assertLessEqual(per_node,
                                     node_gb * GB * NODE_FILL + 1e-6)

    def test_a_forced_size_changes_nothing_but_the_vector_store(self):
        auto = self.size(100.0)
        forced = self.size(100.0, node_gb=256)
        self.assertEqual(forced["machines"]["api_servers"],
                         auto["machines"]["api_servers"])
        self.assertEqual(forced["demand"], auto["demand"])
        self.assertClose(forced["storage"]["hot_vector_ram_bytes"],
                         auto["storage"]["hot_vector_ram_bytes"], "hot RAM")

    def test_a_smaller_forced_size_never_needs_fewer_machines(self):
        counts = [self.size(100.0, node_gb=size)["machines"]["qdrant_servers"]
                  for size in (256, 512, 768)]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_zero_is_refused(self):
        with self.assertRaises(ps.SizingError):
            self.size(100.0, node_gb=0)

    def test_a_negative_size_is_refused(self):
        with self.assertRaises(ps.SizingError):
            self.size(100.0, node_gb=-256)

    def test_a_size_that_is_not_a_number_is_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(node_gb=bad):
                with self.assertRaises(ps.SizingError):
                    self.size(100.0, node_gb=bad)

    def test_the_report_states_the_forced_size(self):
        r = self.size(100.0, node_gb=512)
        text = ps.render_report(r, "FORCED")
        self.assertIn("RAM per vector-store machine", text)
        self.assertIn("512 GB, forced", text)

    def test_the_report_states_the_automatic_choice(self):
        text = ps.render_report(self.size(100.0), "AUTOMATIC")
        self.assertIn("chosen automatically", text)

    def test_an_empty_box_means_the_automatic_choice(self):
        for raw in (None, "", "   ", "auto", "automatic", "AUTOMATIC"):
            with self.subTest(raw=raw):
                self.assertIsNone(ps.parse_node_gb(raw))

    def test_a_number_in_the_box_is_read_as_gb(self):
        self.assertEqual(ps.parse_node_gb("512"), 512.0)
        self.assertEqual(ps.parse_node_gb(" 768 "), 768.0)

    def test_text_in_the_box_is_not_silently_ignored(self):
        for raw in ("big", "512gb", "five hundred"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    ps.parse_node_gb(raw)


class TestForcedNodeSizeOnTheCommandLine(unittest.TestCase):
    """--node-gb is offered by tier, calc and validate, and only by those."""

    def test_calc_forces_two_512_gb_machines_at_100_ops(self):
        proc = run_cli("calc", "--ops", "100", "--node-gb", "512", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        m = json.loads(proc.stdout)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 512)
        self.assertEqual(m["qdrant_servers"], 2)

    def test_tier_forces_two_512_gb_machines_at_the_target_tier(self):
        proc = run_cli("tier", "target", "--node-gb", "512", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        m = json.loads(proc.stdout)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 512)
        self.assertEqual(m["qdrant_servers"], 2)

    def test_tier_without_the_flag_keeps_the_automatic_choice(self):
        proc = run_cli("tier", "target", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        m = json.loads(proc.stdout)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 768)
        self.assertEqual(m["qdrant_servers"], 1)

    def test_validate_carries_the_forced_size_into_every_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "numbers.json")
            proc = run_cli("validate", "--node-gb", "512", "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as handle:
                numbers = json.load(handle)
            self.assertEqual(numbers["target.qdrant_node_ram_gb"], 512)
            self.assertEqual(numbers["target.qdrant_servers"], 2)
            for tier in ("pilot", "target", "scale"):
                with self.subTest(tier=tier):
                    self.assertEqual(numbers[f"{tier}.qdrant_node_ram_gb"], 512)

    def test_the_text_report_names_the_forced_size(self):
        proc = run_cli("calc", "--ops", "100", "--node-gb", "512")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("512 GB, forced", proc.stdout)

    def test_zero_exits_2_with_a_message(self):
        for command in (("calc", "--ops", "100", "--node-gb", "0"),
                        ("tier", "target", "--node-gb", "0"),
                        ("validate", "--node-gb", "0")):
            with self.subTest(command=command[0]):
                proc = run_cli(*command)
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("greater than zero", proc.stderr)

    def test_a_negative_size_exits_2_with_a_message(self):
        for command in (("calc", "--ops", "100", "--node-gb=-256"),
                        ("tier", "target", "--node-gb=-256"),
                        ("validate", "--node-gb=-256")):
            with self.subTest(command=command[0]):
                proc = run_cli(*command)
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("greater than zero", proc.stderr)

    def test_a_size_that_is_not_a_number_exits_non_zero_with_a_message(self):
        proc = run_cli("calc", "--ops", "100", "--node-gb", "big")
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue(proc.stderr.strip())

    def test_the_help_text_says_what_the_default_is(self):
        for command in ("tier", "calc", "validate"):
            with self.subTest(command=command):
                proc = run_cli(command, "--help")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                # argparse wraps help text, so compare on a single line.
                text = " ".join(proc.stdout.split())
                self.assertIn("--node-gb", text)
                self.assertIn("RAM per vector-store machine in GB (default: "
                              "chosen automatically; pass 256, 512 or 768 to "
                              "force a shape)", text)

    def test_the_subcommands_without_a_vector_store_do_not_offer_it(self):
        for command in ("users", "serve"):
            with self.subTest(command=command):
                proc = run_cli(command, "--help")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertNotIn("--node-gb", proc.stdout)


# =============================================================================
# 7. Storage growth figures
# =============================================================================


class TestStorageGrowth(ModelBaseTest):

    def test_episodes_per_day_and_per_year(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                s = self.size(want["ops"])["storage"]
                self.assertClose(s["episodes_per_day"],
                                 want["adds"] * SECONDS_PER_DAY,
                                 "episodes per day")
                self.assertClose(s["episodes_per_year"],
                                 want["adds"] * SECONDS_PER_DAY * 365,
                                 "episodes per year")

    def test_target_tier_episodes_per_day_and_year(self):
        s = self.size(100.0)["storage"]
        self.assertClose(s["episodes_per_day"], 3_888_000.0, "episodes per day")
        self.assertClose(s["episodes_per_year"], 1_419_120_000.0,
                         "episodes per year")

    def test_qdrant_disk_at_the_target_tier(self):
        # 349,920,000 episodes x (1024 x 4 + 256) bytes x 1.3
        s = self.size(100.0)["storage"]
        self.assertClose(s["qdrant_nvme_bytes"], 1_979_707_392_000.0,
                         "Qdrant NVMe bytes")
        self.assertClose(s["qdrant_nvme_gb"], 1_979.707392, "Qdrant NVMe GB")

    def test_postgres_disk_range_at_the_target_tier(self):
        # low  : 349,920,000 x (800 + 400 + 300) x 1.4
        # high : 349,920,000 x (2400 + 400 + 300) x 1.4
        s = self.size(100.0)["storage"]
        self.assertClose(s["postgres_bytes_low"], 734_832_000_000.0,
                         "PostgreSQL bytes, low case")
        self.assertClose(s["postgres_bytes_high"], 1_518_652_800_000.0,
                         "PostgreSQL bytes, high case")
        self.assertLess(s["postgres_bytes_low"], s["postgres_bytes_high"])

    def test_one_year_with_nothing_deleted(self):
        """The 90-day figure scaled up by 365/90."""
        s = self.size(100.0)["storage"]
        factor = 365.0 / 90.0
        self.assertClose(s["unbounded_year_hot_vector_ram_bytes"],
                         s["hot_vector_ram_bytes"] * factor,
                         "one year of hot vector RAM")
        self.assertClose(s["unbounded_year_qdrant_nvme_bytes"],
                         s["qdrant_nvme_bytes"] * factor,
                         "one year of Qdrant NVMe")
        self.assertClose(s["unbounded_year_postgres_bytes_low"],
                         s["postgres_bytes_low"] * factor,
                         "one year of PostgreSQL, low case")
        self.assertClose(s["unbounded_year_postgres_bytes_high"],
                         s["postgres_bytes_high"] * factor,
                         "one year of PostgreSQL, high case")

    def test_one_year_hot_ram_matches_a_year_of_episodes(self):
        s = self.size(100.0)["storage"]
        # 45 adds/s x 86400 x 365 x 1024 x 1 x 1.5
        self.assertClose(s["unbounded_year_hot_vector_ram_bytes"],
                         2_179_768_320_000.0, "one year of hot vector RAM")

    def test_one_year_figures_ignore_the_retention_setting(self):
        """A year of adds is a year of adds, whatever retention is set to.

        These figures used to be worked out by scaling the retained figures by
        365 / retention_days, so at retention 0 the report printed
        1,419,120,000 episodes a year and 0.00 GB to hold them.
        """
        keys = ("unbounded_year_hot_vector_ram_bytes",
                "unbounded_year_qdrant_nvme_bytes",
                "unbounded_year_postgres_bytes_low",
                "unbounded_year_postgres_bytes_high")
        reference = self.size(100.0, retention_days=90)["storage"]
        for days in (0, 1, 7, 90, 365, 3650):
            s = self.size(100.0, retention_days=days)["storage"]
            for key in keys:
                with self.subTest(retention_days=days, figure=key):
                    self.assertClose(s[key], reference[key], key)

    def test_one_year_hot_ram_at_retention_zero(self):
        """The number the plan quotes to argue that retention is required."""
        s = self.size(100.0, retention_days=0)["storage"]
        self.assertClose(s["unbounded_year_hot_vector_ram_bytes"],
                         2_179_768_320_000.0, "one year of hot vector RAM")
        self.assertEqual(s["hot_vector_ram_bytes"], 0.0)

    def test_gb_means_ten_to_the_ninth_bytes(self):
        s = self.size(100.0)["storage"]
        self.assertClose(s["hot_vector_ram_gb"],
                         s["hot_vector_ram_bytes"] / 1_000_000_000,
                         "GB conversion")


# =============================================================================
# 7. PostgreSQL connections
# =============================================================================


class TestPostgresConnections(ModelBaseTest):

    def test_connection_constants(self):
        self.assertEqual(ps.POSTGRES_POOL_SIZE, 5)
        self.assertEqual(ps.POSTGRES_MAX_OVERFLOW, 10)
        self.assertEqual(ps.POSTGRES_CONNECTIONS_PER_WORKER, 15)
        self.assertEqual(ps.GATEWAY_CONNECTIONS_PER_API_SERVER, 20)
        self.assertEqual(ps.POSTGRES_CHART_DEFAULT_MAX_CONNECTIONS, 100)
        self.assertEqual(ps.POSTGRES_PROVEN_MAX_CONNECTIONS, 600)

    def test_connection_totals_at_every_tier(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                p = self.size(want["ops"])["postgres"]
                self.assertEqual(p["core_connections"],
                                 want["pg_core_connections"])
                self.assertEqual(p["gateway_connections"],
                                 want["pg_gateway_connections"])
                self.assertEqual(p["total_connections"],
                                 want["pg_total_connections"])
                self.assertEqual(p["max_connections_required"],
                                 want["pg_total_connections"])

    def test_the_formula_holds_for_any_server_count(self):
        for ops in (20.0, 100.0, 400.0, 1000.0):
            with self.subTest(ops=ops):
                r = self.size(ops)
                servers = r["machines"]["api_servers"]
                p = r["postgres"]
                self.assertEqual(p["core_connections"], servers * 8 * 15)
                self.assertEqual(p["gateway_connections"], servers * 20)
                self.assertEqual(p["total_connections"], servers * 140)

    def test_every_tier_exceeds_the_chart_default_of_100(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                p = self.size(want["ops"])["postgres"]
                self.assertGreater(p["total_connections"], 100)
                self.assertTrue(p["exceeds_chart_default"])

    def test_pilot_and_target_fit_inside_the_proven_600(self):
        for name in ("pilot", "target"):
            with self.subTest(tier=name):
                p = self.size(TIER_EXPECTATIONS[name]["ops"])["postgres"]
                self.assertLessEqual(p["total_connections"], 600)
                self.assertFalse(p["exceeds_proven_setting"])
                self.assertFalse(p["needs_connection_pooler"])

    def test_scale_tier_needs_more_than_has_ever_been_proven(self):
        p = self.size(1000.0)["postgres"]
        self.assertEqual(p["total_connections"], 4060)
        self.assertGreater(p["total_connections"], 600)
        self.assertTrue(p["exceeds_proven_setting"])
        self.assertTrue(p["needs_connection_pooler"])

    def test_statement_rate_is_carried_through(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                r = self.size(want["ops"])
                self.assertClose(r["postgres"]["statements_per_s"],
                                 want["pg_statements"], "statements/s")
                self.assertEqual(r["postgres"]["statements_per_s"],
                                 r["demand"]["postgres_statements_per_s"])

    def test_one_postgres_server_per_tier(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                r = self.size(want["ops"])
                self.assertEqual(r["postgres"]["servers"], 1)
                self.assertEqual(r["machines"]["postgres_servers"], 1)


# =============================================================================
# 8. Network
# =============================================================================


class TestNetwork(ModelBaseTest):

    def test_per_call_byte_sizes_are_built_from_named_constants(self):
        n = self.size(100.0)["network"]
        self.assertEqual(n["embed_bytes_per_call"],
                         ps.EMBED_REQUEST_BYTES
                         + 1024 * ps.ORIGINAL_VECTOR_BYTES_PER_VALUE
                         + ps.EMBED_RESPONSE_ENVELOPE_BYTES)
        self.assertEqual(n["embed_bytes_per_call"], 5296)
        self.assertEqual(n["vector_search_bytes_per_call"], 14396)
        self.assertEqual(n["vector_write_bytes_per_call"], 4796)
        self.assertEqual(n["llm_bytes_per_call"], 10000)

    def test_north_south_peak_at_the_target_tier(self):
        # 45 x (1200 + 300) + 45 x (600 + 10 x 900)
        #   + 10 x (600 + 2000 + 20 x 900) = 705,500 bytes/s, x 1.2 framing.
        n = self.size(100.0)["network"]
        self.assertClose(n["north_south_bytes_per_s"], 846_600.0,
                         "north-south bytes/s")
        self.assertClose(n["north_south_mbps"], 6.7728, "north-south Mbps")

    def test_east_west_peak_at_the_target_tier(self):
        # 355 x 5296 + 265 x 14396 + 45 x 4796 + 620 x 1800 + 15 x 10000
        #   = 7,176,840 bytes/s, x 1.2 framing.
        n = self.size(100.0)["network"]
        self.assertClose(n["east_west_bytes_per_s"], 8_612_208.0,
                         "east-west bytes/s")
        self.assertClose(n["east_west_mbps"], 68.897664, "east-west Mbps")

    def test_mbps_means_ten_to_the_sixth_bits_per_second(self):
        n = self.size(100.0)["network"]
        self.assertClose(n["north_south_mbps"],
                         n["north_south_bytes_per_s"] * 8 / 1_000_000,
                         "Mbps conversion")
        self.assertClose(n["east_west_mbps"],
                         n["east_west_bytes_per_s"] * 8 / 1_000_000,
                         "Mbps conversion")

    def test_traffic_grows_with_the_operation_rate(self):
        one = self.size(100.0)["network"]
        ten = self.size(1000.0)["network"]
        self.assertClose(ten["east_west_mbps"], one["east_west_mbps"] * 10,
                         "ten times the operations")
        self.assertClose(ten["north_south_mbps"], one["north_south_mbps"] * 10,
                         "ten times the operations")

    def test_east_west_is_the_busier_direction(self):
        n = self.size(100.0)["network"]
        self.assertGreater(n["east_west_mbps"], n["north_south_mbps"])
        self.assertClose(n["busiest_link_mbps"], n["east_west_mbps"],
                         "busiest link")

    def test_the_old_70_mbps_claim_is_reproducible_now(self):
        """The figure must come out of the program's own declared inputs."""
        n = self.size(100.0)["network"]
        by_hand = ((355 * 5296 + 265 * 14396 + 45 * 4796
                    + 620 * 1800 + 15 * 10000) * 1.2 * 8 / 1_000_000)
        self.assertClose(n["east_west_mbps"], by_hand, "east-west Mbps by hand")


# =============================================================================
# 9. Users, in both directions
# =============================================================================


class TestUserConversion(ModelBaseTest):

    def test_per_session_rate_constants(self):
        self.assertEqual(ps.HUMAN_SESSION_OPS_PER_S_LOW, 0.011)
        self.assertEqual(ps.HUMAN_SESSION_OPS_PER_S_HIGH, 0.028)
        self.assertEqual(ps.AUTOMATED_CLIENT_OPS_PER_S, 0.4)

    def test_capacity_to_sessions_at_the_target_tier(self):
        u = self.size(100.0)["users"]
        # Busier sessions mean fewer of them fit.
        self.assertClose(u["human_sessions_low"], 100.0 / 0.028,
                         "human sessions at the busy end")
        self.assertClose(u["human_sessions_high"], 100.0 / 0.011,
                         "human sessions at the quiet end")
        self.assertClose(u["automated_client_sessions"], 250.0,
                         "automated clients")

    def test_capacity_to_sessions_at_every_tier(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                u = self.size(want["ops"])["users"]
                self.assertClose(u["human_sessions_low"], want["ops"] / 0.028,
                                 "human sessions, busy end")
                self.assertClose(u["human_sessions_high"], want["ops"] / 0.011,
                                 "human sessions, quiet end")
                self.assertClose(u["automated_client_sessions"],
                                 want["ops"] / 0.4, "automated clients")
                self.assertLess(u["human_sessions_low"], u["human_sessions_high"])

    def test_no_capacity_is_not_a_deployment_to_convert(self):
        """A design peak of zero holds no sessions because it is refused.

        Zero operations per second is not a capacity to divide into sessions,
        so the program rejects it rather than reporting zero sessions. An empty
        population converts the other way round quite happily: no users demand
        no operations.
        """
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(0.0)
        pop = ps.ops_for_population(humans=0, automated=0)
        self.assertEqual(pop["ops_per_s_low"], 0.0)
        self.assertEqual(pop["ops_per_s_high"], 0.0)

    def test_sessions_to_capacity(self):
        pop = ps.ops_for_population(humans=5000, automated=40)
        # 5000 x 0.011 + 40 x 0.4 = 55 + 16 = 71
        # 5000 x 0.028 + 40 x 0.4 = 140 + 16 = 156
        self.assertClose(pop["ops_per_s_low"], 71.0, "ops/s, quiet humans")
        self.assertClose(pop["ops_per_s_high"], 156.0, "ops/s, busy humans")

    def test_the_two_directions_agree(self):
        """Converting one way and back must return the starting number."""
        for humans in (100, 5000, 250000):
            with self.subTest(humans=humans):
                pop = ps.ops_for_population(humans=humans, automated=0)
                back = self.size(pop["ops_per_s_high"])["users"]
                self.assertClose(back["human_sessions_low"], float(humans),
                                 "round trip through the busy rate")

    def test_a_population_of_automated_clients_only(self):
        pop = ps.ops_for_population(humans=0, automated=250)
        self.assertClose(pop["ops_per_s_low"], 100.0, "ops/s")
        self.assertClose(pop["ops_per_s_high"], 100.0, "ops/s")

    def test_smallest_tier_that_holds_a_population(self):
        pop = ps.ops_for_population(humans=5000, automated=40)
        self.assertEqual(pop["tier_for_low"], "target")   # 71 ops/s
        self.assertEqual(pop["tier_for_high"], "scale")   # 156 ops/s

    def test_tier_boundaries(self):
        self.assertEqual(ps.smallest_tier_holding(0.0), "pilot")
        self.assertEqual(ps.smallest_tier_holding(20.0), "pilot")
        self.assertEqual(ps.smallest_tier_holding(20.001), "target")
        self.assertEqual(ps.smallest_tier_holding(100.0), "target")
        self.assertEqual(ps.smallest_tier_holding(100.001), "scale")
        self.assertEqual(ps.smallest_tier_holding(1000.0), "scale")

    def test_above_the_scale_tier_no_tier_holds_it(self):
        self.assertIsNone(ps.smallest_tier_holding(1000.001))
        pop = ps.ops_for_population(humans=0, automated=100000)
        self.assertIsNone(pop["tier_for_high"])

    def test_an_automated_client_is_worth_many_humans(self):
        self.assertGreater(ps.AUTOMATED_CLIENT_OPS_PER_S
                           / ps.HUMAN_SESSION_OPS_PER_S_HIGH, 10.0)


# =============================================================================
# 10. A traffic mix that is not 45 / 45 / 10
# =============================================================================


class TestCustomTrafficMix(ModelBaseTest):

    def test_an_agent_heavy_mix(self):
        """20 adds, 20 plain searches, 60 agent-mode searches per 100 ops."""
        mix = ps.TrafficMix(add=20.0, plain=20.0, agent=60.0)
        r = self.size(100.0, mix=mix)
        d, m = r["demand"], r["machines"]
        self.assertClose(d["adds_per_s"], 20.0, "adds/s")
        self.assertClose(d["plain_searches_per_s"], 20.0, "plain searches/s")
        self.assertClose(d["agent_searches_per_s"], 60.0, "agent searches/s")
        # embeds = 20 + 40 + 1320
        self.assertClose(d["embeds_per_s"], 1380.0, "embeds/s")
        # vector searches = 20 + 1320
        self.assertClose(d["vector_searches_per_s"], 1340.0,
                         "vector searches/s")
        # PostgreSQL = 40 + 40 + 2640
        self.assertClose(d["postgres_statements_per_s"], 2720.0,
                         "PostgreSQL statements/s")
        # work = 1340 + 20 = 1360 -> ceil(1360/108) = 13
        self.assertClose(m["api_work_per_s"], 1360.0, "API work/s")
        self.assertEqual(m["api_servers"], 13)
        # planning LLM calls = 60 x 1.5 = 90 -> ceil(90/15) = 6, +1 spare
        self.assertEqual(m["agent_gpu_cards"], 7)

    def test_a_search_only_mix(self):
        """No adds at all, so no storage grows.

        The searches still have to run on a machine, so the order is one
        vector-store machine even though the stored bytes come to nothing.
        """
        mix = ps.TrafficMix(add=0.0, plain=90.0, agent=10.0)
        r = self.size(100.0, mix=mix)
        self.assertEqual(r["demand"]["adds_per_s"], 0.0)
        self.assertEqual(r["demand"]["vector_writes_per_s"], 0.0)
        self.assertEqual(r["storage"]["episodes_retained"], 0.0)
        self.assertEqual(r["storage"]["hot_vector_ram_bytes"], 0.0)
        self.assertEqual(r["machines"]["qdrant_servers"], 1)
        # vector searches = 90 + 220 = 310 = the work, ceil(310/108) = 3
        self.assertClose(r["machines"]["api_work_per_s"], 310.0, "API work/s")
        self.assertEqual(r["machines"]["api_servers"], 3)

    def test_an_add_only_mix(self):
        mix = ps.TrafficMix(add=100.0, plain=0.0, agent=0.0)
        r = self.size(100.0, mix=mix)
        d = r["demand"]
        self.assertClose(d["embeds_per_s"], 100.0, "embeds/s")
        self.assertEqual(d["vector_searches_per_s"], 0.0)
        self.assertClose(d["vector_writes_per_s"], 100.0, "vector writes/s")
        self.assertClose(d["postgres_statements_per_s"], 200.0, "statements/s")
        # work = 0 searches + 100 adds -> ceil(100/108) = 1
        self.assertClose(r["machines"]["api_work_per_s"], 100.0, "API work/s")
        self.assertEqual(r["machines"]["api_servers"], 1)
        self.assertEqual(r["machines"]["agent_gpu_cards"], 0)

    def test_the_agent_share_dominates_the_hardware_order(self):
        """One agent-mode search costs about 22 plain searches."""
        even = self.size(100.0, mix=ps.TrafficMix(45.0, 55.0, 0.0))
        with_agents = self.size(100.0, mix=ps.TrafficMix(45.0, 45.0, 10.0))
        self.assertGreater(with_agents["machines"]["api_servers"],
                           even["machines"]["api_servers"])

    def test_fractional_mix_shares(self):
        mix = ps.TrafficMix(add=33.5, plain=64.0, agent=2.5)
        r = self.size(200.0, mix=mix)
        d = r["demand"]
        self.assertClose(d["adds_per_s"], 67.0, "adds/s")
        self.assertClose(d["plain_searches_per_s"], 128.0, "plain searches/s")
        self.assertClose(d["agent_searches_per_s"], 5.0, "agent searches/s")

    def test_the_mix_is_reported_back_in_the_inputs(self):
        mix = ps.TrafficMix(add=20.0, plain=20.0, agent=60.0)
        r = self.size(100.0, mix=mix)
        self.assertEqual(r["inputs"]["mix"],
                         {"add": 20.0, "plain": 20.0, "agent": 60.0})


# =============================================================================
# 11. Bad input: a clean message and a non-zero exit, never a traceback
# =============================================================================


class TestBadInputInTheLibrary(ModelBaseTest):

    def test_mix_that_sums_to_less_than_100(self):
        mix = ps.TrafficMix(add=45.0, plain=45.0, agent=5.0)
        with self.assertRaises(ps.SizingError) as caught:
            ps.size_deployment(100.0, mix)
        self.assertIn("100", str(caught.exception))

    def test_mix_that_sums_to_more_than_100(self):
        mix = ps.TrafficMix(add=50.0, plain=50.0, agent=50.0)
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(100.0, mix)

    def test_negative_mix_share(self):
        mix = ps.TrafficMix(add=110.0, plain=-5.0, agent=-5.0)
        with self.assertRaises(ps.SizingError) as caught:
            ps.size_deployment(100.0, mix)
        self.assertIn("negative", str(caught.exception))

    def test_negative_operations_per_second(self):
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(-1.0)

    def test_zero_operations_per_second_is_rejected(self):
        """A design peak of zero operations is not a deployment to size."""
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(0.0)

    def test_negative_retention(self):
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(100.0, retention_days=-1)

    def test_zero_or_negative_dimensions(self):
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(100.0, dims=0)
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(100.0, dims=-1024)

    def test_zero_or_negative_bytes_per_value(self):
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(100.0, bytes_per_value=0)
        with self.assertRaises(ps.SizingError):
            ps.size_deployment(100.0, bytes_per_value=-1)

    def test_negative_user_counts(self):
        with self.assertRaises(ps.SizingError):
            ps.ops_for_population(humans=-1, automated=0)
        with self.assertRaises(ps.SizingError):
            ps.ops_for_population(humans=0, automated=-1)

    def test_not_a_number_and_infinity_are_refused(self):
        """NaN passes every < and > test, so it needs its own check.

        Without one it slips past validation and crashes deep in the rounding
        arithmetic with a Python traceback.
        """
        for kwargs in ({"retention_days": float("nan")},
                       {"retention_days": float("inf")},
                       {"dims": float("nan")},
                       {"bytes_per_value": float("nan")},
                       {"bytes_per_value": float("inf")}):
            with self.subTest(**kwargs):
                with self.assertRaises(ps.SizingError):
                    ps.size_deployment(100.0, **kwargs)
        for ops in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(ops=ops):
                with self.assertRaises(ps.SizingError):
                    ps.size_deployment(ops)

    def test_a_traffic_mix_share_that_is_not_a_number(self):
        """NaN + 45 + 10 is NaN, and NaN never differs from 100 by more than
        the tolerance, so the sum check alone lets it through."""
        for mix in (ps.TrafficMix(float("nan"), 45.0, 10.0),
                    ps.TrafficMix(45.0, float("nan"), 10.0),
                    ps.TrafficMix(45.0, 45.0, float("inf"))):
            with self.subTest(mix=mix):
                with self.assertRaises(ps.SizingError):
                    ps.size_deployment(100.0, mix)

    def test_caller_counts_that_are_not_numbers(self):
        for humans, automated in ((float("nan"), 0), (0, float("nan")),
                                  (float("inf"), 0)):
            with self.subTest(humans=humans, automated=automated):
                with self.assertRaises(ps.SizingError):
                    ps.ops_for_population(humans=humans,
                                          automated=automated)

    def test_ceil_up_refuses_a_quantity_that_is_not_finite(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ps.SizingError):
                    ps.ceil_up(value, 108.0)

    def test_sizing_error_is_a_value_error(self):
        self.assertTrue(issubclass(ps.SizingError, ValueError))


class TestBadInputOnTheCommandLine(unittest.TestCase):
    """Every bad input must exit non-zero with a message, not a traceback."""

    # Named the way unittest names its own assertion helpers, so it
    # reads beside assertEqual rather than against it.
    def assertCleanFailure(self, proc):  # noqa: N802
        self.assertNotEqual(proc.returncode, 0,
                            "bad input must exit non-zero")
        self.assertNotIn("Traceback", proc.stderr,
                         "bad input printed a Python traceback")
        self.assertNotIn("Traceback", proc.stdout,
                         "bad input printed a Python traceback")
        self.assertTrue(proc.stderr.strip() or proc.stdout.strip(),
                        "bad input printed no message at all")

    def test_mix_below_100(self):
        proc = run_cli("calc", "--ops", "100", "--add", "45", "--plain", "45",
                       "--agent", "5")
        self.assertCleanFailure(proc)
        self.assertIn("100", proc.stderr + proc.stdout)

    def test_mix_above_100(self):
        proc = run_cli("calc", "--ops", "100", "--add", "50", "--plain", "50",
                       "--agent", "50")
        self.assertCleanFailure(proc)

    def test_negative_mix_share(self):
        proc = run_cli("calc", "--ops", "100", "--add", "110", "--plain", "-5",
                       "--agent", "-5")
        self.assertCleanFailure(proc)

    def test_negative_ops(self):
        proc = run_cli("calc", "--ops", "-100")
        self.assertCleanFailure(proc)

    def test_zero_ops(self):
        proc = run_cli("calc", "--ops", "0")
        self.assertCleanFailure(proc)

    def test_negative_retention(self):
        proc = run_cli("calc", "--ops", "100", "--retention-days", "-30")
        self.assertCleanFailure(proc)

    def test_zero_dimensions(self):
        proc = run_cli("calc", "--ops", "100", "--dims", "0")
        self.assertCleanFailure(proc)

    def test_negative_user_count(self):
        proc = run_cli("users", "--humans", "-10")
        self.assertCleanFailure(proc)

    def test_bad_tier_name(self):
        proc = run_cli("tier", "enormous")
        self.assertCleanFailure(proc)

    def test_non_numeric_ops(self):
        proc = run_cli("calc", "--ops", "lots")
        self.assertCleanFailure(proc)

    def test_operations_per_second_that_is_not_a_number(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(ops=value):
                proc = run_cli("calc", "--ops", value)
                self.assertCleanFailure(proc)
                self.assertEqual(proc.returncode, 2)

    def test_an_operations_rate_so_large_it_overflows(self):
        """1e300 is finite, but far past the largest peak this will size.

        It used to pass the finite check and overflow to infinity part-way
        through the byte arithmetic, and the reader was told about "inf" - a
        value they never typed. It is refused by name now.
        """
        proc = run_cli("calc", "--ops", "1e300")
        self.assertCleanFailure(proc)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the design peak is 1e+300", proc.stderr + proc.stdout)
        self.assertNotIn("inf", proc.stderr + proc.stdout)

    def test_retention_that_is_not_a_number(self):
        for value in ("inf", "nan"):
            with self.subTest(retention=value):
                proc = run_cli("calc", "--ops", "100", "--retention-days", value)
                self.assertCleanFailure(proc)
                self.assertEqual(proc.returncode, 2)

    def test_bytes_per_value_that_is_not_a_number(self):
        proc = run_cli("calc", "--ops", "100", "--bytes-per-value", "nan")
        self.assertCleanFailure(proc)
        self.assertEqual(proc.returncode, 2)

    def test_a_traffic_mix_share_that_is_not_a_number(self):
        proc = run_cli("calc", "--ops", "100", "--add", "nan", "--plain", "45",
                       "--agent", "10")
        self.assertCleanFailure(proc)
        self.assertEqual(proc.returncode, 2)

    def test_a_user_count_that_is_not_a_number(self):
        """It used to exit 0 and print a report full of the word nan."""
        proc = run_cli("users", "--humans", "nan")
        self.assertCleanFailure(proc)
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("nan ops/s", proc.stdout)

    def test_bad_mix_on_the_validate_subcommand(self):
        proc = run_cli("validate", "--add", "45", "--plain", "45",
                       "--agent", "5")
        self.assertCleanFailure(proc)

    def test_bad_port_on_the_serve_subcommand(self):
        proc = run_cli("serve", "--port", "0")
        self.assertCleanFailure(proc)


# =============================================================================
# 12. The shape of the JSON result
# =============================================================================


class TestResultShape(ModelBaseTest):

    TOP_LEVEL = ("run_name", "inputs", "demand", "machines", "storage",
                 "postgres", "network", "users", "sensitivity")

    def test_top_level_keys(self):
        r = self.size(100.0)
        for key in self.TOP_LEVEL:
            with self.subTest(key=key):
                self.assertIn(key, r)

    def test_inputs_block(self):
        r = self.size(100.0, retention_days=30, dims=768, bytes_per_value=2)
        self.assertEqual(r["inputs"]["ops_per_s"], 100.0)
        self.assertEqual(r["inputs"]["retention_days"], 30)
        self.assertEqual(r["inputs"]["dims"], 768)
        self.assertEqual(r["inputs"]["bytes_per_value"], 2)
        self.assertEqual(set(r["inputs"]["mix"]), {"add", "plain", "agent"})

    def test_demand_keys(self):
        d = self.size(100.0)["demand"]
        for key in ("adds_per_s", "plain_searches_per_s", "agent_searches_per_s",
                    "embeds_per_s", "embeds_per_s_with_types_fix",
                    "vector_searches_per_s", "vector_writes_per_s",
                    "postgres_statements_per_s", "agent_llm_calls_per_s_low",
                    "agent_llm_calls_per_s_high",
                    "agent_llm_calls_per_s_planning"):
            with self.subTest(key=key):
                self.assertIn(key, d)

    def test_machines_keys(self):
        m = self.size(100.0)["machines"]
        for key in ("api_servers", "api_work_per_s",
                    "api_usable_searches_per_server", "postgres_servers",
                    "qdrant_servers", "qdrant_node_ram_gb",
                    "qdrant_usable_gb_per_node", "qdrant_total_ram_gb",
                    "qdrant_options", "embed_gpu_cards_low",
                    "embed_gpu_cards_high", "agent_gpu_cards",
                    "total_cpu_servers"):
            with self.subTest(key=key):
                self.assertIn(key, m)

    def test_machine_counts_are_whole_numbers(self):
        m = self.size(100.0)["machines"]
        for key in ("api_servers", "postgres_servers", "qdrant_servers",
                    "embed_gpu_cards_low", "embed_gpu_cards_high",
                    "agent_gpu_cards", "total_cpu_servers"):
            with self.subTest(key=key):
                self.assertIsInstance(m[key], int)

    def test_total_ordinary_servers_adds_up(self):
        m = self.size(100.0)["machines"]
        self.assertEqual(m["total_cpu_servers"],
                         m["api_servers"] + m["postgres_servers"]
                         + m["qdrant_servers"])

    def test_sensitivity_table_rows(self):
        r = self.size(100.0)
        rates = [row["agent_searches_per_s"] for row in r["sensitivity"]]
        self.assertEqual(rates, [0.0, 2.0, 10.0, 25.0])
        for row in r["sensitivity"]:
            with self.subTest(rate=row["agent_searches_per_s"]):
                for key in ("total_ops_per_s", "vector_searches_per_s",
                            "api_work_per_s", "api_servers",
                            "llm_calls_per_s_low", "llm_calls_per_s_high"):
                    self.assertIn(key, row)

    def test_sensitivity_holds_adds_and_plain_searches_fixed(self):
        r = self.size(100.0)
        for row in r["sensitivity"]:
            rate = row["agent_searches_per_s"]
            with self.subTest(rate=rate):
                # adds 45 and plain searches 45 stay put; only agents vary.
                self.assertClose(row["total_ops_per_s"], 90.0 + rate,
                                 "total ops/s")
                self.assertClose(row["vector_searches_per_s"], 45.0 + 22 * rate,
                                 "vector searches/s")
                self.assertClose(row["api_work_per_s"], 90.0 + 22 * rate,
                                 "API work/s")
                self.assertEqual(row["api_servers"],
                                 math.ceil((90.0 + 22 * rate) / 108.0))

    def test_sensitivity_matches_the_headline_at_the_real_agent_rate(self):
        """At every tier, not only where the tier rate happens to be in the
        fixed list. The scale tier's own rate is 100 agent-mode searches/s,
        which the fixed rates 0/2/10/25 do not cover."""
        for ops in (20.0, 100.0, 1000.0):
            with self.subTest(ops=ops):
                r = self.size(ops)
                want = r["demand"]["agent_searches_per_s"]
                rows = [x for x in r["sensitivity"]
                        if abs(x["agent_searches_per_s"] - want) <= 1e-9]
                self.assertEqual(len(rows), 1,
                                 "the tier's own agent-mode rate must appear "
                                 "exactly once in the sensitivity table")
                self.assertEqual(rows[0]["api_servers"],
                                 r["machines"]["api_servers"])

    def test_sensitivity_rows_are_in_ascending_order(self):
        for ops in (20.0, 100.0, 1000.0):
            with self.subTest(ops=ops):
                rates = [x["agent_searches_per_s"]
                         for x in self.size(ops)["sensitivity"]]
                self.assertEqual(rates, sorted(rates))
                self.assertEqual(len(rates), len(set(rates)))

    def test_sensitivity_always_keeps_the_four_fixed_rates(self):
        rates = [x["agent_searches_per_s"] for x in self.size(1000.0)["sensitivity"]]
        for fixed in ps.SENSITIVITY_AGENT_RATES:
            self.assertIn(fixed, rates)

    def test_the_whole_result_is_json_serialisable(self):
        r = self.size(100.0)
        text = json.dumps(r)
        back = json.loads(text)
        self.assertEqual(back["machines"]["api_servers"],
                         r["machines"]["api_servers"])
        self.assertEqual(set(back), set(r))

    def test_the_json_carries_no_infinity_or_not_a_number(self):
        """Every number must survive a strict parser.

        Python writes a floating-point infinity as the bare token Infinity,
        which jq, JavaScript's JSON.parse and most other parsers reject. The
        default json.loads accepts it, so this test refuses it explicitly.
        """
        def refuse(token):
            raise AssertionError(f"the JSON contains the bare token {token}, "
                                 "which is not valid JSON")

        for ops in (0.5, 20.0, 100.0, 1000.0):
            with self.subTest(ops=ops):
                text = json.dumps(self.size(ops))
                json.loads(text, parse_constant=refuse)
                for token in ("Infinity", "NaN"):
                    self.assertNotIn(token, text)

    def test_the_run_name_is_carried_through(self):
        """run_name names the run - it is not one of the four provenance
        labels, and the README says so."""
        self.assertEqual(self.size(100.0, run_name="target")["run_name"],
                         "target")
        self.assertEqual(self.size(100.0)["run_name"], "custom")
        self.assertNotIn("label", self.size(100.0),
                         "no key called label, so nothing invites a reader to "
                         "mistake the run name for a provenance label")


# =============================================================================
# 13. Every command-line subcommand, run as a real command
# =============================================================================


class TestCommandLine(unittest.TestCase):

    def test_the_interpreter_exists(self):
        self.assertTrue(os.path.exists(PYTHON),
                        f"{PYTHON} is the interpreter the tests must use")

    def test_no_arguments_prints_help(self):
        proc = run_cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("tier", proc.stdout)
        self.assertIn("calc", proc.stdout)

    def test_help_flag(self):
        proc = run_cli("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("memmachine_sizing.py", proc.stdout)

    def test_tier_subcommand_for_each_tier(self):
        for name in ("pilot", "target", "scale"):
            with self.subTest(tier=name):
                proc = run_cli("tier", name)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(name.upper(), proc.stdout)
                self.assertIn("API server", proc.stdout)
                self.assertIn("PostgreSQL", proc.stdout)
                self.assertIn("Qdrant", proc.stdout)
                self.assertNotIn("Traceback", proc.stderr)

    def test_tier_json_matches_the_library(self):
        for name in ("pilot", "target", "scale"):
            with self.subTest(tier=name):
                proc = run_cli("tier", name, "--json")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                got = json.loads(proc.stdout)
                want = ps.size_deployment(ps.TIER_OPS_PER_S[name],
                                          run_name=name)
                self.assertEqual(got["machines"]["api_servers"],
                                 want["machines"]["api_servers"])
                self.assertEqual(got["demand"], want["demand"])
                self.assertEqual(got["run_name"], name)

    def test_tier_report_states_the_expected_server_count(self):
        proc = run_cli("tier", "target")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("3 API server(s)", proc.stdout)

    def test_calc_subcommand(self):
        proc = run_cli("calc", "--ops", "250")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("250", proc.stdout)

    def test_calc_json_matches_the_library(self):
        proc = run_cli("calc", "--ops", "250", "--add", "20", "--plain", "20",
                       "--agent", "60", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        got = json.loads(proc.stdout)
        want = ps.size_deployment(250.0, ps.TrafficMix(20.0, 20.0, 60.0))
        self.assertEqual(got["demand"], want["demand"])
        self.assertEqual(got["machines"]["api_servers"],
                         want["machines"]["api_servers"])

    def test_calc_honours_retention_dims_and_bytes_per_value(self):
        proc = run_cli("calc", "--ops", "100", "--retention-days", "30",
                       "--dims", "768", "--bytes-per-value", "2", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        got = json.loads(proc.stdout)
        self.assertEqual(got["inputs"]["retention_days"], 30)
        self.assertEqual(got["inputs"]["dims"], 768)
        self.assertEqual(got["inputs"]["bytes_per_value"], 2)
        want = ps.size_deployment(100.0, retention_days=30, dims=768,
                                  bytes_per_value=2)
        self.assertAlmostEqual(got["storage"]["hot_vector_ram_bytes"],
                               want["storage"]["hot_vector_ram_bytes"])

    def test_users_subcommand(self):
        proc = run_cli("users", "--humans", "5000", "--automated", "40")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("5,000", proc.stdout)
        self.assertIn("scale", proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)

    def test_users_subcommand_with_no_automated_clients(self):
        proc = run_cli("users", "--humans", "100")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("pilot", proc.stdout)

    def test_validate_subcommand_writes_a_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "numbers.json")
            proc = run_cli("validate", "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.exists(out))
            with open(out, encoding="utf-8") as handle:
                numbers = json.load(handle)
            self.assertGreater(len(numbers), 50)
            self.assertEqual(numbers["target.api_servers"], 3)
            self.assertEqual(numbers["pilot.api_servers"], 1)
            self.assertEqual(numbers["scale.api_servers"], 29)
            self.assertEqual(numbers["target.postgres_total_connections"], 420)
            self.assertEqual(numbers["scale.postgres_total_connections"], 4060)
            self.assertEqual(
                numbers["constants.api_searches_per_s_per_server"], 180.0)
            self.assertEqual(
                numbers["constants.api_usable_searches_per_server"], 108.0)

    def test_validate_prints_label_value_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "numbers.json")
            proc = run_cli("validate", "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("target.api_servers: 3", proc.stdout)
            self.assertIn("target.embeds_per_s: 355", proc.stdout)

    def test_validate_covers_all_three_tiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "numbers.json")
            proc = run_cli("validate", "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as handle:
                numbers = json.load(handle)
            for name, want in TIER_EXPECTATIONS.items():
                with self.subTest(tier=name):
                    self.assertEqual(numbers[f"{name}.api_servers"],
                                     want["api_servers"])
                    self.assertEqual(numbers[f"{name}.qdrant_servers"],
                                     want["qdrant_nodes"])
                    self.assertEqual(numbers[f"{name}.agent_gpu_cards"],
                                     want["agent_cards"])
                    self.assertEqual(numbers[f"{name}.embed_gpu_cards_low"],
                                     want["embed_cards_low"])
                    self.assertEqual(numbers[f"{name}.embed_gpu_cards_high"],
                                     want["embed_cards_high"])

    def test_validate_honours_a_custom_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "numbers.json")
            proc = run_cli("validate", "--add", "20", "--plain", "20",
                           "--agent", "60", "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as handle:
                numbers = json.load(handle)
            self.assertEqual(numbers["target.mix_agent"], 60.0)
            self.assertEqual(numbers["target.api_servers"], 13)

    def test_serve_subcommand_help(self):
        proc = run_cli("serve", "--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--port", proc.stdout)

    def test_every_subcommand_is_reachable(self):
        proc = run_cli("--help")
        for name in ("tier", "calc", "users", "validate", "serve"):
            with self.subTest(subcommand=name):
                self.assertIn(name, proc.stdout)


# =============================================================================
# 14. The web server
# =============================================================================


class ServedCalculator(unittest.TestCase):
    """Start the server on a free port, ask it questions, then stop it.

    A base class rather than one test class's own setUp, so that more than one
    group of tests can drive the real server over a real socket without a
    second copy of the start-up dance.
    """

    proc = None
    base = None

    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.proc = subprocess.Popen(
            [PYTHON, MODULE_PATH, "serve", "--host", "127.0.0.1",
             "--port", str(cls.port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 30.0
        last = None
        while time.monotonic() < deadline:
            if cls.proc.poll() is not None:
                out, err = cls.proc.communicate()
                raise AssertionError(
                    f"the server exited straight away\nstdout: {out}\n"
                    f"stderr: {err}")
            try:
                with urllib.request.urlopen(f"{cls.base}/healthz", timeout=2):
                    return
            except Exception as exc:      # not up yet
                last = exc
                time.sleep(0.1)
        cls.tearDownClass()
        raise AssertionError(f"the server never answered: {last}")

    @classmethod
    def tearDownClass(cls):
        if cls.proc is not None and cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
                cls.proc.communicate(timeout=10)
        cls.proc = None

    def fetch(self, path):
        """Return (status, content type, body text), even for an error status."""
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as resp:
                return (resp.status, resp.headers.get("Content-Type"),
                        resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return (err.code, err.headers.get("Content-Type"),
                    err.read().decode("utf-8"))

    def json(self, path) -> dict:
        """The JSON endpoint's answer, refusing anything that is not a 200."""
        status, ctype, body = self.fetch(path)
        self.assertEqual(status, 200, body)
        self.assertIn("application/json", ctype)
        return json.loads(body)


class TestWebServer(ServedCalculator):
    """The page and the JSON endpoint, asked over a real socket."""

    def test_home_page_status_and_content_type(self):
        status, ctype, body = self.fetch("/")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "text/html; charset=utf-8")
        self.assertIn("<!doctype html>", body.lower())
        self.assertIn("MemMachine sizing calculator", body)

    def test_home_page_carries_the_form(self):
        _, _, body = self.fetch("/")
        for field in ("ops", "add", "plain", "agent", "retention_days", "dims",
                      "bytes_per_value", "node_gb", "humans", "automated",
                      "human_mix", "automated_mix"):
            with self.subTest(field=field):
                self.assertIn(f'name="{field}"', body)

    def test_home_page_shows_the_default_target_tier_answer(self):
        _, _, body = self.fetch("/")
        self.assertIn("Machines", body)
        self.assertIn("PostgreSQL", body)

    def test_home_page_with_explicit_settings(self):
        status, ctype, body = self.fetch(
            "/?ops=1000&add=45&plain=45&agent=10&retention_days=90"
            "&dims=1024&bytes_per_value=1")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "text/html; charset=utf-8")
        self.assertIn("29", body, "the scale tier needs 29 API servers")

    def test_json_endpoint_status_and_content_type(self):
        status, ctype, _ = self.fetch(
            "/api/calc?ops=100&add=45&plain=45&agent=10&retention_days=90"
            "&dims=1024&bytes_per_value=1")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "application/json; charset=utf-8")

    def test_json_endpoint_numbers_match_the_library(self):
        _, _, body = self.fetch(
            "/api/calc?ops=100&add=45&plain=45&agent=10&retention_days=90"
            "&dims=1024&bytes_per_value=1")
        got = json.loads(body)
        want = ps.size_deployment(100.0, ps.TrafficMix(45.0, 45.0, 10.0),
                                  retention_days=90.0, dims=1024,
                                  bytes_per_value=1.0)
        self.assertEqual(got["demand"], want["demand"])
        self.assertEqual(got["storage"], want["storage"])
        self.assertEqual(got["postgres"], want["postgres"])
        self.assertEqual(got["network"], want["network"])
        self.assertEqual(got["users"], want["users"])
        self.assertEqual(got["machines"]["api_servers"],
                         want["machines"]["api_servers"])
        self.assertEqual(got["machines"]["qdrant_servers"],
                         want["machines"]["qdrant_servers"])

    def test_json_endpoint_matches_the_hand_worked_target_tier(self):
        _, _, body = self.fetch("/api/calc?ops=100")
        got = json.loads(body)
        want = TIER_EXPECTATIONS["target"]
        self.assertEqual(got["machines"]["api_servers"], want["api_servers"])
        self.assertEqual(got["machines"]["agent_gpu_cards"],
                         want["agent_cards"])
        self.assertEqual(got["postgres"]["total_connections"],
                         want["pg_total_connections"])
        self.assertAlmostEqual(got["demand"]["embeds_per_s"], want["embeds"])

    def test_json_endpoint_with_a_custom_mix(self):
        _, _, body = self.fetch("/api/calc?ops=100&add=20&plain=20&agent=60")
        got = json.loads(body)
        self.assertAlmostEqual(got["demand"]["agent_searches_per_s"], 60.0)
        self.assertEqual(got["machines"]["api_servers"], 13)

    def test_json_endpoint_defaults_to_the_target_tier(self):
        _, _, body = self.fetch("/api/calc")
        got = json.loads(body)
        self.assertAlmostEqual(got["inputs"]["ops_per_s"], 100.0)
        self.assertEqual(got["machines"]["api_servers"], 3)

    def test_json_endpoint_rejects_a_mix_that_does_not_sum_to_100(self):
        status, ctype, body = self.fetch(
            "/api/calc?ops=100&add=45&plain=45&agent=5")
        self.assertEqual(status, 400)
        self.assertEqual(ctype, "application/json; charset=utf-8")
        payload = json.loads(body)
        self.assertIn("error", payload)
        self.assertNotIn("Traceback", body)

    def test_json_endpoint_rejects_a_negative_operation_rate(self):
        status, _, body = self.fetch("/api/calc?ops=-5")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_json_endpoint_rejects_text_where_a_number_belongs(self):
        status, _, body = self.fetch("/api/calc?ops=lots")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    NON_FINITE_QUERIES = (
        "/api/calc?ops=nan",
        "/api/calc?ops=inf",
        "/api/calc?ops=1e300",
        "/api/calc?ops=100&retention_days=inf",
        "/api/calc?ops=100&retention_days=nan",
        "/api/calc?ops=100&bytes_per_value=nan",
        "/api/calc?ops=100&dims=nan",
        "/api/calc?ops=100&dims=1e400",
        "/api/calc?ops=100&add=nan&plain=45&agent=10",
    )

    def test_json_endpoint_answers_every_non_finite_query(self):
        """Each of these used to close the connection with no HTTP response."""
        for path in self.NON_FINITE_QUERIES:
            with self.subTest(path=path):
                status, ctype, body = self.fetch(path)
                self.assertEqual(status, 400,
                                 "a bad query must get an HTTP 400, not a "
                                 "dropped connection")
                self.assertEqual(ctype, "application/json; charset=utf-8")
                self.assertIn("error", json.loads(body))
                self.assertNotIn("Traceback", body)

    def test_home_page_answers_every_non_finite_query(self):
        for path in self.NON_FINITE_QUERIES:
            page = path.replace("/api/calc?", "/?")
            with self.subTest(path=page):
                status, ctype, body = self.fetch(page)
                self.assertEqual(status, 400)
                self.assertEqual(ctype, "text/html; charset=utf-8")
                self.assertIn("Cannot calculate", body)
                self.assertNotIn("Traceback", body)

    def test_the_server_is_still_running_after_a_non_finite_query(self):
        for path in self.NON_FINITE_QUERIES:
            self.fetch(path)
        status, _, _ = self.fetch("/healthz")
        self.assertEqual(status, 200)

    def test_home_page_reports_a_bad_mix_in_the_page(self):
        status, ctype, body = self.fetch("/?ops=100&add=45&plain=45&agent=5")
        self.assertEqual(status, 400)
        self.assertEqual(ctype, "text/html; charset=utf-8")
        self.assertIn("Cannot calculate", body)
        self.assertNotIn("Traceback", body)

    # Every flag that changes what is sized, on any subcommand, against the
    # box that sets the same thing on the form. Nothing may be reachable by
    # flag but not by the form.
    FLAGS_AND_BOXES: ClassVar[tuple] = (
        ("--ops", "ops"), ("--add", "add"), ("--plain", "plain"),
        ("--agent", "agent"), ("--retention-days", "retention_days"),
        ("--dims", "dims"), ("--bytes-per-value", "bytes_per_value"),
        ("--node-gb", "node_gb"),
        # The users subcommand. --automated counts callers that are programs;
        # --agent above is the agent-mode share of the requests.
        ("--humans", "humans"), ("--automated", "automated"),
        ("--human-mix", "human_mix"), ("--automated-mix", "automated_mix"),
    )

    def test_the_form_offers_every_model_input_the_command_line_offers(self):
        """No input may be reachable by flag but not by the form."""
        _, _, body = self.fetch("/")
        for _, key in self.FLAGS_AND_BOXES:
            with self.subTest(field=key):
                self.assertIn(f'name="{key}"', body)

    def test_every_flag_in_that_pairing_is_a_flag_the_command_line_takes(self):
        """The other direction: the pairing must not invent a flag.

        A box paired with a flag nobody can type would let the form and the
        command line drift apart while this test still passed.
        """
        helps = {name: run_cli(name, "--help").stdout
                 for name in ("tier", "calc", "users")}
        for flag, key in self.FLAGS_AND_BOXES:
            with self.subTest(flag=flag):
                self.assertTrue(
                    any(flag in text for text in helps.values()),
                    f"{flag} is paired with the {key} box but no subcommand "
                    f"offers it")

    def test_the_node_size_box_is_labelled_in_plain_english(self):
        _, _, body = self.fetch("/")
        self.assertIn("RAM per vector-store machine, GB", body)

    def test_the_node_size_box_starts_empty(self):
        _, _, body = self.fetch("/")
        self.assertIn('name="node_gb" value=""', body)

    def test_json_endpoint_forces_two_512_gb_machines(self):
        status, _, body = self.fetch("/api/calc?ops=100&node_gb=512")
        self.assertEqual(status, 200, body)
        m = json.loads(body)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 512)
        self.assertEqual(m["qdrant_servers"], 2)
        self.assertTrue(m["qdrant_node_ram_gb_forced"])

    def test_json_endpoint_accepts_the_other_offered_sizes(self):
        for node_gb, nodes in ((256, 3), (512, 2), (768, 1)):
            with self.subTest(node_gb=node_gb):
                status, _, body = self.fetch(
                    f"/api/calc?ops=100&node_gb={node_gb}")
                self.assertEqual(status, 200, body)
                m = json.loads(body)["machines"]
                self.assertEqual(m["qdrant_node_ram_gb"], node_gb)
                self.assertEqual(m["qdrant_servers"], nodes)

    def test_json_endpoint_accepts_a_size_that_is_not_one_of_the_three(self):
        status, _, body = self.fetch("/api/calc?ops=100&node_gb=1024")
        self.assertEqual(status, 200, body)
        m = json.loads(body)["machines"]
        self.assertEqual(m["qdrant_node_ram_gb"], 1024)
        self.assertEqual(m["qdrant_servers"], 1)

    def test_an_empty_node_size_box_keeps_the_automatic_choice(self):
        for query in ("/api/calc?ops=100",
                      "/api/calc?ops=100&node_gb=",
                      "/api/calc?ops=100&node_gb=automatic",
                      "/api/calc?ops=100&node_gb=auto"):
            with self.subTest(query=query):
                status, _, body = self.fetch(query)
                self.assertEqual(status, 200, body)
                m = json.loads(body)["machines"]
                self.assertEqual(m["qdrant_node_ram_gb"], 768)
                self.assertEqual(m["qdrant_servers"], 1)
                self.assertFalse(m["qdrant_node_ram_gb_forced"])

    def test_json_endpoint_rejects_a_node_size_of_zero(self):
        status, ctype, body = self.fetch("/api/calc?ops=100&node_gb=0")
        self.assertEqual(status, 400)
        self.assertIn("application/json", ctype)
        self.assertIn("greater than zero", json.loads(body)["error"])

    def test_json_endpoint_rejects_a_negative_node_size(self):
        status, _, body = self.fetch("/api/calc?ops=100&node_gb=-256")
        self.assertEqual(status, 400)
        self.assertIn("greater than zero", json.loads(body)["error"])

    def test_json_endpoint_rejects_text_in_the_node_size_box(self):
        for value in ("big", "512gb", "five hundred"):
            with self.subTest(value=value):
                status, ctype, body = self.fetch(
                    f"/api/calc?ops=100&node_gb={value.replace(' ', '%20')}")
                self.assertEqual(status, 400)
                self.assertIn("application/json", ctype)
                self.assertIn("error", json.loads(body))
                self.assertNotIn("Traceback", body)

    def test_json_endpoint_rejects_a_node_size_that_is_not_finite(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                status, _, body = self.fetch(
                    f"/api/calc?ops=100&node_gb={value}")
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))

    def test_the_page_shows_the_forced_node_size(self):
        status, _, body = self.fetch("/?ops=100&node_gb=512")
        self.assertEqual(status, 200)
        self.assertIn("512 GB, forced", body)
        self.assertIn('name="node_gb" value="512"', body)

    def test_the_page_reports_a_bad_node_size_in_the_page(self):
        status, ctype, body = self.fetch("/?ops=100&node_gb=0")
        self.assertEqual(status, 400)
        self.assertIn("text/html", ctype)
        self.assertIn("Cannot calculate", body)
        self.assertIn("greater than zero", body)
        self.assertNotIn("Traceback", body)

    def test_the_server_is_still_running_after_a_bad_node_size(self):
        self.fetch("/api/calc?ops=100&node_gb=0")
        self.fetch("/api/calc?ops=100&node_gb=big")
        status, _, _ = self.fetch("/healthz")
        self.assertEqual(status, 200)

    def test_health_endpoint(self):
        status, ctype, body = self.fetch("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "text/plain; charset=utf-8")
        self.assertEqual(body.strip(), "ok")

    def test_unknown_path_is_a_404(self):
        status, _, body = self.fetch("/nowhere")
        self.assertEqual(status, 404)
        self.assertIn("not found", body)

    def test_the_page_needs_no_outside_files(self):
        """No <script src>, no <link href> - the page must stand alone."""
        _, _, body = self.fetch("/")
        self.assertNotIn("<script", body.lower())
        self.assertNotIn("<link", body.lower())
        self.assertNotIn("http://", body.replace("http://127.0.0.1", ""))

    def test_the_server_is_still_running_after_a_bad_request(self):
        self.fetch("/api/calc?ops=lots")
        status, _, _ = self.fetch("/healthz")
        self.assertEqual(status, 200)

    # -- an empty box is a blank answer, not a request for the default -------
    #
    # The command line refuses to run calc without --ops. The form used to
    # invent 100 operations per second for a box the reader had cleared, and
    # print a full hardware plan for it with no message at all.

    FULL_FORM = ("ops=100&add=45&plain=45&agent=10&retention_days=90"
                 "&dims=1024&bytes_per_value=1&node_gb=")

    def submit(self, **changes):
        """Submit the form as a browser does: all eight boxes, some changed."""
        boxes = dict(pair.split("=", 1)
                     for pair in self.FULL_FORM.split("&"))
        boxes.update(changes)
        return "&".join(f"{k}={v}" for k, v in boxes.items())

    def test_an_empty_design_peak_box_is_reported_and_not_defaulted(self):
        status, ctype, body = self.fetch("/?" + self.submit(ops=""))
        self.assertEqual(status, 400,
                         "a blank required box must not return a report")
        self.assertIn("text/html", ctype)
        self.assertIn("design peak box is empty", body)
        self.assertNotIn("Total ordinary servers", body,
                         "no hardware plan may be printed for a blank box")
        self.assertIn('name="ops" value=""', body,
                      "the box must stay blank, not be refilled with 100")

    def test_an_empty_design_peak_box_is_reported_by_the_json_endpoint(self):
        status, _, body = self.fetch("/api/calc?ops=&add=45")
        self.assertEqual(status, 400)
        self.assertIn("design peak box is empty", json.loads(body)["error"])

    def test_every_box_that_must_be_answered_is_named_when_left_blank(self):
        wanted = {
            "ops": "design peak box is empty",
            "add": "adds per 100 operations box is empty",
            "plain": "plain searches per 100 operations box is empty",
            "agent": "agent-mode searches per 100 operations box is empty",
            "retention_days": "retention, days box is empty",
            "dims": "vector dimensions box is empty",
            "bytes_per_value": "bytes per number box is empty",
        }
        for field, message in wanted.items():
            with self.subTest(field=field):
                status, _, body = self.fetch("/?" + self.submit(**{field: ""}))
                self.assertEqual(status, 400)
                self.assertIn(message, body)

    def test_a_box_holding_only_spaces_counts_as_blank(self):
        status, _, body = self.fetch("/?" + self.submit(ops="%20%20%20"))
        self.assertEqual(status, 400)
        self.assertIn("design peak box is empty", body)

    def test_a_parameter_left_out_altogether_still_takes_the_default(self):
        """The deliberate difference: an absent parameter is not a blank box.

        A bare /api/calc call names no parameters at all, and must keep
        working exactly as the command line does when a flag is left off.
        """
        status, _, body = self.fetch("/api/calc?ops=100")
        self.assertEqual(status, 200)
        got = json.loads(body)
        self.assertEqual(got["inputs"]["retention_days"],
                         ps.DEFAULT_RETENTION_DAYS)
        self.assertEqual(got["inputs"]["dims"], ps.DEFAULT_VECTOR_DIMS)
        self.assertAlmostEqual(got["inputs"]["mix"]["add"], ps.DEFAULT_MIX_ADD)

    def test_an_empty_node_size_box_still_means_the_automatic_choice(self):
        """The one box where empty is itself an answer."""
        status, _, body = self.fetch("/?" + self.submit(node_gb=""))
        self.assertEqual(status, 200)
        self.assertIn("chosen automatically", body)

    # -- the boxes read the same on the first view as on every later view ----

    def test_the_first_view_shows_whole_number_defaults_without_a_zero(self):
        _, _, body = self.fetch("/")
        for field, shown in (("ops", "100"), ("add", "45"), ("plain", "45"),
                             ("agent", "10"), ("retention_days", "90"),
                             ("dims", "1024"), ("bytes_per_value", "1")):
            with self.subTest(field=field):
                self.assertIn(f'name="{field}" value="{shown}"', body)
        self.assertNotIn('value="100.0"', body)
        self.assertNotIn('value="45.0"', body)

    def test_the_boxes_read_the_same_before_and_after_a_submit(self):
        _, _, first = self.fetch("/")
        _, _, later = self.fetch("/?" + self.submit())
        for field in ("ops", "add", "plain", "agent", "retention_days",
                      "dims", "bytes_per_value", "node_gb"):
            with self.subTest(field=field):
                self.assertEqual(self.box_value(first, field),
                                 self.box_value(later, field))

    @staticmethod
    def box_value(body, field):
        mark = f'name="{field}" value="'
        start = body.index(mark) + len(mark)
        return body[start:body.index('"', start)]

    # -- a bad number names its own box --------------------------------------

    def test_a_bad_number_names_the_box_and_quotes_what_was_typed(self):
        status, _, body = self.fetch("/?" + self.submit(ops="abc"))
        self.assertEqual(status, 400)
        self.assertIn("design peak box says", body)
        self.assertIn("abc", body)
        self.assertNotIn("every field must be a number", body)

    def test_each_box_names_itself_rather_than_all_eight(self):
        for field, name in (("dims", "vector dimensions box says"),
                            ("bytes_per_value", "bytes per number box says"),
                            ("node_gb",
                             "RAM per vector-store machine box says")):
            with self.subTest(field=field):
                status, _, body = self.fetch(
                    "/?" + self.submit(**{field: "lots"}))
                self.assertEqual(status, 400)
                self.assertIn(name, body)

    def test_a_thousands_comma_gets_advice_of_its_own(self):
        status, _, body = self.fetch("/api/calc?ops=1,000")
        self.assertEqual(status, 400)
        message = json.loads(body)["error"]
        self.assertIn("design peak box says", message)
        self.assertIn("1,000", message)
        self.assertIn("comma", message)

    # -- the message is announced, and points at the box that caused it ------

    def test_the_error_box_is_announced_to_a_screen_reader(self):
        _, _, body = self.fetch("/?" + self.submit(ops="abc"))
        self.assertIn('class="err"', body)
        self.assertIn('role="alert"', body)
        self.assertIn(f'id="{ps.FORM_ERROR_ID}"', body)

    def test_the_offending_box_is_marked_invalid_and_takes_the_focus(self):
        _, _, body = self.fetch("/?" + self.submit(ops="abc"))
        start = body.index('id="ops"')
        box = body[start:body.index(">", start)]
        self.assertIn('aria-invalid="true"', box)
        self.assertIn(f'aria-describedby="{ps.FORM_ERROR_ID}"', box)
        self.assertIn("autofocus", box)
        other = body[body.index('id="dims"'):]
        other = other[:other.index(">")]
        self.assertNotIn("aria-invalid", other,
                         "only the box at fault may be marked invalid")

    def test_the_message_comes_before_the_form_in_the_page(self):
        _, _, body = self.fetch("/?" + self.submit(ops="abc"))
        self.assertLess(body.index('class="err"'), body.index("<form"),
                        "a reader must meet the message before the boxes")

    def test_a_good_answer_marks_no_box_invalid(self):
        _, _, body = self.fetch("/?" + self.submit())
        self.assertNotIn("aria-invalid", body)
        self.assertNotIn("autofocus", body)

    # -- a clean page load leaves nothing in the browser console -------------

    def test_the_favicon_request_is_answered_with_no_content(self):
        """Every browser asks for a tab icon; a 404 is a console error."""
        status, _, body = self.fetch("/favicon.ico")
        self.assertEqual(status, 204)
        self.assertEqual(body, "")

    # -- a wide table scrolls inside itself, not by dragging the page --------

    def test_every_table_sits_inside_a_scrolling_container(self):
        _, _, body = self.fetch("/")
        self.assertEqual(body.count("<table>"),
                         body.count('<div class="tablewrap"><table>'),
                         "a table outside a scrolling box drags the whole "
                         "page sideways on a narrow screen")
        self.assertIn(".tablewrap { overflow-x: auto;", body)

    def test_a_long_example_url_is_allowed_to_break(self):
        _, _, body = self.fetch("/")
        style = body[body.index("<style>"):body.index("</style>")]
        code_rule = style[style.index("code {"):]
        self.assertIn("overflow-wrap: anywhere", code_rule)

    # -- the page never names a command-line flag the reader did not type ----

    def test_the_page_names_the_box_that_forced_the_node_size(self):
        _, _, body = self.fetch("/?" + self.submit(node_gb="512"))
        self.assertIn("The size was forced with the RAM per vector-store "
                      "machine box.", body)
        self.assertNotIn("--node-gb", body)

    def test_the_inputs_row_does_not_claim_an_automatic_choice(self):
        _, _, body = self.fetch("/?" + self.submit(node_gb="512"))
        self.assertIn("512 GB, forced", body)
        self.assertIn("assumption - set by hand", body)
        self.assertNotIn("the automatic choice buys the least total RAM", body)

    # -- values the reader typed are never quietly changed -------------------

    def test_a_fractional_dimension_is_refused_rather_than_cut_down(self):
        status, _, body = self.fetch("/?" + self.submit(dims="1024.7"))
        self.assertEqual(status, 400)
        self.assertIn("vector dimensions is 1024.7", body)
        self.assertIn("whole number", body)

    def test_a_design_peak_too_large_to_size_is_refused_by_name(self):
        status, _, body = self.fetch("/api/calc?ops=1e307")
        self.assertEqual(status, 400)
        message = json.loads(body)["error"]
        self.assertIn("the design peak is 1e+307", message)
        self.assertIn("1,000,000,000", message)
        self.assertNotIn("inf", message,
                         "the reader typed 1e307, not infinity")

    def test_a_tiny_design_peak_still_orders_a_whole_machine(self):
        _, _, body = self.fetch("/api/calc?ops=1e-9")
        got = json.loads(body)
        self.assertEqual(got["machines"]["api_servers"], 1)
        self.assertEqual(got["machines"]["qdrant_servers"], 1)


# =============================================================================
# 15. The module uses nothing outside the standard library
# =============================================================================


class TestStandardLibraryOnly(unittest.TestCase):

    def top_level_imports(self):
        with open(MODULE_PATH, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=MODULE_PATH)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:          # a relative import, so a local package
                    names.add("<relative import>")
                elif node.module:
                    names.add(node.module.split(".")[0])
        return names

    def test_every_import_is_in_the_standard_library(self):
        allowed = set(sys.stdlib_module_names) | {"__future__"}
        outside = sorted(name for name in self.top_level_imports()
                         if name not in allowed)
        self.assertEqual(outside, [],
                         f"the calculator imports something outside the "
                         f"standard library: {outside}")

    def test_no_relative_imports(self):
        self.assertNotIn("<relative import>", self.top_level_imports(),
                         "the calculator must be a single self-contained file")

    def test_the_module_imports_with_no_third_party_packages_installed(self):
        """Import it in a fresh interpreter with site-packages switched off."""
        proc = subprocess.run(
            [PYTHON, "-S", "-c",
             (f"import sys; sys.path.insert(0, {HERE!r}); "
              "import memmachine_sizing; "
              "print(memmachine_sizing.API_SEARCHES_PER_S_PER_SERVER)")],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "180.0")

    def test_the_imports_it_actually_uses(self):
        expected = {"__future__", "argparse", "difflib", "json", "math",
                    "sys", "dataclasses", "html", "http", "urllib"}
        self.assertEqual(self.top_level_imports(), expected)


# =============================================================================
# 16. Cross-checks that tie the whole model together
# =============================================================================


class TestModelConsistency(ModelBaseTest):

    def test_ten_times_the_traffic_is_ten_times_the_demand(self):
        one = self.size(100.0)["demand"]
        ten = self.size(1000.0)["demand"]
        for key in ("adds_per_s", "plain_searches_per_s", "embeds_per_s",
                    "vector_searches_per_s", "postgres_statements_per_s"):
            with self.subTest(key=key):
                self.assertClose(ten[key], one[key] * 10, key)

    def test_machine_counts_never_shrink_as_traffic_grows(self):
        previous = None
        for ops in (1.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 3000.0):
            m = self.size(ops)["machines"]
            if previous is not None:
                self.assertGreaterEqual(m["api_servers"],
                                        previous["api_servers"])
                self.assertGreaterEqual(m["embed_gpu_cards_high"],
                                        previous["embed_gpu_cards_high"])
                self.assertGreaterEqual(m["qdrant_servers"],
                                        previous["qdrant_servers"])
            previous = m

    def test_the_reranker_adds_no_gpu(self):
        """rrf-hybrid runs on the API server's own CPU; its cost is in the 180/s
        anchor already, so nothing in the model prices a reranker card."""
        source = open(MODULE_PATH, encoding="utf-8").read().lower()
        self.assertIn("rrf-hybrid", source,
                      "the reranker must be named, so a reader knows which one "
                      "the 180/s anchor was measured with")
        m = self.size(100.0)["machines"]
        gpu_keys = [k for k in m if "gpu" in k]
        self.assertEqual(sorted(gpu_keys),
                         ["agent_gpu_cards", "agent_gpu_spare",
                          "embed_gpu_cards_high", "embed_gpu_cards_low",
                          "embed_gpu_spare"],
                         "the only GPU roles are embedding and the agent model")

    def test_the_report_renders_for_every_tier(self):
        for name in ("pilot", "target", "scale"):
            with self.subTest(tier=name):
                r = self.size(ps.TIER_OPS_PER_S[name], run_name=name)
                text = ps.render_report(r, name.upper())
                self.assertIn("Demand per second", text)
                self.assertIn("Machines", text)
                self.assertIn("Sensitivity", text)

    # The two tables of what-ifs carry no per-row label column, because each
    # of their rows is an alternative rather than a published figure. The
    # README and the note above each table both say so.
    WHAT_IF_SECTIONS = ("Qdrant node choice",
                        "Sensitivity: what the agent-mode quota costs")

    def test_every_row_of_every_report_section_carries_a_label(self):
        """measured, derived, estimate or assumption on EVERY row, not just
        the first row of each section, and at every tier - the scale tier has
        an extra sensitivity row and a different vector-store choice."""
        wanted = ("measured", "derived", "estimate", "assumption")
        checked = 0
        for ops in (20.0, 100.0, 1000.0):
            for section in ps.report_sections(self.size(ops)):
                if section["title"] in self.WHAT_IF_SECTIONS:
                    continue
                for index, row in enumerate(section["rows"]):
                    with self.subTest(ops=ops, section=section["title"],
                                      row=index):
                        last = str(row[-1]).lower()
                        self.assertTrue(
                            any(word in last for word in wanted),
                            f"row label {last!r} is not one of "
                            f"measured / derived / estimate / assumption")
                        checked += 1
        self.assertGreater(checked, 40,
                           "every row of every labelled section must be "
                           "inspected, not one row per section")

    def test_the_two_what_if_tables_say_in_their_note_where_their_numbers_come_from(self):
        """They are the only sections with no label column, so the README's
        claim about labels holds only if each explains itself in its note."""
        seen = []
        for section in ps.report_sections(self.size(100.0)):
            if section["title"] not in self.WHAT_IF_SECTIONS:
                continue
            seen.append(section["title"])
            with self.subTest(section=section["title"]):
                self.assertNotIn("Label", section["headers"])
                self.assertIn("what-if", section["note"])
                self.assertIn("derived", section["note"])
        self.assertEqual(sorted(seen), sorted(self.WHAT_IF_SECTIONS))

    def test_the_report_footer_names_the_two_tables_that_carry_no_label(self):
        text = ps.render_report(self.size(100.0), "TARGET")
        self.assertIn("assumption = a planning choice, not a finding.", text)
        self.assertIn("carry no label", text)
        self.assertIn("Qdrant node choice table and the sensitivity table",
                      text)

    def test_no_network_row_is_labelled_plainly_derived(self):
        """Nothing measured enters the network figures.

        Every input to them is one of the per-call byte-size estimates, so a
        bare "derived" label there would claim a measurement that does not
        exist.
        """
        r = self.size(100.0)
        for section in ps.report_sections(r):
            if section["title"] != "Network":
                continue
            for row in section["rows"]:
                with self.subTest(row=row[0]):
                    self.assertNotEqual(row[-1].strip().lower(), "derived")


# =============================================================================
# 17. Values the reader gave are never quietly changed, and no positive
#     workload comes back as no machines at all
# =============================================================================


class TestWorkAlwaysCostsAMachine(ModelBaseTest):
    """A positive workload has to run somewhere.

    ceil_up used to subtract a flat 1e-9 to absorb floating-point dust, which
    is dust next to three machines but is larger than the whole answer for a
    very small workload: any work below about 1.08e-7 searches per second came
    back as zero machines. The dust allowance is a fraction of the answer now.
    """

    def test_a_workload_far_below_one_machine_still_needs_one(self):
        for work in (1e-12, 1e-9, 1e-7, 1e-3, 0.5):
            with self.subTest(work=work):
                self.assertEqual(ps.api_servers_for_work(work), 1)

    def test_no_work_still_needs_no_machine(self):
        self.assertEqual(ps.api_servers_for_work(0.0), 0)
        self.assertEqual(ps.ceil_up(0.0, 108.0), 0)
        self.assertEqual(ps.ceil_up(-5.0, 108.0), 0)

    def test_exact_multiples_still_do_not_round_up(self):
        """The reason the dust allowance exists at all."""
        self.assertEqual(ps.ceil_up(324.0, 108.0), 3)
        self.assertEqual(ps.ceil_up(108.0, 108.0), 1)
        self.assertEqual(ps.ceil_up(0.1 + 0.2, 0.1), 3)

    def test_a_hair_above_a_multiple_still_rounds_up(self):
        self.assertEqual(ps.ceil_up(324.1, 108.0), 4)

    def test_a_tiny_design_peak_orders_a_real_deployment(self):
        r = self.size(1e-9)
        self.assertEqual(r["machines"]["api_servers"], 1)
        self.assertEqual(r["machines"]["qdrant_servers"], 1)
        self.assertGreaterEqual(r["machines"]["qdrant_total_ram_gb"], 1)
        self.assertGreaterEqual(r["machines"]["embed_gpu_cards_low"],
                                1 + SPARE_CARDS)

    def test_a_deployment_that_searches_vectors_owns_somewhere_to_search(self):
        """Stored bytes are not the only reason to own a vector-store machine.

        A search-only mix stores nothing, and so does a retention of zero days,
        which the report itself calls a placeholder. Both used to order zero
        vector-store machines beside hundreds of vector searches a second.
        """
        cases = {
            "search only": {"mix": ps.TrafficMix(0.0, 90.0, 10.0)},
            "retention zero": {"retention_days": 0},
            "search only and retention zero": {
                "mix": ps.TrafficMix(0.0, 90.0, 10.0), "retention_days": 0},
            "adds only": {"mix": ps.TrafficMix(100.0, 0.0, 0.0),
                          "retention_days": 0},
        }
        for label, kwargs in cases.items():
            with self.subTest(case=label):
                r = self.size(100.0, **kwargs)
                d, m = r["demand"], r["machines"]
                self.assertGreater(d["vector_searches_per_s"]
                                   + d["vector_writes_per_s"], 0.0)
                self.assertGreaterEqual(m["qdrant_servers"], 1)
                self.assertGreaterEqual(m["total_cpu_servers"],
                                        m["api_servers"]
                                        + m["postgres_servers"] + 1)

    def test_the_written_order_never_names_zero_vector_store_machines(self):
        headline = ps.render_tier_headline(self.size(100.0, retention_days=0))
        self.assertIn("1 Qdrant server(s)", headline)


class TestVectorDimensionsMustBeWhole(ModelBaseTest):
    """A vector cannot hold part of a number.

    The report itself warns that the dimension count must be fixed before the
    first episode is ingested, so cutting 1024.7 down to 1024 without saying so
    is exactly the kind of change worth refusing.
    """

    def test_a_fraction_of_a_dimension_is_refused(self):
        with self.assertRaises(ps.SizingError) as caught:
            self.size(100.0, dims=1024.7)
        message = str(caught.exception)
        self.assertIn("1024.7", message)
        self.assertIn("whole number", message)

    def test_a_whole_number_written_as_a_float_is_accepted(self):
        r = self.size(100.0, dims=1024.0)
        self.assertEqual(r["inputs"]["dims"], 1024)
        self.assertIsInstance(r["inputs"]["dims"], int)

    def test_a_fraction_is_not_sized_as_the_whole_number_below_it(self):
        with self.assertRaises(ps.SizingError):
            self.size(100.0, dims=1024.7)

    def test_the_message_for_a_negative_dimension_is_unchanged(self):
        with self.assertRaises(ps.SizingError) as caught:
            self.size(100.0, dims=-8)
        self.assertIn("vector dimensions is -8", str(caught.exception))


class TestSanityBounds(ModelBaseTest):
    """A number far larger than any deployment is refused by name.

    1e307 is a finite number, so it passed the finite check at the front of
    size_deployment and only overflowed to infinity later, deep in the byte
    arithmetic - and came back as a complaint about "inf", a value the reader
    never typed. The bounds change no machine count; they only decide what is
    refused, and they say what the limit is.
    """

    def test_the_bounds_are_far_above_the_largest_tier(self):
        self.assertGreater(ps.MAX_OPS_PER_S, ps.TIER_OPS_PER_S["scale"] * 1000)
        self.assertGreater(ps.MAX_VECTOR_DIMS, ps.DEFAULT_VECTOR_DIMS * 100)
        self.assertGreater(ps.MAX_RETENTION_DAYS,
                           ps.DEFAULT_RETENTION_DAYS * 100)

    def test_a_design_peak_above_the_bound_is_refused(self):
        for ops in (ps.MAX_OPS_PER_S * 1.001, 1e12, 1e307):
            with self.subTest(ops=ops):
                with self.assertRaises(ps.SizingError) as caught:
                    self.size(ops)
                message = str(caught.exception)
                self.assertIn("the design peak is", message)
                self.assertIn("1,000,000,000 operations/s", message)
                self.assertNotIn("inf", message)

    def test_a_design_peak_at_the_bound_is_still_sized(self):
        r = self.size(ps.MAX_OPS_PER_S)
        self.assertGreater(r["machines"]["api_servers"], 0)
        self.assertTrue(math.isfinite(r["storage"]["hot_vector_ram_bytes"]))

    # One above each bound, and the limit each one prints.
    JUST_OVER_EACH_BOUND: ClassVar[tuple] = (
        ("ops_per_s", ps.MAX_OPS_PER_S + 1, "the design peak is",
         "1,000,000,000"),
        ("retention_days", ps.MAX_RETENTION_DAYS + 1, "retention is",
         "36,500"),
        ("dims", ps.MAX_VECTOR_DIMS + 1, "vector dimensions is",
         "1,000,000"),
        ("bytes_per_value", ps.MAX_BYTES_PER_VALUE + 1,
         "bytes per number is", "64"),
        ("node_gb", ps.MAX_NODE_GB + 1, "RAM per vector-store machine is",
         "1,000,000"),
    )

    def refusal(self, field, value):
        if field == "ops_per_s":
            args, kwargs = (value,), {}
        else:
            args, kwargs = (100.0,), {field: value}
        with self.assertRaises(ps.SizingError) as caught:
            self.size(*args, **kwargs)
        return str(caught.exception)

    def test_every_bounded_input_is_refused_by_name(self):
        for field, value, opening, _limit in self.JUST_OVER_EACH_BOUND:
            with self.subTest(field=field):
                message = self.refusal(field, value)
                self.assertTrue(message.startswith(opening),
                                f"{message!r} must name the field it is about")
                self.assertIn("larger than this calculator will size", message)

    def test_the_refused_value_reads_as_larger_than_the_limit(self):
        """The message used to print the value with :g, which rounded it to
        look exactly like the limit it was being compared against: "the design
        peak is 1e+09 ... the most it accepts is 1,000,000,000"."""
        for field, value, _opening, limit in self.JUST_OVER_EACH_BOUND:
            with self.subTest(field=field):
                message = self.refusal(field, value)
                before, _, after = message.partition(
                    "larger than this calculator will size")
                self.assertIn(limit, after, "the limit is named")
                self.assertNotIn(limit, before,
                                 f"the rejected value reads as {limit}, which "
                                 "is the limit it is said to exceed")
                self.assertIn(ps.as_given(value), before,
                              "the rejected value is quoted as it was given")

    def test_each_bound_is_itself_still_accepted(self):
        for field, value in (("retention_days", ps.MAX_RETENTION_DAYS),
                             ("dims", ps.MAX_VECTOR_DIMS),
                             ("bytes_per_value", ps.MAX_BYTES_PER_VALUE),
                             ("node_gb", ps.MAX_NODE_GB)):
            with self.subTest(field=field):
                r = self.size(100.0, **{field: value})
                self.assertTrue(
                    math.isfinite(r["storage"]["hot_vector_ram_bytes"]))

    def test_the_bounds_move_no_machine_count(self):
        """Every tier is sized exactly as it was before the bounds existed."""
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                r = self.size(ps.TIER_OPS_PER_S[name])
                self.assertEqual(r["machines"]["api_servers"],
                                 want["api_servers"])


class TestTheReportNamesWhatForcedTheNodeSize(ModelBaseTest):
    """The reader of a web page never typed a command-line flag."""

    @staticmethod
    def qdrant_note(r):
        for section in ps.report_sections(r):
            if section["title"] == "Qdrant node choice":
                return section["note"]
        raise AssertionError("the report lost its Qdrant node choice section")

    def test_the_command_line_report_names_the_flag(self):
        note = self.qdrant_note(self.size(100.0, node_gb=512))
        self.assertIn("The size was forced with --node-gb.", note)

    def test_the_web_report_names_the_box(self):
        r = self.size(100.0, node_gb=512,
                      node_gb_source=ps.NODE_GB_SOURCE_WEB)
        note = self.qdrant_note(r)
        self.assertIn("The size was forced with the RAM per vector-store "
                      "machine box.", note)
        self.assertNotIn("--node-gb", ps.render_report(r, "FORCED"),
                         "a web reader never typed a command-line flag")

    def test_the_automatic_choice_names_neither(self):
        note = self.qdrant_note(self.size(100.0))
        self.assertIn("buys the least total RAM", note)
        self.assertNotIn("The size was forced", note)

    def test_a_forced_size_is_not_labelled_an_automatic_choice(self):
        rows = self.node_row(self.size(100.0, node_gb=512))
        self.assertEqual(rows[1], "512 GB, forced")
        self.assertEqual(rows[2], "assumption - set by hand")

    def test_an_automatic_size_keeps_the_label_that_describes_it(self):
        rows = self.node_row(self.size(100.0))
        self.assertEqual(rows[1], "chosen automatically")
        self.assertIn("automatic choice buys the least total RAM", rows[2])

    @staticmethod
    def node_row(r):
        for section in ps.report_sections(r):
            if section["title"] != "Inputs":
                continue
            for row in section["rows"]:
                if row[0] == "RAM per vector-store machine":
                    return row
        raise AssertionError("the Inputs table lost its node-size row")


class TestReadingOneFormBox(unittest.TestCase):
    """read_number names the box, so a reader is not left to hunt."""

    def test_a_blank_box_is_named(self):
        with self.assertRaises(ps.FieldError) as caught:
            ps.read_number("", "ops")
        self.assertEqual(caught.exception.field, "ops")
        self.assertIn("design peak box is empty", str(caught.exception))

    def test_text_is_quoted_back_with_the_box_that_holds_it(self):
        with self.assertRaises(ps.FieldError) as caught:
            ps.read_number("lots", "dims")
        self.assertEqual(caught.exception.field, "dims")
        self.assertIn('vector dimensions box says "lots"',
                      str(caught.exception))

    def test_a_thousands_comma_is_called_out_on_its_own(self):
        with self.assertRaises(ps.FieldError) as caught:
            ps.read_number("1,000", "ops")
        self.assertIn("comma", str(caught.exception))

    def test_a_good_number_is_read(self):
        self.assertEqual(ps.read_number(" 512 ", "node_gb"), 512.0)

    def test_a_field_error_is_still_an_ordinary_bad_input(self):
        self.assertIsInstance(ps.FieldError("ops", "x"), ps.SizingError)
        self.assertIsInstance(ps.FieldError("ops", "x"), ValueError)

    def test_every_box_on_the_form_has_a_name_and_a_what_to_type(self):
        for key, _, _ in ps.FORM_FIELDS:
            with self.subTest(box=key):
                self.assertIn(key, ps.FORM_FIELD_HELP)
                name, what = ps.FORM_FIELD_HELP[key]
                self.assertTrue(name and what)
        self.assertEqual(set(ps.FORM_DEFAULTS),
                         {key for key, _, _ in ps.FORM_FIELDS})

    def test_a_whole_number_default_reads_as_a_whole_number(self):
        self.assertEqual(ps.format_default(100.0), "100")
        self.assertEqual(ps.format_default(45.0), "45")
        self.assertEqual(ps.format_default(90), "90")
        self.assertEqual(ps.format_default(None), "")

    def test_a_box_that_was_never_sent_is_told_apart_from_an_empty_one(self):
        self.assertIsNone(ps.submitted_text({}, "ops"))
        self.assertEqual(ps.submitted_text({"ops": [""]}, "ops"), "")
        self.assertEqual(ps.submitted_text({"ops": ["100"]}, "ops"), "100")


class TestResultFromQuery(unittest.TestCase):
    """The form's answer, without going through a socket."""

    FULL: ClassVar[dict] = {
        "ops": ["100"], "add": ["45"], "plain": ["45"], "agent": ["10"],
        "retention_days": ["90"], "dims": ["1024"],
        "bytes_per_value": ["1"], "node_gb": [""]}

    def test_a_full_submission_is_sized(self):
        values, result, error, bad = ps.result_from_query(dict(self.FULL))
        self.assertIsNone(error)
        self.assertIsNone(bad)
        self.assertEqual(result["machines"]["api_servers"], 3)
        self.assertEqual(values["ops"], "100")

    def test_a_blank_box_is_an_error_that_names_itself(self):
        query = dict(self.FULL, ops=[""])
        values, result, error, bad = ps.result_from_query(query)
        self.assertIsNone(result)
        self.assertEqual(bad, "ops")
        self.assertIn("design peak box is empty", error)
        self.assertEqual(values["ops"], "",
                         "the box must stay as the reader left it")

    def test_an_absent_box_takes_the_default(self):
        values, result, error, _bad = ps.result_from_query({"ops": ["100"]})
        self.assertIsNone(error)
        self.assertEqual(values["dims"], "1024")
        self.assertEqual(result["inputs"]["dims"], ps.DEFAULT_VECTOR_DIMS)

    def test_no_box_at_all_gives_the_target_tier(self):
        values, result, error, _ = ps.result_from_query({})
        self.assertIsNone(error)
        self.assertEqual(values["ops"], "100")
        self.assertEqual(result["inputs"]["ops_per_s"],
                         ps.TIER_OPS_PER_S["target"])

    def test_a_blank_node_size_box_is_the_automatic_choice(self):
        _, result, error, _ = ps.result_from_query(dict(self.FULL))
        self.assertIsNone(error)
        self.assertFalse(result["machines"]["qdrant_node_ram_gb_forced"])

    def test_a_fault_that_belongs_to_no_single_box_blames_none(self):
        query = dict(self.FULL, add=["50"], plain=["50"], agent=["10"])
        _, result, error, bad = ps.result_from_query(query)
        self.assertIsNone(result)
        self.assertIsNone(bad, "the mix is three boxes together, not one")
        self.assertIn("must add up to 100", error)

    def test_the_web_result_says_the_box_forced_the_node_size(self):
        query = dict(self.FULL, node_gb=["512"])
        _, result, _, _ = ps.result_from_query(query)
        self.assertEqual(result["inputs"]["node_gb_source"],
                         ps.NODE_GB_SOURCE_WEB)

    def test_the_first_box_at_fault_is_the_one_reported(self):
        query = dict(self.FULL, ops=[""], dims=[""])
        _, _, error, bad = ps.result_from_query(query)
        self.assertEqual(bad, "ops")
        self.assertIn("design peak", error)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# =============================================================================
# 22. What `validate` exports, and the README's claim about it
# =============================================================================


class TestTheExportedNumbers(unittest.TestCase):
    """`validate` promises a fixed, named list of keys. These tests pin it."""

    # Every key one tier writes, apart from the sensitivity rows, which are
    # checked separately because how many there are depends on the mix.
    PINNED_TIER_KEYS: ClassVar[list] = [
        "adds_per_s", "agent_gpu_cards", "agent_gpu_spare",
        "agent_llm_calls_per_s_high", "agent_llm_calls_per_s_low",
        "agent_llm_calls_per_s_planning", "agent_searches_per_s",
        "api_server_spec", "api_servers", "automated_client_sessions",
        "automated_client_sessions_sustained",
        "api_usable_searches_per_server", "api_work_per_s",
        "bytes_per_value", "embed_gpu_cards_high", "embed_gpu_cards_low",
        "embed_gpu_spare", "embeds_per_s", "embeds_per_s_with_types_fix",
        "episodes_per_day", "episodes_per_year", "episodes_retained",
        "hot_vector_ram_gb", "human_sessions_heavy", "human_sessions_high",
        "human_sessions_low",
        "mix_add", "mix_agent", "mix_plain", "network_busiest_link_mbps",
        "network_east_west_mbps", "network_embed_bytes_per_call",
        "network_headroom_on_10gbe", "network_llm_bytes_per_call",
        "network_north_south_mbps", "network_vector_search_bytes_per_call",
        "network_vector_write_bytes_per_call", "node_gb_forced", "ops_per_s",
        "plain_searches_per_s", "postgres_core_connections",
        "postgres_exceeds_chart_default", "postgres_exceeds_proven_setting",
        "postgres_gateway_connections", "postgres_gb_high", "postgres_gb_low",
        "postgres_max_connections_required",
        "postgres_needs_connection_pooler", "postgres_servers",
        "postgres_statements_per_s", "postgres_total_connections",
        "qdrant_fill_of_allowance_pct", "qdrant_node_ram_gb",
        "qdrant_nodes_at_256gb", "qdrant_nodes_at_512gb",
        "qdrant_nodes_at_768gb", "qdrant_nvme_gb", "qdrant_servers",
        "qdrant_tight_fit", "qdrant_total_ram_gb", "qdrant_usable_gb_per_node",
        "retention_days", "total_cpu_servers",
        "unbounded_year_hot_vector_ram_gb", "unbounded_year_postgres_gb_high",
        "unbounded_year_postgres_gb_low", "unbounded_year_qdrant_nvme_gb",
        "vector_dims", "vector_searches_per_s", "vector_writes_per_s",
    ]

    SENSITIVITY_ROW_FIELDS: ClassVar[list] = [
        "agent_searches_per_s", "api_servers", "api_work_per_s",
        "llm_calls_per_s_high", "llm_calls_per_s_low", "total_ops_per_s",
        "vector_searches_per_s",
    ]

    def exported(self, name="target", **kwargs):
        r = ps.size_deployment(ps.TIER_OPS_PER_S[name], run_name=name, **kwargs)
        return dict(ps.published_numbers(name, r)), r

    def test_the_exported_key_list_is_pinned_so_none_can_quietly_disappear(self):
        """The README promises a fixed, named list. A key that is dropped or
        renamed must fail here rather than vanish from the published file."""
        pairs, r = self.exported()
        got = sorted(k.split(".", 1)[1] for k in pairs
                     if not k.startswith("target.sensitivity."))
        self.assertEqual(got, sorted(self.PINNED_TIER_KEYS))
        rows = len(r["sensitivity"])
        for position in range(1, rows + 1):
            with self.subTest(row=position):
                fields = sorted(
                    k.rsplit(".", 1)[1] for k in pairs
                    if k.startswith(f"target.sensitivity.row{position}."))
                self.assertEqual(fields, sorted(self.SENSITIVITY_ROW_FIELDS))
        self.assertNotIn(f"target.sensitivity.row{rows + 1}.api_servers", pairs)

    def test_every_constant_the_readme_names_is_exported(self):
        """The README's input table calls itself the whole model, and the
        `validate` section promises every constant in it. This test reads the
        README, so the two cannot drift apart."""
        named = set()
        with open(os.path.join(HERE, "README.md"), encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("|"):
                    continue
                for quoted in re.findall(r"`([^`]+)`", line):
                    if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", quoted):
                        named.add(quoted)
        self.assertGreater(len(named), 60,
                           "the README input table must still be readable as a "
                           "table of named constants")
        keys = {key for key, _ in ps.constant_numbers()}
        for constant in sorted(named):
            with self.subTest(constant=constant):
                stem = "constants." + constant.lower()
                self.assertTrue(
                    stem in keys or any(k.startswith(stem + "_")
                                        for k in keys),
                    f"{constant} is in the README's input table but "
                    f"`validate` never writes it")

    def test_every_exported_constant_is_a_real_constant_of_the_model(self):
        """The other direction: no key naming something the program does not
        actually define."""
        for key, value in ps.constant_numbers():
            with self.subTest(key=key):
                self.assertTrue(key.startswith("constants."))
                self.assertIsNotNone(value)

    def test_the_per_call_network_sizes_the_report_prints_are_exported(self):
        """The Network table prints these four and the busiest link, so the
        file has to be able to back the report."""
        pairs, r = self.exported()
        n = r["network"]
        self.assertEqual(pairs["target.network_embed_bytes_per_call"],
                         n["embed_bytes_per_call"])
        self.assertEqual(pairs["target.network_vector_search_bytes_per_call"],
                         n["vector_search_bytes_per_call"])
        self.assertEqual(pairs["target.network_vector_write_bytes_per_call"],
                         n["vector_write_bytes_per_call"])
        self.assertEqual(pairs["target.network_llm_bytes_per_call"],
                         n["llm_bytes_per_call"])
        self.assertEqual(pairs["target.network_busiest_link_mbps"],
                         n["busiest_link_mbps"])

    def test_the_node_size_input_is_exported_under_its_own_name(self):
        """Two runs that ordered the same machines for different reasons must
        not look identical in the file."""
        automatic, _ = self.exported()
        forced, _ = self.exported(node_gb=768)
        self.assertFalse(automatic["target.node_gb_forced"])
        self.assertTrue(forced["target.node_gb_forced"])
        self.assertEqual(forced["target.qdrant_node_ram_gb"], 768)


class TestFractionalSensitivityRates(unittest.TestCase):
    """A mix can put a fractional agent rate in the sensitivity table. Keys
    built by truncating the rate put 2.0 and 2.5 in the same place and the
    second silently overwrote the first."""

    # 47.5 adds, 50 plain, 2.5 agent-mode per 100 operations, at 100 ops/s.
    # The tier's own agent rate is 2.5/s, so the table shows 0, 2, 2.5, 10, 25.
    MIX = ps.TrafficMix(47.5, 50.0, 2.5)
    EXPECTED_RATES: ClassVar[list] = [0.0, 2.0, 2.5, 10.0, 25.0]

    def result(self):
        return ps.size_deployment(100.0, self.MIX, run_name="target")

    def test_a_fractional_mix_exports_every_sensitivity_row(self):
        r = self.result()
        self.assertEqual([row["agent_searches_per_s"] for row in r["sensitivity"]],
                         self.EXPECTED_RATES)
        pairs = dict(ps.published_numbers("target", r))
        exported = [pairs[f"target.sensitivity.row{i}.agent_searches_per_s"]
                    for i in range(1, len(self.EXPECTED_RATES) + 1)]
        self.assertEqual(exported, self.EXPECTED_RATES,
                         "all five rows must survive the export")
        # The 2.0 row and the 2.5 row are different sizings, and the file has
        # to keep them apart: 89.0 vector searches against 100.0.
        self.assertEqual(pairs["target.sensitivity.row2.vector_searches_per_s"],
                         r["sensitivity"][1]["vector_searches_per_s"])
        self.assertEqual(pairs["target.sensitivity.row3.vector_searches_per_s"],
                         r["sensitivity"][2]["vector_searches_per_s"])
        self.assertNotEqual(
            pairs["target.sensitivity.row2.vector_searches_per_s"],
            pairs["target.sensitivity.row3.vector_searches_per_s"])

    def test_the_sensitivity_table_prints_a_fractional_rate_as_itself(self):
        """Rounding the rate to a whole number printed two rows both labelled
        "2", which is two different answers under one heading."""
        printed = sensitivity_rate_cells(self.result())
        self.assertEqual(printed, ["0.0", "2.0", "2.5  <- this run",
                                   "10.0", "25.0"])
        self.assertEqual(len(set(printed)), len(printed),
                         "no two rows may print the same rate")

    def test_validate_writes_every_row_of_a_fractional_mix_to_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "numbers.json")
            proc = run_cli("validate", "--add", "47.5", "--plain", "50",
                           "--agent", "2.5", "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as handle:
                written = json.load(handle)
        rates = [written[f"target.sensitivity.row{i}.agent_searches_per_s"]
                 for i in range(1, 6)]
        self.assertEqual(rates, self.EXPECTED_RATES)
        # The pilot tier's own agent rate is 0.5/s. Truncating it printed the
        # key pilot.sensitivity.agent0 twice, with two different values.
        keys = [k for k in written if k.startswith("pilot.sensitivity.")]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(written["pilot.sensitivity.row1.agent_searches_per_s"],
                         0.0)
        self.assertEqual(written["pilot.sensitivity.row2.agent_searches_per_s"],
                         0.5)


# =============================================================================
# 23. A web address the calculator cannot honour
# =============================================================================


class TestUnknownQueryParameters(unittest.TestCase):
    """A misspelled setting used to be ignored in silence, so the endpoint
    answered a question the reader had not asked."""

    def test_a_misspelled_setting_is_refused_and_a_correction_offered(self):
        query = {"ops": ["100"], "retention_day": ["1"]}
        values, result, error, bad = ps.result_from_query(query)
        self.assertIsNone(result)
        self.assertIn("retention_day", error)
        self.assertIn("retention_days", error, "it must suggest the spelling")
        self.assertIsNone(bad, "no box on the form is at fault")
        self.assertEqual(values["ops"], "100",
                         "the boxes keep what was sent, so the page can be "
                         "redrawn with them")

    def test_a_setting_in_the_wrong_case_is_named_against_its_spelling(self):
        """The commonest miss is the right word in the wrong case, and it used
        to get no suggestion at all: "Ops" scores 0.67 against "ops" and "OPS"
        scores 0, both below the cutoff, so both got the full list instead."""
        for typed in ("OPS", "Ops", "Retention_Days", "NODE_GB"):
            with self.subTest(typed=typed):
                _, result, error, _ = ps.result_from_query({typed: ["100"]})
                self.assertIsNone(result)
                self.assertIn(typed, error)
                self.assertIn(f'Did you mean "{typed.lower()}"?', error)

    def test_a_setting_with_no_near_spelling_lists_the_ones_that_work(self):
        _, result, error, _ = ps.result_from_query({"nonsense": ["zzz"]})
        self.assertIsNone(result)
        self.assertIn("nonsense", error)
        for known in ("ops", "retention_days", "node_gb"):
            self.assertIn(known, error)

    def test_a_setting_given_twice_is_refused_rather_than_first_wins(self):
        _, result, error, bad = ps.result_from_query(
            {"ops": ["100", "999999"]})
        self.assertIsNone(result)
        self.assertIn("ops", error)
        self.assertIn("2 times", error)
        self.assertIsNone(bad)

    def test_a_web_address_the_calculator_knows_is_still_accepted(self):
        _, result, error, _ = ps.result_from_query(
            {"ops": ["100"], "node_gb": ["512"]})
        self.assertIsNone(error)
        self.assertEqual(result["machines"]["qdrant_node_ram_gb"], 512)


# =============================================================================
# 24. The web server is a local development server
# =============================================================================


class TestTheServerDoesNotHoldStalledConnections(unittest.TestCase):
    """A client that connects and sends partial headers used to hold one
    thread and one file descriptor for as long as it liked."""

    def test_the_shipped_handler_sets_a_read_timeout(self):
        self.assertIsInstance(ps.SERVER_REQUEST_TIMEOUT_S, (int, float))
        self.assertGreater(ps.SERVER_REQUEST_TIMEOUT_S, 0)
        self.assertLessEqual(ps.SERVER_REQUEST_TIMEOUT_S, 60,
                             "long enough to be useless is the same as none")
        self.assertEqual(ps.SizingHandler.timeout, ps.SERVER_REQUEST_TIMEOUT_S,
                         "socketserver applies this attribute to the socket")

    def test_a_client_that_sends_partial_headers_is_dropped(self):
        """Run the real handler with a short timeout so the test is quick; the
        mechanism under test is the class attribute, not the number."""

        class QuickHandler(ps.SizingHandler):
            timeout = 0.5

            def log_message(self, fmt, *args):
                pass

        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuickHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            sock = socket.create_connection(httpd.server_address, timeout=10)
            with sock:
                sock.sendall(b"GET /healthz HTTP/1.1\r\nHost: x\r\n")
                sock.settimeout(10)
                started = time.monotonic()
                # The server drops the connection, so the read ends in end-of
                # -file rather than blocking until the test's own timeout.
                data = sock.recv(4096)
                waited = time.monotonic() - started
            self.assertEqual(data, b"",
                             "a half-sent request must be dropped, not served")
            self.assertLess(waited, 8.0,
                            "the connection has to be dropped on the handler's "
                            "own timeout, not held open")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)


class TestTheServerStaysQuietWhenAClientHangsUp(unittest.TestCase):
    """A reader closing the tab is ordinary. It used to be logged as an
    internal error, and the 500 was then written to the same dead socket."""

    def handler_whose_work_raises(self, exc):
        handler = ps.SizingHandler.__new__(ps.SizingHandler)
        handler.path = "/api/calc?ops=100"
        sent = []

        def raiser():
            raise exc

        # Reaching into the handler is the point: do_GET's job is to decide
        # what happens when the work it delegates to raises.
        handler._handle_get = raiser                      # noqa: SLF001
        handler._send = lambda *args: sent.append(args)   # noqa: SLF001
        return handler, sent

    def run_do_get(self, exc):
        handler, sent = self.handler_whose_work_raises(exc)
        noise = io.StringIO()
        real_stderr, sys.stderr = sys.stderr, noise
        try:
            handler.do_GET()
        finally:
            sys.stderr = real_stderr
        return sent, noise.getvalue()

    def test_a_broken_pipe_is_not_reported_as_an_internal_error(self):
        for exc in (BrokenPipeError(32, "Broken pipe"),
                    ConnectionResetError(104, "Connection reset by peer")):
            with self.subTest(exc=type(exc).__name__):
                sent, noise = self.run_do_get(exc)
                self.assertEqual(noise, "")
                self.assertEqual(sent, [],
                                 "there is no socket left to answer on")

    def test_a_real_fault_is_still_reported_and_answered_with_a_500(self):
        sent, noise = self.run_do_get(RuntimeError("the model exploded"))
        self.assertIn("the model exploded", noise)
        self.assertEqual([args[0] for args in sent], [500])


class TestTheServeHelpSaysItIsForLocalUse(unittest.TestCase):

    def test_serve_help_warns_that_this_is_a_development_server(self):
        proc = run_cli("serve", "--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        text = proc.stdout.lower()
        self.assertIn("local development server", text)
        self.assertIn("authentication", text)

    def test_the_readme_warns_that_this_is_a_development_server(self):
        with open(os.path.join(HERE, "README.md"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("local development server, not a service", text)
        self.assertIn("SERVER_REQUEST_TIMEOUT_S", text)


# =============================================================================
# 25. The README's own examples, worked out by hand
# =============================================================================


class TestTheReadmeExamples(unittest.TestCase):

    def test_the_readme_calc_example_sizes_its_own_mix(self):
        """README: `calc --ops 250 --agent 4 --plain 51`. By hand, at 250
        ops/s with 45 adds / 51 plain / 4 agent-mode per 100 operations:
          adds           = 250 x 0.45 = 112.5/s
          plain searches = 250 x 0.51 = 127.5/s
          agent searches = 250 x 0.04 =  10.0/s
          vector searches = 127.5 x 1 + 10 x 22 = 347.5/s
          API work        = 347.5 + 112.5 = 460.0/s
          API servers     = ceil(460 / 108) = 5
        """
        proc = run_cli("calc", "--ops", "250", "--agent", "4", "--plain", "51",
                       "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        got = json.loads(proc.stdout)
        self.assertEqual(got["demand"]["adds_per_s"], 112.5)
        self.assertEqual(got["demand"]["plain_searches_per_s"], 127.5)
        self.assertEqual(got["demand"]["agent_searches_per_s"], 10.0)
        self.assertEqual(got["demand"]["vector_searches_per_s"], 347.5)
        self.assertEqual(got["machines"]["api_work_per_s"], 460.0)
        self.assertEqual(got["machines"]["api_servers"], 5)

    def test_the_readme_calc_example_renders_as_a_report(self):
        proc = run_cli("calc", "--ops", "250", "--agent", "4", "--plain", "51")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("5 API server(s)", proc.stdout)
        self.assertIn("45 adds, 51 plain searches, 4 agent-mode searches",
                      proc.stdout)


class TestTheUsersReportCarriesLabels(unittest.TestCase):
    """The `users` table was the third table with no label column, and its
    provenance lived only in the prose above it."""

    def test_the_users_table_has_a_label_column_on_every_row(self):
        proc = run_cli("users", "--humans", "5000", "--automated", "40")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.splitlines()
        header = next(line for line in lines if "Kind of caller" in line)
        self.assertTrue(header.rstrip().endswith("Label"), header)
        for kind in ("Concurrent human chat sessions",
                     "Concurrent automated client sessions", "Total"):
            row = next(line for line in lines
                       if line.strip().startswith(kind))
            with self.subTest(kind=kind):
                self.assertTrue(
                    any(word in row for word in ("estimate", "derived")),
                    f"{kind} row carries no label: {row!r}")


class TestVectorDimensionsAreQuotedFaithfully(unittest.TestCase):
    """The refusal used to format the value with :g, so a dimension count of
    1024.0000000001 was reported as "vector dimensions is 1024, but it must be
    a whole number" - naming the value as the very thing it required."""

    def test_a_dimension_just_off_a_whole_number_is_quoted_as_typed(self):
        with self.assertRaises(ps.SizingError) as caught:
            ps.size_deployment(100.0, dims=1024.0000000001)
        message = str(caught.exception)
        self.assertIn("1024.0000000001", message)
        self.assertIn("whole number", message)


class TestTheWebPageDoesNotOverclaimItsLabels(unittest.TestCase):
    """The page used to say "Every number is labelled", which two of its own
    tables contradict."""

    def page(self, query=None):
        _, result, error, bad = ps.result_from_query(query or {})
        self.assertIsNone(error)
        return ps.render_html({}, result, None, bad)

    def test_the_page_names_the_two_tables_that_carry_no_label(self):
        html = self.page()
        self.assertIn("Qdrant node choice", html)
        self.assertIn("what-ifs rather than findings", html)
        self.assertNotIn("Every number is labelled", html)

    def test_the_page_says_the_json_carries_no_labels(self):
        html = self.page()
        self.assertIn("without labels", html)
        self.assertIn("run_name", html)


# =============================================================================
# 26. One name per input, and inputs echoed exactly as they were given
# =============================================================================


class TestEveryMessageCallsAnInputByTheSameName(ModelBaseTest):
    """One box, one name.

    The bound check and the older zero check used to call the same box two
    different things: "bytes per number is 65 bytes" when it was too large and
    "bytes per value is 0" when it was zero, next to a form that labels it
    "Bytes per number".
    """

    # The name each input answers to, and a too-small and a too-large value
    # for it. Each name is the label on the web form, less the units.
    INPUTS: ClassVar[tuple] = (
        ("the design peak", "ops_per_s", 0, ps.MAX_OPS_PER_S + 1),
        ("retention", "retention_days", -1, ps.MAX_RETENTION_DAYS + 1),
        ("vector dimensions", "dims", -8, ps.MAX_VECTOR_DIMS + 1),
        ("bytes per number", "bytes_per_value", 0, ps.MAX_BYTES_PER_VALUE + 1),
        ("RAM per vector-store machine", "node_gb", 0, ps.MAX_NODE_GB + 1),
    )

    def refusal(self, field, value):
        if field == "ops_per_s":
            args, kwargs = (value,), {}
        else:
            args, kwargs = (100.0,), {field: value}
        with self.assertRaises(ps.SizingError) as caught:
            self.size(*args, **kwargs)
        return str(caught.exception)

    def test_both_refusals_of_one_input_open_with_the_same_name(self):
        for name, field, too_small, too_large in self.INPUTS:
            for value in (too_small, too_large):
                with self.subTest(field=field, value=value):
                    message = self.refusal(field, value)
                    self.assertTrue(
                        message.startswith(name + " is"),
                        f"{message!r} should open with {name!r}")

    def test_every_name_is_the_label_on_the_form(self):
        """A message that names a box the page does not have helps nobody."""
        labels = {key: title for key, title, _hint in ps.FORM_FIELDS}
        for name, field, _small, _large in self.INPUTS:
            key = "ops" if field == "ops_per_s" else field
            with self.subTest(field=field):
                label = labels[key].split(",")[0].lower()
                self.assertIn(label, name.lower())


class TestTheReportEchoesItsInputsExactly(ModelBaseTest):
    """A reader has to be able to reproduce the answer from the inputs printed
    above it. The Inputs table used to round them to whole numbers, so half a
    byte per number read as "0" above a table sized for half a byte."""

    def inputs_table(self, r):
        return {row[0]: row[1]
                for row in report_section(r, "Inputs")["rows"]}

    def test_a_fractional_input_is_printed_as_itself(self):
        r = self.size(100.0, retention_days=0.5, bytes_per_value=0.5)
        rows = self.inputs_table(r)
        self.assertEqual(rows["Retention"], "0.5 days")
        self.assertEqual(rows["Bytes stored per number"], "0.5")
        self.assertIn("0.5 days of retention",
                      report_section(r, "Storage")["note"])
        self.assertIn("0.5 byte(s) per number",
                      report_section(r, "Storage")["note"])

    def test_a_fractional_traffic_mix_is_printed_as_itself(self):
        r = self.size(100.0, mix=ps.TrafficMix(45.4, 44.6, 10.0))
        row = self.inputs_table(r)["Traffic mix per 100 operations"]
        self.assertEqual(row, "45.4 adds, 44.6 plain searches, "
                              "10 agent-mode searches")
        # The demand table one section below is worked out from these.
        self.assertClose(r["demand"]["adds_per_s"], 45.4, "adds/s")

    def test_a_whole_number_keeps_its_thousands_separator(self):
        rows = self.inputs_table(self.size(1000.0))
        self.assertEqual(rows["Design peak"], "1,000 operations/s")
        self.assertEqual(rows["Vector dimensions"], "1,024")

    def test_a_design_peak_below_half_an_operation_still_prints(self):
        """0.0000001 ops/s used to head the whole report as "0.0
        operations/s", a rate the program refuses when it is typed."""
        rows = self.inputs_table(self.size(1e-7))
        self.assertEqual(rows["Design peak"], "1e-07 operations/s")

    def test_the_command_line_titles_the_report_with_the_rate_it_sized(self):
        done = run_cli("calc", "--ops", "0.5")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("DESIGN PEAK 0.5 operations/s", done.stdout)


class TestAsGiven(unittest.TestCase):
    """The one formatter both the report and the refusals use."""

    def test_whole_numbers_lose_their_point_zero(self):
        self.assertEqual(ps.as_given(100.0), "100")
        self.assertEqual(ps.as_given(1000.0), "1,000")
        self.assertEqual(ps.as_given(64), "64")

    def test_a_fraction_keeps_every_digit(self):
        self.assertEqual(ps.as_given(0.5), "0.5")
        self.assertEqual(ps.as_given(1024.0000000001), "1024.0000000001")

    def test_a_number_too_large_to_count_exactly_stays_short(self):
        """1e307 as 309 digits is not what anybody typed either."""
        self.assertEqual(ps.as_given(1e307), "1e+307")

    def test_it_never_rounds_one_number_into_another(self):
        self.assertNotEqual(ps.as_given(ps.MAX_OPS_PER_S + 1),
                            ps.as_given(ps.MAX_OPS_PER_S))


# =============================================================================
# 27. The report says only what the model knows
# =============================================================================


class TestHeadroomIsMeasuredOnTheBusiestLink(ModelBaseTest):
    """The row is labelled "busiest direction". It used to be computed from
    the east-west figure whatever the north-south figure came to, and the real
    maximum was computed and then never used."""

    def test_headroom_matches_the_busiest_of_the_two_directions(self):
        for ops in (20.0, 100.0, 1000.0):
            with self.subTest(ops=ops):
                n = self.size(ops)["network"]
                self.assertClose(n["headroom_on_10gbe"],
                                 10000.0 / n["busiest_link_mbps"], "headroom")

    def test_headroom_would_follow_a_busier_north_south_link(self):
        """The label has to stay true if the byte sizes ever change."""
        original = ps.NS_ADD_RESPONSE_BYTES
        ps.NS_ADD_RESPONSE_BYTES = original * 10_000
        try:
            n = self.size(100.0)["network"]
            self.assertGreater(n["north_south_mbps"], n["east_west_mbps"])
            self.assertClose(n["headroom_on_10gbe"],
                             10000.0 / n["north_south_mbps"], "headroom")
        finally:
            ps.NS_ADD_RESPONSE_BYTES = original


class TestTheMachinesTableStatesNoSpecItDoesNotHave(ModelBaseTest):
    """API_SERVER_VCPU and API_SERVER_RAM_GB describe the machine the API
    benchmark ran on. The PostgreSQL and Qdrant rows used to borrow them, so
    the report stated a spec for two machines nobody has sized."""

    def spec(self, machine):
        rows = report_section(self.size(100.0), "Machines")["rows"]
        return next(row[2] for row in rows if row[0].startswith(machine))

    def test_the_api_row_keeps_its_measured_machine_class(self):
        spec = self.spec("API server")
        self.assertIn(f"{ps.API_SERVER_VCPU} vCPU", spec)
        self.assertIn(f"{ps.API_SERVER_RAM_GB} GB", spec)

    def test_the_other_two_rows_claim_no_vcpu(self):
        for machine in ("PostgreSQL server", "Qdrant server"):
            with self.subTest(machine=machine):
                spec = self.spec(machine)
                self.assertNotIn("vCPU,", spec)
                self.assertIn("undecided", spec)

    def test_the_qdrant_row_keeps_the_ram_the_model_did_choose(self):
        r = self.size(100.0)
        rows = report_section(r, "Machines")["rows"]
        spec = next(row[2] for row in rows if row[0].startswith("Qdrant"))
        self.assertIn(f"{r['machines']['qdrant_node_ram_gb']} GB RAM", spec)


class TestTheSensitivityTableSaysWhichRowIsThisRun(ModelBaseTest):
    """One decimal place is not enough to keep every pair of rates apart: a
    mix whose own agent rate is 2.04 sits beside the fixed rate 2.0 and both
    print as "2.0". The table used to leave the reader to guess which of the
    two rows was the traffic mix they had asked about."""

    def test_a_rate_that_collides_with_a_fixed_rate_is_still_marked(self):
        r = self.size(20.4)
        self.assertClose(r["demand"]["agent_searches_per_s"], 2.04, "agents/s")
        cells = sensitivity_rate_cells(r)
        self.assertEqual(cells, ["0.0", "2.0", "2.0  <- this run",
                                 "10.0", "25.0"])

    def test_exactly_one_row_is_marked_at_every_tier(self):
        for ops in (20.0, 100.0, 1000.0, 20.4):
            with self.subTest(ops=ops):
                r = self.size(ops)
                marked = [row for row in r["sensitivity"] if row["is_this_run"]]
                self.assertEqual(len(marked), 1)
                self.assertEqual(marked[0]["api_servers"],
                                 r["machines"]["api_servers"])
                self.assertEqual(
                    sum("<- this run" in cell
                        for cell in sensitivity_rate_cells(r)), 1)

    def test_the_note_points_at_the_mark_rather_than_at_nothing(self):
        note = report_section(self.size(20.4), "Sensitivity")["note"]
        self.assertIn("this run", note)


class TestAPopulationOfNobodyGetsNoTier(unittest.TestCase):
    """`users --humans 0 --automated 0` used to answer "Plan for the high rate:
    pilot" - a hardware recommendation for a population that makes no
    requests. The sizing direction refuses a design peak of zero by name."""

    def test_the_report_names_no_tier_for_an_empty_population(self):
        text = ps.render_users_report(ps.ops_for_population(0, 0))
        self.assertIn("makes no requests", text)
        self.assertNotIn("Plan for the high rate", text)
        self.assertNotIn("fits the pilot tier", text)

    def test_one_user_is_still_a_population_to_size(self):
        text = ps.render_users_report(ps.ops_for_population(1, 0))
        self.assertIn("Plan for the high rate", text)

    def test_the_command_still_prints_the_demand_it_worked_out(self):
        done = run_cli("users", "--humans", "0", "--automated", "0")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("0.00 ops/s", done.stdout)
        self.assertIn("nothing to size", done.stdout)


# =============================================================================
# 28. Two things that sound alike and are not: agent-mode search is a property
#     of a REQUEST, an automated client is a property of a CALLER
# =============================================================================


class TestTheOldAgentsFlagIsRefusedByName(unittest.TestCase):
    """`--agents` meant the count of callers that are programs. It was one
    letter from `--agent`, the agent-mode share of the traffic mix, and the
    two mean completely different things. Accepting it quietly, or letting
    argparse read it as the mix share, would size the wrong deployment."""

    def test_the_old_flag_is_refused_and_names_the_new_one(self):
        proc = run_cli("users", "--humans", "5000", "--agents", "40")
        self.assertEqual(proc.returncode, 2, proc.stdout)
        message = proc.stderr + proc.stdout
        self.assertIn("--agents is no longer a flag", message)
        self.assertIn("--automated", message)
        self.assertNotIn("Traceback", message)

    def test_the_refusal_explains_the_difference_in_one_sentence(self):
        proc = run_cli("users", "--humans", "5000", "--agents", "40")
        message = proc.stderr + proc.stdout
        self.assertIn("CALLER", message)
        self.assertIn("REQUEST", message)

    def test_the_old_flag_never_becomes_a_report(self):
        """It used to be the flag that answered. It must answer nothing now."""
        proc = run_cli("users", "--humans", "5000", "--agents", "40")
        self.assertNotIn("Plan for the high rate", proc.stdout)
        self.assertNotIn("ops/s", proc.stdout)

    def test_the_old_flag_is_refused_where_the_mix_share_lives_too(self):
        """On tier, calc and validate, `--agents` sits beside `--agent`.

        Left undeclared it would be an unrecognized argument, which never
        names the flag the reader wants; declared, there is a real risk of it
        being read as the mix share. It is refused by name on all three.
        """
        for args in (("tier", "target", "--agents", "40"),
                     ("calc", "--ops", "100", "--agents", "40"),
                     ("validate", "--agents", "40")):
            with self.subTest(args=args):
                proc = run_cli(*args)
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertIn("--agents is no longer a flag",
                              proc.stderr + proc.stdout)

    def test_the_old_flag_is_never_read_as_the_agent_mode_mix_share(self):
        """`calc --ops 100 --agents 60` must not size 60 agent-mode searches."""
        proc = run_cli("calc", "--ops", "100", "--agents", "60", "--json")
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertNotIn("agent_searches_per_s", proc.stdout)

    def test_the_new_flag_gives_what_the_old_one_used_to_give(self):
        """`--automated 40` must reproduce the old `--agents 40` figures.

        5000 x 0.011 + 40 x 0.4 = 55 + 16 = 71 ops/s
        5000 x 0.028 + 40 x 0.4 = 140 + 16 = 156 ops/s
        """
        proc = run_cli("users", "--humans", "5000", "--automated", "40")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("71.00 to 156.00 ops/s", proc.stdout)
        pop = ps.ops_for_population(humans=5000, automated=40)
        self.assertEqual(pop["ops_per_s_low"], 71.0)
        self.assertEqual(pop["ops_per_s_high"], 156.0)

    def test_the_web_address_setting_agents_is_answered_the_same_way(self):
        """"agents" must not be answered with "did you mean agent?".

        `agent` is the mix share. A reader who typed `agents` wants
        `automated`, the count of callers that are programs.
        """
        _, result, error, bad = ps.result_from_query({"ops": ["100"],
                                                      "agents": ["40"]})
        self.assertIsNone(result)
        self.assertIsNone(bad)
        self.assertIn('"automated"', error)
        self.assertNotIn('Did you mean "agent"?', error)


class TestEachPopulationCarriesItsOwnTrafficMix(unittest.TestCase):
    """The model used to force one mix on every caller, so it could not say
    "5,000 people who rarely use multi-hop search, plus 75 automated clients
    that use it constantly" - even though the kind of caller and the kind of
    request go together in practice."""

    # Worked out by hand. 2,500 human chat sessions at the busy rate:
    #   2,500 x 0.028 = 70.0 ops/s, mixed 90 adds / 9 plain / 1 agent-mode.
    # 75 automated clients:
    #   75 x 0.4 = 30.0 ops/s, mixed 10 adds / 20 plain / 70 agent-mode.
    # Total 100.0 ops/s, so each share is the two shares weighted 70 and 30:
    #   adds  = (70 x 90 + 30 x 10) / 100 = (6,300 +   300) / 100 = 66.0
    #   plain = (70 x  9 + 30 x 20) / 100 = (  630 +   600) / 100 = 12.3
    #   agent = (70 x  1 + 30 x 70) / 100 = (   70 + 2,100) / 100 = 21.7
    #   and 66.0 + 12.3 + 21.7 = 100.0
    HUMANS = 2500
    AUTOMATED = 75
    HUMAN_MIX = ps.TrafficMix(90.0, 9.0, 1.0)
    AUTOMATED_MIX = ps.TrafficMix(10.0, 20.0, 70.0)
    BLENDED: ClassVar[dict] = {"add": 66.0, "plain": 12.3, "agent": 21.7}

    def population(self):
        return ps.ops_for_population(self.HUMANS, self.AUTOMATED,
                                     self.HUMAN_MIX, self.AUTOMATED_MIX)

    def test_each_population_demands_its_own_operations_per_second(self):
        pop = self.population()
        self.assertAlmostEqual(pop["human_ops_per_s_low"], 27.5)
        self.assertAlmostEqual(pop["human_ops_per_s_high"], 70.0)
        self.assertAlmostEqual(pop["automated_ops_per_s"], 30.0)
        self.assertAlmostEqual(pop["ops_per_s_low"], 57.5)
        self.assertAlmostEqual(pop["ops_per_s_high"], 100.0)

    def test_the_blended_mix_matches_the_hand_worked_arithmetic(self):
        blended = self.population()["blended_mix"]
        for share, want in self.BLENDED.items():
            with self.subTest(share=share):
                self.assertAlmostEqual(blended[share], want, places=9)

    def test_the_blended_mix_still_adds_up_to_100(self):
        blended = self.population()["blended_mix"]
        self.assertAlmostEqual(sum(blended.values()), 100.0)
        ps.TrafficMix(**blended).validate()

    def test_two_identical_mixes_blend_to_themselves(self):
        """The default mixes must leave every existing answer where it was."""
        pop = ps.ops_for_population(5000, 40)
        self.assertEqual(pop["blended_mix"],
                         {"add": 45.0, "plain": 45.0, "agent": 10.0})

    def test_a_population_of_only_one_kind_takes_that_kind_of_mix(self):
        only_automated = ps.ops_for_population(
            0, 100, self.HUMAN_MIX, self.AUTOMATED_MIX)
        self.assertEqual(only_automated["blended_mix"],
                         self.AUTOMATED_MIX.as_dict())
        only_humans = ps.ops_for_population(
            100, 0, self.HUMAN_MIX, self.AUTOMATED_MIX)
        self.assertEqual(only_humans["blended_mix"], self.HUMAN_MIX.as_dict())

    def test_a_mix_that_does_not_add_up_to_100_is_refused(self):
        with self.assertRaises(ps.SizingError):
            ps.ops_for_population(100, 100, ps.TrafficMix(45.0, 45.0, 5.0))


class TestTheBlendedMixSizesTheDeployment(unittest.TestCase):
    """Reporting a blended mix and then sizing on the global default would be
    a hardware order for traffic nobody described."""

    # The population above: 100.0 ops/s blended to 66 adds / 12.3 plain /
    # 21.7 agent-mode. By hand, at 100 ops/s:
    #   adds            = 66.0/s
    #   plain searches  = 12.3/s
    #   agent searches  = 21.7/s
    #   vector searches = 12.3 x 1 + 21.7 x 22 = 12.3 + 477.4 = 489.7/s
    #   API work        = 489.7 + 66.0 = 555.7/s
    #   API servers     = ceil(555.7 / 108) = ceil(5.145) = 6
    # The same 100 ops/s at the default 45/45/10 mix needs only 3.
    WANT_API_SERVERS = 6
    WANT_AT_THE_DEFAULT_MIX = 3

    def population(self):
        return ps.ops_for_population(2500, 75, ps.TrafficMix(90.0, 9.0, 1.0),
                                     ps.TrafficMix(10.0, 20.0, 70.0))

    def test_the_machines_are_sized_from_the_blended_mix(self):
        sizing = self.population()["sizing"]
        self.assertEqual(sizing["inputs"]["mix"],
                         {"add": 66.0, "plain": 12.3, "agent": 21.7})
        self.assertAlmostEqual(sizing["demand"]["vector_searches_per_s"], 489.7)
        self.assertAlmostEqual(sizing["machines"]["api_work_per_s"], 555.7)
        self.assertEqual(sizing["machines"]["api_servers"],
                         self.WANT_API_SERVERS)

    def test_the_default_mix_would_have_ordered_fewer_machines(self):
        """The two mixes must not be interchangeable, or the test above proves
        nothing."""
        at_default = ps.size_deployment(100.0)
        self.assertEqual(at_default["machines"]["api_servers"],
                         self.WANT_AT_THE_DEFAULT_MIX)
        self.assertGreater(self.WANT_API_SERVERS, self.WANT_AT_THE_DEFAULT_MIX)

    def test_the_report_prints_the_blended_mix_and_the_machines_it_bought(self):
        text = ps.render_users_report(self.population())
        self.assertIn("Blended across the whole population", text)
        self.assertIn("66 adds, 12.3 plain searches, 21.7 agent-mode searches",
                      text)
        self.assertIn("Machines this population needs", text)
        self.assertIn(str(self.WANT_API_SERVERS), text)

    def test_the_command_line_prints_the_same_blended_mix(self):
        proc = run_cli("users", "--humans", "2500", "--automated", "75",
                       "--human-mix", "90/9/1", "--automated-mix", "10/20/70")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("66 adds, 12.3 plain searches, 21.7 agent-mode searches",
                      proc.stdout)

    def test_an_empty_population_is_still_not_a_deployment_to_size(self):
        pop = ps.ops_for_population(0, 0)
        self.assertIsNone(pop["sizing"])


class TestAMixWrittenAsThreeNumbers(unittest.TestCase):
    """`--human-mix 45/45/10` and `--human-mix 45,45,10` are the same mix."""

    def test_both_separators_are_read_the_same_way(self):
        for text in ("45/45/10", "45,45,10", " 45 / 45 / 10 "):
            with self.subTest(text=text):
                self.assertEqual(ps.parse_mix_text(text, "--human-mix"),
                                 ps.TrafficMix(45.0, 45.0, 10.0))

    def test_the_default_is_the_models_own_default_mix(self):
        self.assertEqual(ps.parse_mix_text(ps.default_mix_text(), "x"),
                         ps.TrafficMix())
        self.assertEqual(ps.default_mix_text(), "45/45/10")

    def test_too_few_or_too_many_numbers_are_refused_by_name(self):
        for text in ("45/45", "45/45/10/0", "45"):
            with self.subTest(text=text):
                with self.assertRaises(ps.SizingError) as caught:
                    ps.parse_mix_text(text, "--human-mix")
                self.assertIn("--human-mix", str(caught.exception))
                self.assertIn("adds/plain/agent-mode", str(caught.exception))

    def test_text_where_a_number_belongs_is_quoted_back(self):
        with self.assertRaises(ps.SizingError) as caught:
            ps.parse_mix_text("45/lots/10", "--automated-mix")
        self.assertIn('"lots"', str(caught.exception))

    def test_a_mix_that_does_not_add_up_is_refused(self):
        with self.assertRaises(ps.SizingError) as caught:
            ps.parse_mix_text("45/45/5", "--human-mix")
        self.assertIn("must add up to 100", str(caught.exception))

    def test_the_command_line_refuses_a_bad_mix_without_a_traceback(self):
        for flag, value in (("--human-mix", "45/45"),
                            ("--automated-mix", "45/45/5"),
                            ("--human-mix", "lots")):
            with self.subTest(flag=flag, value=value):
                proc = run_cli("users", "--humans", "10", flag, value)
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn(flag, proc.stderr + proc.stdout)


class TestTheFormTakesACallerPopulationToo(unittest.TestCase):
    """Whatever `users` takes as a flag, the form has to take as a box."""

    def test_both_counts_empty_asks_nothing_about_a_population(self):
        _, result, error, _ = ps.result_from_query({"ops": ["100"]})
        self.assertIsNone(error)
        self.assertIsNone(result["population"],
                          "an unasked question must not be answered")

    def test_a_population_typed_into_the_boxes_is_answered(self):
        _, result, error, _ = ps.result_from_query({
            "ops": ["100"], "humans": ["2500"], "automated": ["75"],
            "human_mix": ["90/9/1"], "automated_mix": ["10,20,70"]})
        self.assertIsNone(error)
        pop = result["population"]
        self.assertEqual(pop["blended_mix"],
                         {"add": 66.0, "plain": 12.3, "agent": 21.7})
        self.assertEqual(pop["sizing"]["machines"]["api_servers"], 6)

    def test_one_count_left_empty_means_none_of_that_kind_of_caller(self):
        _, result, error, _ = ps.result_from_query({"humans": ["2500"],
                                                    "automated": [""]})
        self.assertIsNone(error)
        self.assertEqual(result["population"]["automated"], 0.0)

    def test_the_mix_boxes_start_at_the_models_default_mix(self):
        values, _, _, _ = ps.result_from_query({})
        self.assertEqual(values["human_mix"], "45/45/10")
        self.assertEqual(values["automated_mix"], "45/45/10")

    def test_a_bad_mix_box_names_itself_rather_than_all_twelve(self):
        _, result, error, bad = ps.result_from_query({"humans": ["10"],
                                                      "human_mix": ["45/45"]})
        self.assertIsNone(result)
        self.assertEqual(bad, "human_mix")
        self.assertIn("human traffic mix box", error)

    def test_an_empty_mix_box_is_a_blank_answer(self):
        _, result, error, bad = ps.result_from_query({"humans": ["10"],
                                                      "human_mix": [""]})
        self.assertIsNone(result)
        self.assertEqual(bad, "human_mix")
        self.assertIn("box is empty", error)

    def test_the_page_draws_the_population_tables(self):
        values, result, _, _ = ps.result_from_query({"humans": ["2500"],
                                                     "automated": ["75"]})
        html = ps.render_html(values, result, None, None)
        self.assertIn("Demand from this population", html)
        self.assertIn("Blended across the whole population", html)
        self.assertIn("Automated clients", html)

    def test_the_page_says_nothing_about_a_population_when_none_was_asked(self):
        values, result, _, _ = ps.result_from_query({})
        html = ps.render_html(values, result, None, None)
        self.assertNotIn("Demand from this population", html)

    def test_the_population_boxes_are_labelled_for_a_caller_not_a_request(self):
        values, result, _, _ = ps.result_from_query({})
        html = ps.render_html(values, result, None, None)
        self.assertIn("Concurrent automated client sessions", html)
        self.assertIn("Concurrent human chat sessions", html)
        self.assertNotIn("Software agents", html)


class TestNeitherConceptIsCalledByTheOthersName(unittest.TestCase):
    """The rename is only worth having if the old word is gone everywhere a
    caller is meant, and the product's own word is kept where a request is."""

    def source(self, name):
        with open(os.path.join(HERE, name), encoding="utf-8") as handle:
            return handle.read()

    def test_no_caller_anywhere_is_still_called_a_software_agent(self):
        for name in ("memmachine_sizing.py", "README.md"):
            with self.subTest(name=name):
                text = self.source(name).lower()
                self.assertNotIn("software agent", text)

    def test_the_request_flag_keeps_the_products_own_name(self):
        """agent_mode is the real field on a MemMachine search. Renaming it
        here would misrepresent the API."""
        proc = run_cli("calc", "--help")
        self.assertIn("--agent", proc.stdout)
        self.assertIn("agent-mode searches per 100 operations", proc.stdout)

    def test_the_report_says_the_agent_mode_share_is_a_request_flag(self):
        row = next(r for r in report_section(ps.size_deployment(100.0),
                                             "Inputs")["rows"]
                   if r[0].startswith("Traffic mix"))
        self.assertIn("request flag, not a kind of caller", row[-1])

    def test_the_readme_explains_the_two_concepts_near_the_top(self):
        text = self.source("README.md")
        passage = text[:text.index("## What it does not do")]
        self.assertIn("Agent-mode search is a property of a request", passage)
        self.assertIn("An automated client is a property of a caller", passage)
        self.assertIn("size the deployment wrongly", passage)

    def test_the_module_docstring_explains_them_too(self):
        self.assertIn("Agent-mode search is a property of a REQUEST",
                      ps.__doc__)
        self.assertIn("An automated client is a property of a CALLER",
                      ps.__doc__)

    def test_the_web_page_explains_them_too(self):
        _, result, _, _ = ps.result_from_query({})
        html = ps.render_html({}, result, None, None)
        self.assertIn("Agent-mode search", html)
        self.assertIn("Automated clients", html)
        self.assertIn("how fast a caller sends", html)

    def test_the_exported_numbers_name_the_caller_rate_after_the_caller(self):
        keys = {key for key, _ in ps.constant_numbers()}
        self.assertIn("constants.automated_client_ops_per_s", keys)
        self.assertNotIn("constants.agent_session_ops_per_s", keys)


class TestTheDefaultAnswersDidNotMove(unittest.TestCase):
    """No machine count may change because of the rename. These are the same
    hand-worked figures as the tier tests above, pinned again here so that a
    change made for the sake of wording cannot quietly move an order."""

    def test_every_tier_orders_what_it_ordered_before(self):
        for name, want in TIER_EXPECTATIONS.items():
            with self.subTest(tier=name):
                m = ps.size_deployment(ps.TIER_OPS_PER_S[name])["machines"]
                self.assertEqual(m["api_servers"], want["api_servers"])
                self.assertEqual(m["qdrant_servers"], want["qdrant_nodes"])
                self.assertEqual(m["agent_gpu_cards"], want["agent_cards"])
                self.assertEqual(m["total_cpu_servers"],
                                 want["api_servers"] + 1 + want["qdrant_nodes"])

    def test_calc_at_100_ops_a_second_is_the_target_tier(self):
        proc = run_cli("calc", "--ops", "100")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("3 API server(s)", proc.stdout)


# =============================================================================
# 34. One population, two doors. The page and the command line have to order
#     the same hardware for it.
# =============================================================================


def machines_from_users_report(text: str) -> dict:
    """The machine table the `users` subcommand prints, read back as counts.

    Read out of the printed report rather than out of the library, so that the
    comparison below is between the hardware a person is actually shown on the
    command line and the hardware the web actually answers - not between two
    calls of the same function.
    """
    wanted = (("API server", "api_servers"),
              ("PostgreSQL server", "postgres_servers"),
              ("Qdrant server", "qdrant_servers"),
              ("Embedding GPU card", "embed_gpu_cards"),
              ("Agent-model GPU card", "agent_gpu_cards"))
    counts = {}
    inside = False
    for line in text.splitlines():
        if line.startswith("Machines this population needs"):
            inside = True
            continue
        if not inside:
            continue
        cells = re.split(r"\s{2,}", line.strip())
        if len(cells) < 2:
            continue
        for start, key in wanted:
            if cells[0].startswith(start):
                counts[key] = cells[1].replace(",", "")
                if key == "qdrant_servers":
                    ram = re.search(r"\(([\d,]+) GB RAM each\)", cells[0])
                    counts["qdrant_node_ram_gb"] = ram.group(1).replace(",", "")
    return counts


def machines_from_web_result(result: dict) -> dict:
    """The same five counts out of the JSON endpoint's answer, as text."""
    m = result["machines"]
    low, high = m["embed_gpu_cards_low"], m["embed_gpu_cards_high"]
    return {
        "api_servers": str(m["api_servers"]),
        "postgres_servers": str(m["postgres_servers"]),
        "qdrant_servers": str(m["qdrant_servers"]),
        "qdrant_node_ram_gb": str(m["qdrant_node_ram_gb"]),
        "embed_gpu_cards": str(low) if low == high else f"{low} to {high}",
        "agent_gpu_cards": str(m["agent_gpu_cards"]),
    }


class TestOnePopulationOrdersTheSameHardwareEitherWay(ServedCalculator):
    """The page worked out a blended mix, printed it, and then sized the
    machines from the default mix boxes anyway.

    The same 2,500 people and 75 programs bought 6 API servers on the command
    line and 3 on the page: two hardware orders for one population, depending
    on which door the reader came through.

    This is not the same check as "the form offers every input the command
    line offers". That one compares the names of the boxes against the names
    of the flags, and it passed for as long as this defect existed, because
    every box was there - the answer was simply worked out without them. These
    tests compare the machine counts the two doors return.

    The population, worked out by hand:
      2,500 human chat sessions x 0.028 ops/s = 70 ops/s at the busy end
      75 automated client sessions x 0.4 ops/s = 30 ops/s
      total 100 ops/s, weighted 70:30 between the two mixes
      adds   = (70 x 90 + 30 x 10) / 100 = 66
      plain  = (70 x  9 + 30 x 20) / 100 = 12.3
      agent  = (70 x  1 + 30 x 70) / 100 = 21.7
    and at 100 ops/s with that mix:
      vector searches = 12.3 x 1 + 21.7 x 22 = 489.7/s
      API work        = 489.7 + 66 = 555.7/s
      API servers     = ceil(555.7 / 108) = 6
      episodes        = 66 x 86,400 x 90 = 513,216,000
      hot vector RAM  = 513,216,000 x 1,024 x 1 x 1.5 = 788.3 GB
                        256 GB nodes: ceil(788.3/179.2) = 5, buying 1,280 GB
                        512 GB nodes: ceil(788.3/358.4) = 3, buying 1,536 GB
                        768 GB nodes: ceil(788.3/537.6) = 2, buying 1,536 GB
                        so 5 machines of 256 GB, the least total RAM
      embeds          = 66 + 12.3 x 2 + 21.7 x 22 = 568.0/s
                        cards = ceil(568/300) + 1 = 3 to ceil(568/180) + 1 = 5
      LLM calls       = 21.7 x 1.5 = 32.55/s -> ceil(32.55/15) + 1 = 4 cards
    The same 100 ops/s at the default 45/45/10 mix needs 3 API servers and a
    single 768 GB vector-store machine, so the two mixes are not
    interchangeable and this proves something.
    """

    POPULATION = ("humans=2500&automated=75"
                  "&human_mix=90/9/1&automated_mix=10/20/70")
    WANT: ClassVar[dict] = {
        "api_servers": "6",
        "postgres_servers": "1",
        "qdrant_servers": "5",
        "qdrant_node_ram_gb": "256",
        "embed_gpu_cards": "3 to 5",
        "agent_gpu_cards": "4",
    }

    def test_the_web_endpoint_orders_the_hand_worked_machines(self):
        result = self.json(f"/api/calc?ops=100&{self.POPULATION}")
        self.assertEqual(machines_from_web_result(result), self.WANT)
        self.assertEqual(result["machines"]["api_servers"], 6)
        # The mix that bought them, and the demand it was applied to.
        self.assertEqual(result["inputs"]["mix"],
                         {"add": 66.0, "plain": 12.3, "agent": 21.7})
        self.assertEqual(result["inputs"]["ops_per_s"], 100.0)
        self.assertAlmostEqual(result["demand"]["vector_searches_per_s"], 489.7)
        self.assertAlmostEqual(result["machines"]["api_work_per_s"], 555.7)

    def test_the_typed_design_peak_no_longer_sizes_a_named_population(self):
        """The bug itself: 100 ops/s at the default mix is 3 API servers, and
        that is what the endpoint used to answer for this population."""
        default_mix = self.json("/api/calc?ops=100")
        self.assertEqual(default_mix["machines"]["api_servers"], 3)
        self.assertNotEqual(default_mix["machines"]["api_servers"],
                            int(self.WANT["api_servers"]))

    def test_the_web_and_the_command_line_order_the_same_machines(self):
        proc = run_cli("users", "--humans", "2500", "--automated", "75",
                       "--human-mix", "90/9/1", "--automated-mix", "10/20/70")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        from_the_command_line = machines_from_users_report(proc.stdout)
        from_the_web = machines_from_web_result(
            self.json(f"/api/calc?ops=100&{self.POPULATION}"))
        self.assertEqual(from_the_command_line, from_the_web)
        # And both are the hand-worked order, so a shared mistake in the two
        # would not pass as agreement.
        self.assertEqual(from_the_command_line, self.WANT)

    def test_the_readme_promises_these_counts_through_both_doors(self):
        """The README makes the claim in words; this pins it to the numbers."""
        with open(os.path.join(HERE, "README.md"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("both order 6 API servers, 1 PostgreSQL server, 5 "
                      "vector-store machines of 256 GB,", text)
        self.assertIn("3 to 5 embedding GPU cards and 4 agent-model GPU cards",
                      text)

    def test_the_page_says_the_population_superseded_the_design_peak_box(self):
        status, _, body = self.fetch(f"/?ops=100&{self.POPULATION}")
        self.assertEqual(status, 200)
        self.assertIn("What sized these machines", body)
        self.assertIn("Sized from the caller population, not from the design "
                      "peak box", body)
        self.assertIn("The design peak box, which says &quot;100&quot;, and "
                      "the three traffic-mix boxes were not used", body)

    def test_a_blank_design_peak_box_is_no_fault_when_a_population_sizes_it(self):
        """A box this run does not read is not a box it can complain about.

        With both caller boxes empty the design peak box must still be
        answered, and the test above the fold pins that; here the population
        answers the same question, so clearing the box is allowed."""
        result = self.json(f"/api/calc?ops=&{self.POPULATION}")
        self.assertEqual(result["machines"]["api_servers"], 6)
        self.assertIn("was left empty", result["sized_from_note"])
        status, _, _ = self.fetch("/?ops=&humans=&automated=")
        self.assertEqual(status, 400,
                         "with no population it is still a blank answer")

    def test_the_json_says_what_sized_it_too(self):
        result = self.json(f"/api/calc?ops=100&{self.POPULATION}")
        self.assertEqual(result["inputs"]["sized_from"],
                         ps.SIZED_FROM_POPULATION)
        self.assertIn("not from the design peak box",
                      result["sized_from_note"])

    def test_the_inputs_table_does_not_call_them_numbers_anybody_typed(self):
        result = self.json(f"/api/calc?ops=100&{self.POPULATION}")
        rows = {row[0]: row[-1]
                for row in report_section(result, "Inputs")["rows"]}
        self.assertIn("derived from the caller population",
                      rows["Design peak"])
        self.assertIn("derived from the caller population",
                      rows["Traffic mix per 100 operations"])

    def test_both_caller_boxes_blank_sizes_from_the_design_peak_box(self):
        """Still legal, and nothing about it changed."""
        result = self.json("/api/calc?ops=100&humans=&automated=")
        self.assertIsNone(result["population"])
        self.assertIsNone(result["inputs"]["sized_from"])
        self.assertEqual(result["inputs"]["mix"],
                         {"add": 45.0, "plain": 45.0, "agent": 10.0})
        self.assertEqual(result["machines"]["api_servers"], 3)
        self.assertIn("Sized from the design peak box",
                      result["sized_from_note"])

    def test_only_the_human_box_filled_means_no_automated_clients(self):
        """The empty box is zero of that caller, which is what --automated
        defaulting to 0 does on the command line.

        2,500 x 0.028 = 70 ops/s at the human mix 90/9/1:
          vector searches = 6.3 x 1 + 0.7 x 22 = 21.7/s
          API work        = 21.7 + 63 = 84.7/s -> ceil(84.7/108) = 1 server
        """
        result = self.json("/api/calc?ops=100&humans=2500&human_mix=90/9/1")
        self.assertEqual(result["population"]["automated"], 0.0)
        self.assertEqual(result["inputs"]["ops_per_s"], 70.0)
        self.assertEqual(result["inputs"]["mix"],
                         {"add": 90.0, "plain": 9.0, "agent": 1.0})
        self.assertEqual(result["machines"]["api_servers"], 1)

    def test_only_the_automated_box_filled_means_no_human_sessions(self):
        """75 x 0.4 = 30 ops/s at the automated mix 10/20/70:
          vector searches = 6 x 1 + 21 x 22 = 468/s
          API work        = 468 + 3 = 471/s -> ceil(471/108) = 5 servers
        """
        result = self.json(
            "/api/calc?ops=100&automated=75&automated_mix=10/20/70")
        self.assertEqual(result["population"]["humans"], 0.0)
        self.assertEqual(result["inputs"]["ops_per_s"], 30.0)
        self.assertEqual(result["inputs"]["mix"],
                         {"add": 10.0, "plain": 20.0, "agent": 70.0})
        self.assertEqual(result["machines"]["api_servers"], 5)

    def test_the_command_line_agrees_about_one_box_left_empty(self):
        proc = run_cli("users", "--humans", "2500", "--human-mix", "90/9/1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            machines_from_users_report(proc.stdout)["api_servers"],
            machines_from_web_result(self.json(
                "/api/calc?ops=100&humans=2500&human_mix=90/9/1"
            ))["api_servers"])

    def test_a_population_of_nobody_falls_back_to_the_design_peak_box(self):
        """Zero callers is a real question - it is answered with a demand of
        zero - but it is not a deployment to size: no traffic buys no
        machines, and nobody can order that. So the design peak box governs,
        and the page says which of the two it used rather than leaving the
        reader to work it out from the tables."""
        result = self.json("/api/calc?ops=100&humans=0&automated=0")
        self.assertIsNotNone(result["population"],
                             "the question was asked, so it is answered")
        self.assertEqual(result["population"]["ops_per_s_high"], 0.0)
        self.assertIsNone(result["population"]["sizing"],
                          "no traffic is not a deployment to size")
        self.assertEqual(result["machines"]["api_servers"], 3)
        self.assertEqual(result["inputs"]["ops_per_s"], 100.0)
        self.assertIsNone(result["inputs"]["sized_from"])
        self.assertIn("This population sends nothing at all",
                      result["sized_from_note"])
        self.assertIn("sized from the design peak box instead",
                      result["sized_from_note"].lower())

    def test_the_page_says_so_for_a_population_of_nobody_too(self):
        status, _, body = self.fetch("/?ops=100&humans=0&automated=0")
        self.assertEqual(status, 200)
        self.assertIn("This population sends nothing at all", body)
        self.assertIn("Demand from this population", body,
                      "the zero demand is still reported")

    def test_a_population_of_nobody_never_divides_by_its_own_demand(self):
        """Blending two mixes weights each by the operations it demands, so a
        population that demands nothing has a zero divisor. There must be no
        crash and no not-a-number anywhere in the answer."""
        for humans, automated in ((0, 0), (0.0, 0.0)):
            with self.subTest(humans=humans, automated=automated):
                pop = ps.ops_for_population(humans, automated,
                                            ps.TrafficMix(90, 9, 1),
                                            ps.TrafficMix(10, 20, 70))
                self.assertEqual(pop["ops_per_s_high"], 0.0)
                # With nothing to weight by, the human mix stands unchanged
                # rather than becoming three not-a-numbers.
                self.assertEqual(pop["blended_mix"],
                                 {"add": 90.0, "plain": 9.0, "agent": 1.0})
                for share in pop["blended_mix"].values():
                    self.assertFalse(math.isnan(share))
        text = json.dumps(self.json("/api/calc?humans=0&automated=0"))
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)


class TestTheCallerCountsAreLabelledConcurrent(ServedCalculator):
    """Both counts are multiplied by a rate per session, so they mean sessions
    running at the same moment.

    A reader who types their total registered users into a box labelled only
    "Human chat sessions" gets an order hundreds of times too large, and
    nothing on the page contradicts them. The word has to be on the label, in
    both interfaces, and it must stay there.
    """

    def test_both_population_boxes_say_concurrent_on_the_page(self):
        _, _, body = self.fetch("/")
        self.assertIn("Concurrent human chat sessions", body)
        self.assertIn("Concurrent automated client sessions", body)

    def test_the_hints_say_what_concurrent_rules_out(self):
        _, _, body = self.fetch("/")
        self.assertIn("at the same moment", body)
        self.assertIn("not accounts, not visitors a day", body)

    def test_the_labels_and_the_error_messages_use_the_same_words(self):
        labels = {key: title for key, title, _hint in ps.FORM_FIELDS}
        for key in ("humans", "automated"):
            with self.subTest(box=key):
                self.assertTrue(labels[key].lower().startswith("concurrent"))
                name, what = ps.FORM_FIELD_HELP[key]
                self.assertEqual(name, labels[key].lower())
                self.assertIn("at the same moment", what)

    def test_the_command_line_help_says_concurrent_too(self):
        help_text = run_cli("users", "--help").stdout
        self.assertIn("concurrent human chat sessions", help_text)
        self.assertIn("concurrent automated client sessions", help_text)

    def test_the_reports_name_those_rows_the_same_way(self):
        pop = ps.ops_for_population(2500, 75)
        titles = [row[0] for section in ps.population_sections(pop)
                  for row in section["rows"]]
        self.assertIn("Concurrent human chat sessions", titles)
        self.assertIn("Concurrent automated client sessions", titles)
        held = [row[0] for row in
                report_section(ps.size_deployment(100.0),
                               "Callers this capacity holds")["rows"]]
        for row in held:
            with self.subTest(row=row):
                self.assertTrue(row.startswith("Concurrent"), row)

    def test_every_box_on_the_form_says_what_unit_it_is_in(self):
        """A bare noun on a label is read in whatever unit the reader has in
        mind, so every box names its own."""
        units = {
            "ops": "operations/s",
            "add": "per 100 operations",
            "plain": "per 100 operations",
            "agent": "per 100 operations",
            "retention_days": "days",
            "dims": "numbers per vector",
            "bytes_per_value": "Bytes",
            "node_gb": "GB",
            "humans": "sessions",
            "automated": "sessions",
            "human_mix": "adds/plain/agent-mode",
            "automated_mix": "adds/plain/agent-mode",
            "human_users": "People",
            "human_peak_share": "per 100 users",
            "human_sessions_per_active_user": "Sessions per active person",
            "automated_users": "Users",
            "automated_peak_share": "per 100 users",
            "automated_sessions_per_active_user": "sessions per active user",
        }
        for key, title, hint in ps.FORM_FIELDS:
            with self.subTest(box=key):
                self.assertIn(units[key], f"{title} {hint}")


# =============================================================================
# 36. A user count is not a count of concurrent sessions. The model multiplies
#     concurrent sessions by a rate per session; whoever commissions a
#     deployment answers in users. Two figures turn one into the other, they
#     multiply, and neither has a default.
# =============================================================================


def population_section(pop: dict, title_starts_with: str) -> dict:
    """One section of the population report, by the start of its title."""
    for section in ps.population_sections(pop):
        if section["title"].startswith(title_starts_with):
            return section
    raise AssertionError(f"the report has no {title_starts_with} section")


def conversion_rows(pop: dict) -> dict:
    """The conversion table's rows, keyed by the name in the first cell."""
    section = population_section(pop, "From users to concurrent sessions")
    return {row[0]: row for row in section["rows"]}


# The worked example these tests share, and the numbers it must produce.
#
#   People:            50,000 users x 2 per 100 active x 1 session   = 1,000
#   Automated clients:    200 users x 25 per 100 active x 20 sessions = 1,000
#
# Two very different user counts, two different shares and two different
# sessions per active user, arriving at the same number of sessions - which is
# the point: neither session count can be read off its user count.
#
#   1,000 human sessions    x 0.028 ops/s = 28 ops/s at the busy end
#   1,000 automated sessions x 0.4 ops/s = 400 ops/s
#   total 428 ops/s, at the default 45/45/10 mix on both populations
#     adds            = 428 x 0.45 = 192.6/s
#     plain searches  = 428 x 0.45 = 192.6/s
#     agent searches  = 428 x 0.10 =  42.8/s
#     vector searches = 192.6 x 1 + 42.8 x 22 = 1,134.2/s
#     API work        = 1,134.2 + 192.6 = 1,326.8/s
#     API servers     = ceil(1,326.8 / 108) = 13
WORKED_USERS_CLI = (
    "--human-users", "50000", "--human-peak-share", "2",
    "--human-sessions-per-active-user", "1",
    "--automated-users", "200", "--automated-peak-share", "25",
    "--automated-sessions-per-active-user", "20")
WORKED_USERS_QUERY = (
    "human_users=50000&human_peak_share=2&human_sessions_per_active_user=1"
    "&automated_users=200&automated_peak_share=25"
    "&automated_sessions_per_active_user=20")


def flat(text: str) -> str:
    """One long line, so a wrapped paragraph can be searched as a sentence."""
    return " ".join(text.split())


class TestABareUserCountNowAnswers(unittest.TestCase):
    """The refusal became an example default, and the warning survived.

    A user count that arrived without both conversion figures used to exit 2
    with a long refusal. The two figures now have example defaults, so the
    calculator always answers - and everything the refusal said is printed as
    a note with the conversion table instead of being lost.
    """

    def report(self, *args) -> str:
        proc = run_cli("users", *args)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotEqual(proc.stdout, "", "an answer must print a report")
        return flat(proc.stdout)

    def refusal(self, *args) -> str:
        proc = run_cli("users", *args)
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertEqual(proc.stdout, "", "a refusal must print no report")
        return proc.stderr.strip()

    def test_a_user_count_with_neither_figure_now_answers(self):
        """The whole point of this change: it used to be exit code 2."""
        text = self.report("--human-users", "50000")
        self.assertIn("From users to concurrent sessions", text)
        # 50,000 x 10 per 100 x 1 session = 5,000 concurrent sessions.
        self.assertIn("Concurrent sessions 5,000", text)

    def test_the_four_defaults_are_the_agreed_values(self):
        self.assertEqual(ps.DEFAULT_HUMAN_PEAK_ACTIVE_PER_100, 10.0)
        self.assertEqual(ps.DEFAULT_HUMAN_SESSIONS_PER_ACTIVE_USER, 1.0)
        self.assertEqual(ps.DEFAULT_AUTOMATED_PEAK_ACTIVE_PER_100, 100.0)
        self.assertEqual(ps.DEFAULT_AUTOMATED_SESSIONS_PER_ACTIVE_USER, 1.0)
        self.assertEqual(ps.HUMAN_PEAK_ACTIVE_PER_100_LOW, 5.0)
        self.assertEqual(ps.HUMAN_PEAK_ACTIVE_PER_100_HIGH, 20.0)

    def test_each_default_carries_its_own_honest_label(self):
        """Convention, and no evidence at all, are different claims."""
        rows = conversion_rows(ps.ops_for_population(
            humans=5000.0, automated=200.0,
            human_conversion=ps.UsersToSessions(
                50000.0, 10.0, 1.0, peak_active_per_100_is_default=True,
                sessions_per_active_user_is_default=True),
            automated_conversion=ps.UsersToSessions(
                200.0, 100.0, 1.0, peak_active_per_100_is_default=True,
                sessions_per_active_user_is_default=True)))
        share = rows["Share of those users active at the busiest moment"][-1]
        per_user = rows["Sessions per active user"][-1]
        self.assertIn("people: example default - a convention, not a "
                      "measurement", share)
        self.assertIn("people: example default - no published evidence "
                      "exists", per_user)
        for label in (share, per_user):
            with self.subTest(label=label[:30]):
                self.assertIn("automated clients: example default - a client "
                              "that is running is already a session", label)

    def test_the_share_default_is_called_a_convention_not_a_measurement(self):
        """Two conventions from two fields that agree is not a measurement,
        and the help must not claim more than that."""
        help_text = flat(run_cli("users", "--help").stdout)
        self.assertIn("It is a convention, not a measurement", help_text)
        self.assertIn("SharePoint capacity guidance assumes 10 percent "
                      "concurrency", help_text)
        self.assertIn("teletraffic engineering uses 10-16% of subscribers "
                      "busy in the busy hour", help_text)
        self.assertIn("Two conventions from two fields that agree is weaker "
                      "evidence than one measurement", help_text)
        self.assertIn("Plausible range 5 to 20", help_text)

    def test_the_sessions_default_says_no_published_evidence_exists(self):
        help_text = flat(run_cli("users", "--help").stdout)
        self.assertIn("No published evidence exists for this figure",
                      help_text)

    def test_the_automated_side_defaults_to_identity(self):
        """A client that is running IS a session, so nothing is converted."""
        text = self.report("--humans", "0", "--automated-users", "200")
        self.assertIn("100 per 100 users", text)
        self.assertIn("Concurrent sessions 0 200", text)
        pop = ps.ops_for_population(
            humans=0.0, automated=200.0,
            automated_conversion=ps.UsersToSessions(
                200.0, ps.DEFAULT_AUTOMATED_PEAK_ACTIVE_PER_100,
                ps.DEFAULT_AUTOMATED_SESSIONS_PER_ACTIVE_USER,
                peak_active_per_100_is_default=True,
                sessions_per_active_user_is_default=True))
        self.assertEqual(pop["conversion"]["automated"]["concurrent_sessions"],
                         200.0)

    def test_the_help_warns_against_applying_the_human_share_to_programs(self):
        help_text = flat(run_cli("users", "--help").stdout)
        self.assertIn("10-per-100 convention describes people, most of whom "
                      "are asleep or busy", help_text)
        self.assertIn("applying it to load generators or deployed clients "
                      "would divide the load by ten", help_text)

    def test_the_note_names_which_figures_were_defaulted(self):
        text = self.report("--human-users", "50000")
        self.assertIn("EXAMPLE DEFAULTS WERE USED HERE", text)
        self.assertIn("The share of people active at the busiest moment and "
                      "the sessions one active person holds are not figures "
                      "you gave", text)
        # The automated side was not used at all, so it is not named.
        self.assertNotIn("the share of automated-client users", text)

    def test_the_note_names_only_the_figure_that_was_defaulted(self):
        text = self.report("--human-users", "50000",
                           "--human-peak-share", "2")
        self.assertIn("The sessions one active person holds is not a figure "
                      "you gave", text)
        self.assertNotIn("The share of people active at the busiest moment "
                         "and", text)

    def test_the_note_says_the_share_can_be_wrong_by_a_factor_of_two(self):
        text = self.report("--human-users", "50000")
        self.assertIn("10 per 100 users is a convention rather than a "
                      "measurement, the plausible range is 5 to 20 per 100, "
                      "so it can be wrong by a factor of two either way",
                      text)

    def test_the_note_says_what_to_replace_the_examples_with(self):
        text = self.report("--human-users", "50000")
        self.assertIn("meter the concurrent sessions the deployment actually "
                      "holds - callers sending requests at the same moment - "
                      "from the first day of running, and give those session "
                      "counts instead of a user count", text)

    def test_the_note_keeps_what_the_old_refusal_said(self):
        """The refusal text was good. None of it may be lost."""
        text = self.report("--human-users", "50000")
        self.assertIn("A user count is not a count of concurrent sessions",
                      text)

    def test_there_is_no_note_when_every_figure_was_supplied(self):
        """A reader who gave their own numbers is not warned about numbers
        they did not use."""
        text = self.report(*WORKED_USERS_CLI)
        self.assertNotIn("EXAMPLE DEFAULTS WERE USED HERE", text)
        self.assertEqual(ps.defaulted_figures_note(None), "")

    def test_supplied_figures_are_marked_differently_from_defaults(self):
        """One kind supplied, the other defaulted, in the same row."""
        pop = ps.ops_for_population(
            humans=1000.0, automated=200.0,
            human_conversion=ps.UsersToSessions(50000.0, 2.0, 1.0),
            automated_conversion=ps.UsersToSessions(
                200.0, 100.0, 1.0, peak_active_per_100_is_default=True,
                sessions_per_active_user_is_default=True))
        label = conversion_rows(pop)[
            "Share of those users active at the busiest moment"][-1]
        self.assertIn("people: supplied", label)
        self.assertIn("automated clients: example default", label)
        self.assertNotIn("people: example default", label)

    def test_a_conversion_figure_with_no_user_count_is_refused_still(self):
        """The reverse mistake: a share of nothing says nothing, and that is
        still an error rather than something a default can rescue."""
        message = self.refusal("--humans", "1000", "--human-peak-share", "2")
        self.assertIn("--human-peak-share is given, but --human-users is "
                      "missing", message)

    def test_a_share_above_a_hundred_per_hundred_users_is_refused(self):
        message = self.refusal("--human-users", "50000",
                               "--human-peak-share", "150",
                               "--human-sessions-per-active-user", "1")
        self.assertIn("--human-peak-share is 150", message)
        self.assertIn("cannot be more than 100", message)

    def test_the_web_form_answers_a_bare_user_count_too(self):
        values, result, error, _ = ps.result_from_query(
            {"human_users": ["50000"]})
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        del values
        conversion = result["population"]["conversion"]["human"]
        self.assertEqual(conversion["peak_active_per_100"], 10.0)
        self.assertTrue(conversion["peak_active_per_100_is_default"])
        self.assertTrue(conversion["sessions_per_active_user_is_default"])
        self.assertEqual(conversion["concurrent_sessions"], 5000.0)

    def test_both_doors_give_the_same_warning(self):
        """The names differ - flags there, boxes here - and nothing else may."""
        page = ps.result_from_query({"human_users": ["50000"]})[1]
        page_note = flat(ps.conversion_sections(page["population"])[0]["note"])
        command_line = self.report("--human-users", "50000")
        for sentence in ("EXAMPLE DEFAULTS WERE USED HERE",
                         "A user count is not a count of concurrent sessions"):
            with self.subTest(sentence=sentence[:40]):
                self.assertIn(sentence, page_note)
                self.assertIn(sentence, command_line)


class TestTheCountingRuleIsStatedEverywhere(unittest.TestCase):
    """What a session is, in the four places a reader could start from.

    The rule decides the answer: one developer driving a ten-user load test is
    ten sessions, so reading it as one caller under-counts the load tenfold.
    """

    def test_the_rule_says_what_a_session_is_and_gives_the_load_test_case(self):
        rule = flat(ps.COUNTING_RULE)
        self.assertIn("a session sending requests at the same moment, whoever "
                      "is behind it", rule)
        self.assertIn("One developer driving a ten-user load test is ten "
                      "sessions, not one", rule)
        self.assertIn("give that number directly as a count of concurrent "
                      "sessions and skip the conversion", rule)

    def test_the_module_docstring_states_it(self):
        doc = flat(ps.__doc__)
        self.assertIn("THE COUNTING RULE", doc)
        self.assertIn("One developer driving a ten-user load test is ten "
                      "sessions, not one", doc)
        self.assertIn("--humans and --automated on the command line", doc)

    def test_the_users_help_states_it(self):
        help_text = flat(run_cli("users", "--help").stdout)
        self.assertIn(flat(ps.COUNTING_RULE), help_text)
        self.assertIn(flat(ps.COUNTING_RULE_FLAGS), help_text)

    def test_the_report_states_it(self):
        """Both doors into the population report: the run that converts a
        user count, and the run that is given sessions directly."""
        converted = flat(run_cli("users", "--human-users", "50000").stdout)
        self.assertIn(flat(ps.COUNTING_RULE), converted)
        sessions = flat(run_cli("users", "--humans", "2500").stdout)
        self.assertIn(flat(ps.COUNTING_RULE), sessions)

    def test_the_report_does_not_state_it_twice_in_a_row(self):
        """The conversion table opens with it; the demand table under it must
        not repeat the same paragraph."""
        converted = flat(run_cli("users", "--human-users", "50000").stdout)
        self.assertEqual(converted.count(flat(ps.COUNTING_RULE)), 1)

    def test_the_readme_states_it(self):
        with open(os.path.join(HERE, "README.md"), encoding="utf-8") as handle:
            text = flat(handle.read())
        self.assertIn("One developer driving a ten-user load test is ten "
                      "sessions, not one", text)
        self.assertIn("a session sending requests at the same moment, whoever "
                      "is behind it", text)


class TestThePeakToAverageFigureIsReportedAndNeverCounted(unittest.TestCase):
    """A multiplier for a reader who knows their average load, and nothing
    else. Changing it must not move one machine."""

    def test_the_figures_are_the_agreed_values(self):
        self.assertEqual(ps.PEAK_TO_AVERAGE_SINGLE_ORGANISATION, 4.0)
        self.assertEqual(ps.PEAK_TO_AVERAGE_MEASURED_SMALL_SERVICE, 4.64)
        self.assertEqual(ps.PEAK_TO_AVERAGE_LARGE_SERVICE_LOW, 1.3)
        self.assertEqual(ps.PEAK_TO_AVERAGE_LARGE_SERVICE_HIGH, 1.64)

    def test_the_note_gives_the_range_the_reason_and_both_sources(self):
        note = flat(ps.design_peak_note())
        self.assertIn("multiply it by 4 to get a design peak for a deployment "
                      "inside one organisation", note)
        self.assertIn("REPORTED ONLY: no machine count in this program reads "
                      "it", note)
        self.assertIn("4.64 times its own mean", note)
        self.assertIn("Hotmail at 1.64 times", note)
        self.assertIn("Google production cell of 12,500 machines ran at 1.3 "
                      "times", note)
        self.assertIn("scale flattens the curve", note)
        self.assertIn("https://arxiv.org/abs/1207.6295", note)
        self.assertIn("https://www.cs.virginia.edu/~cr4bd/papers/socc12.pdf",
                      note)

    def test_the_report_carries_it(self):
        note = flat(report_section(ps.size_deployment(100.0), "Inputs")["note"])
        self.assertIn("multiply it by 4", note)
        population = flat(population_section(
            ps.ops_for_population(humans=2500.0, automated=75.0),
            "Demand from this population")["note"])
        self.assertIn("multiply it by 4", population)

    def test_changing_it_moves_no_machine_count(self):
        """The proof that it is reported only."""
        def counts():
            tier = ps.size_deployment(100.0)["machines"]
            pop = ps.ops_for_population(humans=2500.0,
                                        automated=75.0)["sizing"]["machines"]
            return tier, pop
        before = counts()
        originals = {name: getattr(ps, name) for name in (
            "PEAK_TO_AVERAGE_SINGLE_ORGANISATION",
            "PEAK_TO_AVERAGE_MEASURED_SMALL_SERVICE",
            "PEAK_TO_AVERAGE_LARGE_SERVICE_LOW",
            "PEAK_TO_AVERAGE_LARGE_SERVICE_HIGH")}
        try:
            for name in originals:
                setattr(ps, name, 99.0)
            after = counts()
            note = flat(ps.design_peak_note())
        finally:
            for name, value in originals.items():
                setattr(ps, name, value)
        self.assertEqual(before, after)
        # And the change really did reach the text, so the test is not
        # passing because nothing happened at all.
        self.assertIn("multiply it by 99", note)


class TestTheFiveMinuteWindowIsCited(unittest.TestCase):
    """The design peak is a five-minute figure, and five minutes is an
    engineering convention with a standard behind it, not a house rule."""

    def test_the_citation_names_the_recommendation_and_quotes_it(self):
        note = flat(ps.design_peak_note())
        self.assertIn("ITU-T E.500", note)
        self.assertIn("greater than 5 minutes... so that resources are not "
                      "dimensioned for infrequent small interval peak traffic "
                      "levels", note)
        self.assertIn("https://www.itu.int/rec/T-REC-E.500-199811-I/en", note)

    def test_the_docstring_cites_it_where_the_design_peak_is_defined(self):
        doc = flat(ps.__doc__)
        self.assertIn("Each is a DESIGN PEAK - the worst rate the system must "
                      "sustain for five minutes", doc)
        self.assertIn("ITU-T E.500, the international recommendation for "
                      "measuring telephone traffic, independently requires "
                      "measurement windows", doc)

    def test_the_readme_cites_it(self):
        with open(os.path.join(HERE, "README.md"), encoding="utf-8") as handle:
            text = flat(handle.read())
        self.assertIn("ITU-T E.500", text)
        self.assertIn("T-REC-E.500-199811-I", text)


class TestUsersConvertToConcurrentSessionsByHand(unittest.TestCase):
    """The worked conversion, with the two caller kinds unlike each other.

    50,000 people, 2 of every 100 active at the busiest moment, 1 session each
    = 1,000 concurrent human chat sessions.
    200 users of automated clients, 25 of every 100 active, 20 sessions each
    = 1,000 concurrent automated client sessions.
    Then 1,000 x 0.028 + 1,000 x 0.4 = 428 operations/s, which needs
    ceil(1,326.8 / 108) = 13 API servers at the default mix.
    """

    def population(self):
        return ps.ops_for_population(
            humans=1000.0, automated=1000.0,
            human_conversion=ps.UsersToSessions(50000.0, 2.0, 1.0),
            automated_conversion=ps.UsersToSessions(200.0, 25.0, 20.0))

    def test_the_three_figures_multiply_to_the_concurrent_sessions(self):
        self.assertEqual(ps.UsersToSessions(50000.0, 2.0, 1.0).sessions(),
                         1000.0)
        self.assertEqual(ps.UsersToSessions(200.0, 25.0, 20.0).sessions(),
                         1000.0)

    def test_the_command_line_converts_and_then_sizes_from_the_result(self):
        proc = run_cli("users", *WORKED_USERS_CLI)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        counts = machines_from_users_report(proc.stdout)
        self.assertEqual(counts["api_servers"], "13")
        self.assertIn("Sized at 428.00 operations/s", proc.stdout)
        self.assertIn("1,000", proc.stdout)

    def test_the_conversion_table_shows_the_multiplication_row_by_row(self):
        rows = conversion_rows(self.population())
        self.assertEqual(rows["Users"][1:3], ["50,000", "200"])
        self.assertEqual(
            rows["Share of those users active at the busiest moment"][1:3],
            ["2 per 100 users", "25 per 100 users"])
        self.assertEqual(rows["Sessions per active user"][1:3], ["1", "20"])
        self.assertEqual(rows["Concurrent sessions"][1:3],
                         ["1,000", "1,000"])

    def test_the_two_supplied_figures_are_labelled_as_the_readers(self):
        """Not this model's assumptions. The label has to say whose they are."""
        rows = conversion_rows(self.population())
        for name in ("Share of those users active at the busiest moment",
                     "Sessions per active user"):
            with self.subTest(row=name):
                self.assertEqual(rows[name][-1],
                                 "assumption supplied by the reader, "
                                 "not by this model")
        self.assertEqual(rows["Users"][-1], "given by the reader")
        self.assertEqual(rows["Concurrent sessions"][-1],
                         "derived: users x share active / 100 x sessions per "
                         "active user")

    def test_a_caller_kind_given_as_sessions_says_so_in_its_column(self):
        """One kind converted from users, the other typed in as sessions."""
        pop = ps.ops_for_population(
            humans=1000.0, automated=75.0,
            human_conversion=ps.UsersToSessions(50000.0, 2.0, 1.0))
        rows = conversion_rows(pop)
        self.assertEqual(rows["Users"][1:3], ["50,000", "given as sessions"])
        self.assertEqual(rows["Concurrent sessions"][1:3], ["1,000", "75"])

    def test_the_conversion_is_the_first_table_in_the_report(self):
        """It has to be read before the demand it produced, or the session
        counts below it look like numbers somebody typed."""
        titles = [section["title"]
                  for section in ps.population_sections(self.population())]
        self.assertEqual(titles[0], "From users to concurrent sessions")
        self.assertEqual(titles[1], "Demand from this population")

    def test_the_conversion_travels_with_the_population_in_the_json(self):
        pop = self.population()
        self.assertEqual(pop["conversion"]["human"], {
            "users": 50000.0, "peak_active_per_100": 2.0,
            "peak_active_per_100_is_default": False,
            "sessions_per_active_user": 1.0,
            "sessions_per_active_user_is_default": False,
            "concurrent_sessions": 1000.0})
        self.assertEqual(pop["conversion"]["automated"]["concurrent_sessions"],
                         1000.0)
        self.assertEqual(pop["ops_per_s_high"], 428.0)
        self.assertEqual(pop["sizing"]["machines"]["api_servers"], 13)


class TestAUserCountIsNotAcceptedAsAConcurrentSessionCount(unittest.TestCase):
    """The mistake this whole layer exists to stop.

    50,000 users typed into the concurrent-sessions box is 50,000 x 0.028 =
    1,400 operations/s, which orders 41 API servers - no named tier holds it.
    The same 50,000 users, 2 of every 100 active at the busiest moment holding
    1 session each, is 1,000 concurrent sessions and 28 operations/s, which
    orders 1 API server in the target tier. Forty-one machines against one.
    """

    def test_the_same_number_means_two_wildly_different_orders(self):
        as_sessions = ps.ops_for_population(humans=50000.0, automated=0.0)
        self.assertEqual(as_sessions["ops_per_s_high"], 1400.0)
        self.assertEqual(as_sessions["sizing"]["machines"]["api_servers"], 41)
        self.assertIsNone(as_sessions["tier_for_high"],
                          "no named tier holds 1,400 operations/s")
        as_users = ps.ops_for_population(
            humans=1000.0, automated=0.0,
            human_conversion=ps.UsersToSessions(50000.0, 2.0, 1.0))
        self.assertEqual(as_users["ops_per_s_high"], 28.0)
        self.assertEqual(as_users["sizing"]["machines"]["api_servers"], 1)
        self.assertEqual(as_users["tier_for_high"], "target")

    def test_the_session_flag_never_converts_what_it_is_given(self):
        """--humans is read as sessions and nothing is done to it, so a user
        count typed there cannot come back quietly reduced either."""
        proc = run_cli("users", "--humans", "50000")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Concurrent human chat sessions", proc.stdout)
        self.assertIn("1,400.00", proc.stdout)
        self.assertNotIn("From users to concurrent sessions", proc.stdout)

    def test_one_caller_kind_counted_both_ways_is_refused(self):
        """Neither count silently wins over the other."""
        proc = run_cli("users", "--humans", "1000", "--human-users", "50000")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--humans and --human-users are both given", proc.stderr)
        self.assertIn("Give one or the other, not both", proc.stderr)
        _, result, error, _ = ps.result_from_query(
            {"humans": ["1000"], "human_users": ["50000"]})
        self.assertIsNone(result)
        self.assertIn("are both given", error)

    def test_the_readme_quotes_that_cost_in_machines_correctly(self):
        """The README makes the 41-against-1 claim in words. This pins it to
        the program, so the passage cannot go stale."""
        with open(os.path.join(HERE, "README.md"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("1,400 operations per second and **41 API servers**",
                      text)
        self.assertIn("1,000 concurrent sessions, which is 28 operations per "
                      "second and\n**1 API server**", text)
        self.assertIn("concurrent sessions = users x share active / 100 x "
                      "sessions per active user", text)

    def test_the_two_are_separate_inputs_on_both_doors(self):
        """A count box and a user box, never one box asked to be both."""
        boxes = {key for key, _, _ in ps.FORM_FIELDS}
        self.assertIn("humans", boxes)
        self.assertIn("human_users", boxes)
        help_text = run_cli("users", "--help").stdout
        self.assertIn("--humans", help_text)
        self.assertIn("--human-users", help_text)
        self.assertIn("not a session count", help_text)


class TestTheUsersFlagsAndTheFormBoxesAgree(ServedCalculator):
    """Whatever the users subcommand takes as a flag, the form takes as a box,
    and the two order the same machines for the same users - not merely the
    same names on the same page."""

    def test_the_web_and_the_command_line_convert_users_the_same_way(self):
        proc = run_cli("users", *WORKED_USERS_CLI)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        from_the_command_line = machines_from_users_report(proc.stdout)
        from_the_web = machines_from_web_result(
            self.json(f"/api/calc?{WORKED_USERS_QUERY}"))
        self.assertEqual(from_the_command_line, from_the_web)
        # And both are the hand-worked order, so a shared mistake would not
        # pass as agreement.
        self.assertEqual(from_the_command_line["api_servers"], "13")

    def test_the_web_answer_carries_the_conversion_and_the_demand(self):
        result = self.json(f"/api/calc?{WORKED_USERS_QUERY}")
        pop = result["population"]
        self.assertEqual(pop["humans"], 1000.0)
        self.assertEqual(pop["automated"], 1000.0)
        self.assertEqual(pop["conversion"]["human"]["users"], 50000.0)
        self.assertEqual(pop["conversion"]["automated"]
                         ["sessions_per_active_user"], 20.0)
        self.assertEqual(result["inputs"]["ops_per_s"], 428.0)
        self.assertEqual(result["machines"]["api_servers"], 13)

    def test_every_new_flag_has_a_box_of_the_same_name(self):
        flags = run_cli("users", "--help").stdout
        boxes = {key for key, _, _ in ps.FORM_FIELDS}
        for key in ("human_users", "human_peak_share",
                    "human_sessions_per_active_user", "automated_users",
                    "automated_peak_share",
                    "automated_sessions_per_active_user"):
            with self.subTest(box=key):
                self.assertIn(key, boxes)
                self.assertIn("--" + key.replace("_", "-"), flags)
                self.assertIn(key, ps.FORM_DEFAULTS)
                self.assertIsNone(ps.FORM_DEFAULTS[key],
                                  "the box starts empty; the example default "
                                  "is applied when a user count is given "
                                  "without it")

    def test_the_page_draws_the_conversion_table_and_says_it_converted(self):
        status, _, body = self.fetch(f"/?{WORKED_USERS_QUERY}")
        self.assertEqual(status, 200)
        self.assertIn("From users to concurrent sessions", body)
        self.assertIn("Those session counts were not typed in", body)
        self.assertIn("assumption supplied by the reader, not by this model",
                      body)

    def test_the_page_offers_the_six_boxes_and_names_their_examples(self):
        _, _, body = self.fetch("/")
        for key in ("human_users", "human_peak_share",
                    "human_sessions_per_active_user", "automated_users",
                    "automated_peak_share",
                    "automated_sessions_per_active_user"):
            with self.subTest(box=key):
                self.assertIn(f'name="{key}"', body)
        self.assertIn("example default", body)
        self.assertIn("blank uses 10, an example - a convention, not a "
                      "measurement", body)
        self.assertIn("blank uses 1, an example - no published evidence "
                      "exists", body)

    def test_the_page_tells_the_reader_which_boxes_to_clear(self):
        """Telling somebody to clear a caller-count box they left empty helps
        nobody, so the advice names the boxes they actually filled in."""
        converted = self.json(f"/api/calc?{WORKED_USERS_QUERY}")
        self.assertIn("Clear the two caller-count boxes and the six user "
                      "boxes", converted["sized_from_note"])
        sessions = self.json("/api/calc?humans=2500&automated=75")
        self.assertIn("Clear both caller-count boxes",
                      sessions["sized_from_note"])

    def test_a_user_count_alone_now_answers_on_the_web_too(self):
        """It used to be a 400 whose reason named the two missing figures."""
        status, _, body = self.fetch("/api/calc?human_users=50000")
        self.assertEqual(status, 200)
        conversion = json.loads(body)["population"]["conversion"]["human"]
        self.assertEqual(conversion["concurrent_sessions"], 5000.0)
        self.assertTrue(conversion["peak_active_per_100_is_default"])


class TestGivingConcurrentSessionsDirectlyDidNotMove(unittest.TestCase):
    """Nobody who already knows their concurrent sessions may see any change.

    The same 2,500 people and 75 programs as section 34, pinned again here so
    that a change made for the sake of the new layer cannot move an order.
    """

    def test_the_older_population_orders_exactly_what_it_did(self):
        proc = run_cli("users", "--humans", "2500", "--automated", "75",
                       "--human-mix", "90/9/1", "--automated-mix", "10/20/70")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(machines_from_users_report(proc.stdout), {
            "api_servers": "6", "postgres_servers": "1",
            "qdrant_servers": "5", "qdrant_node_ram_gb": "256",
            "embed_gpu_cards": "3 to 5", "agent_gpu_cards": "4"})

    def test_a_run_without_a_user_count_prints_no_conversion_table(self):
        proc = run_cli("users", "--humans", "5000", "--automated", "40")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("From users to concurrent sessions", proc.stdout)
        pop = ps.ops_for_population(humans=5000, automated=40)
        self.assertIsNone(pop["conversion"])
        first = next(s["title"] for s in ps.population_sections(pop))
        self.assertEqual(first, "Demand from this population")

    def test_the_automated_count_still_defaults_to_none_of_them(self):
        """--automated used to default to 0 and now defaults to "not given",
        so that a user count for the same kind is not read as a second answer
        to the same question. None of them is still what it comes to."""
        proc = run_cli("users", "--humans", "100")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = next(line for line in proc.stdout.splitlines()
                   if line.strip().startswith(
                       "Concurrent automated client sessions"))
        self.assertIn("0.00 ops/s", row)

    def test_the_users_subcommand_still_insists_on_a_human_count(self):
        """It always has. There are two ways to answer now, and the refusal
        names both instead of only the older one."""
        proc = run_cli("users")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("needs to know how many human callers there are",
                      proc.stderr)
        self.assertIn("Give --humans", proc.stderr)
        self.assertIn("or give --human-users", proc.stderr)
        self.assertIn("pass --humans 0", proc.stderr)

    def test_an_empty_form_still_asks_nothing_about_users(self):
        _, result, error, _ = ps.result_from_query({"ops": ["100"]})
        self.assertIsNone(error)
        self.assertIsNone(result["population"])
        html = ps.render_html({}, result, None, None)
        self.assertNotIn("From users to concurrent sessions", html)


# =============================================================================
# 37. The two per-caller rates now cite published measurements, and the two
#     new figures those measurements revealed are reported and never counted.
#     Neither number moved. What changed is that a reader can check where it
#     came from, and that the report now says what a heavy human session and
#     an idle-most-of-the-time automated client actually demand.
# =============================================================================


class TestThePerCallerRatesCiteTheirSources(unittest.TestCase):
    """BurstGPT for the human rates, TraceLab for the automated ones."""

    def source(self, name):
        with open(os.path.join(HERE, name), encoding="utf-8") as handle:
            return handle.read()

    def test_the_four_rate_constants_are_the_numbers_the_model_agreed(self):
        """The two old ones did not move. The two new ones are the figures the
        published measurements added."""
        self.assertEqual(ps.HUMAN_SESSION_OPS_PER_S_LOW, 0.011)
        self.assertEqual(ps.HUMAN_SESSION_OPS_PER_S_HIGH, 0.028)
        self.assertEqual(ps.HUMAN_SESSION_OPS_PER_S_HEAVY, 0.06)
        self.assertEqual(ps.AUTOMATED_CLIENT_OPS_PER_S, 0.4)
        self.assertEqual(ps.AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED, 0.07)

    def test_the_heavy_session_is_about_twice_the_busy_end_of_the_band(self):
        """The 99th percentile against the 90th, from the same measurement."""
        ratio = (ps.HUMAN_SESSION_OPS_PER_S_HEAVY
                 / ps.HUMAN_SESSION_OPS_PER_S_HIGH)
        self.assertGreater(ratio, 1.9)
        self.assertLess(ratio, 2.4)

    def test_the_design_peak_is_about_six_times_the_sustained_rate(self):
        """The claim the report and the README both make in words."""
        ratio = (ps.AUTOMATED_CLIENT_OPS_PER_S
                 / ps.AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED)
        self.assertGreater(ratio, 5.0)
        self.assertLess(ratio, 6.5)

    def test_the_module_docstring_cites_both_papers_by_name_and_address(self):
        doc = " ".join(ps.__doc__.split())
        self.assertIn("BurstGPT: A Real-world Workload Dataset to Optimize "
                      "LLM Serving Systems", doc)
        self.assertIn("2025", doc)
        self.assertIn("https://arxiv.org/abs/2401.17644", doc)
        self.assertIn("TraceLab", doc)
        self.assertIn("2026", doc)
        self.assertIn("https://arxiv.org/abs/2606.30560", doc)

    def test_the_docstring_says_what_each_source_is_and_is_not(self):
        """Neither source is a measurement of a MemMachine deployment, and
        neither is a measurement of the world."""
        doc = " ".join(ps.__doc__.split())
        self.assertIn("one regional deployment", doc)
        self.assertIn("43 developers using coding agents", doc)

    def test_the_docstring_names_every_reported_only_figure(self):
        doc = " ".join(ps.__doc__.split())
        self.assertIn("Heavy human chat session: 0.06 operations per second",
                      doc)
        self.assertIn("Sustained automated client: 0.07 operations per second",
                      doc)
        self.assertIn("Peak against average: multiply an average load by 4. "
                      "REPORTED ONLY", doc)
        self.assertEqual(doc.count("REPORTED ONLY"), 3)

    def test_the_callers_table_names_both_sources_in_its_note(self):
        note = report_section(ps.size_deployment(100.0),
                              "Callers this capacity holds")["note"]
        for phrase in ("BurstGPT", "https://arxiv.org/abs/2401.17644",
                       "TraceLab", "https://arxiv.org/abs/2606.30560",
                       "55,295", "4,300", "43 developers",
                       "one regional deployment"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, note)

    def test_the_callers_table_reports_a_heavy_session_and_a_sustained_one(self):
        """Two new rows, and the heavy one is named as what a heavy session
        demands rather than as something the order was built on."""
        rows = report_section(ps.size_deployment(100.0),
                              "Callers this capacity holds")["rows"]
        self.assertEqual(len(rows), 4)
        heavy = next(r for r in rows if "heavy" in r[0])
        sustained = next(r for r in rows if "sustained rate" in r[0])
        self.assertIn("0.06 ops/s each", heavy[0])
        self.assertIn("busiest one in a hundred", heavy[0])
        self.assertEqual(heavy[1], "1,667")          # 100 / 0.06
        self.assertIn("99th percentile", heavy[2])
        self.assertIn("no machine count uses it", heavy[2])
        self.assertIn("0.07 ops/s each", sustained[0])
        self.assertEqual(sustained[1], "1,429")      # 100 / 0.07
        self.assertIn("mean step", sustained[2])
        self.assertIn("no machine count uses it", sustained[2])

    def test_the_report_explains_the_six_times_gap_in_words(self):
        note = report_section(ps.size_deployment(100.0),
                              "Callers this capacity holds")["note"]
        self.assertIn("differ by about 6 times", note)
        self.assertIn("idle most of the wall-clock time", note)
        self.assertIn("under-provisioned", note)
        self.assertIn("over-provisioned", note)

    def test_the_population_report_carries_the_same_two_extra_rows(self):
        pop = ps.ops_for_population(humans=2500, automated=75)
        section = population_section(pop, "Demand from this population")
        rows = {row[0]: row for row in section["rows"]}
        heavy = rows["Headroom check: every human session a heavy one"]
        self.assertEqual(heavy[2], "0.06 ops/s each")
        self.assertEqual(heavy[3], "150.00 ops/s")   # 2500 x 0.06
        sustained = rows["Average-load check: automated clients over a whole "
                         "session"]
        self.assertEqual(sustained[2], "0.07 ops/s each")
        self.assertEqual(sustained[3], "5.25 ops/s")  # 75 x 0.07
        self.assertIn("BurstGPT", section["note"])
        self.assertIn("TraceLab", section["note"])

    def test_the_two_extra_rows_are_not_added_into_the_total(self):
        """They are checks, not demand. The total is the two sized rates."""
        pop = ps.ops_for_population(humans=2500, automated=75)
        self.assertAlmostEqual(pop["ops_per_s_low"], 27.5 + 30.0)
        self.assertAlmostEqual(pop["ops_per_s_high"], 70.0 + 30.0)
        self.assertAlmostEqual(pop["human_ops_per_s_heavy"], 150.0)
        self.assertAlmostEqual(pop["automated_ops_per_s_sustained"], 5.25)

    def test_the_readme_input_table_names_both_new_constants_and_both_urls(self):
        text = self.source("README.md")
        table = text[text.index("## Every input, and what it is set to"):
                     text.index("## The measured anchor")]
        for phrase in ("`HUMAN_SESSION_OPS_PER_S_HEAVY`",
                       "`AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED`",
                       "https://arxiv.org/abs/2401.17644",
                       "https://arxiv.org/abs/2606.30560",
                       "BurstGPT", "TraceLab",
                       "reported only, no machine count uses it"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, table)

    def test_the_readme_has_a_section_naming_both_sources_in_full(self):
        text = self.source("README.md")
        section = text[text.index(
            "### Where the two per-caller rates come from"):
            text.index("## Tests")]
        # Markdown wraps its lines, so compare against the section with its
        # line breaks flattened: a citation split over two lines is still the
        # citation.
        section = " ".join(section.split())
        for phrase in (("BurstGPT: A Real-world Workload Dataset to Optimize "
                        "LLM Serving Systems"),
                       "2025", "https://arxiv.org/abs/2401.17644",
                       "https://github.com/HPMLL/BurstGPT",
                       "TraceLab", "2026",
                       "https://arxiv.org/abs/2606.30560",
                       "55,295", "176,466", "4,300", "43 developers",
                       "92.3%",
                       "one regional deployment",
                       "differ by about six times",
                       "under-provisioned", "over-provisioned"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, section)

    def test_the_readme_no_longer_calls_the_two_rates_never_measured(self):
        text = self.source("README.md")
        for constant in ("HUMAN_SESSION_OPS_PER_S_LOW",
                         "HUMAN_SESSION_OPS_PER_S_HIGH",
                         "AUTOMATED_CLIENT_OPS_PER_S"):
            row = next(line for line in text.splitlines()
                       if f"`{constant}`" in line)
            with self.subTest(constant=constant):
                self.assertNotIn("never measured", row)
                self.assertIn("estimate", row)

    def test_the_web_page_carries_the_citations_too(self):
        _, result, _, _ = ps.result_from_query({"ops": ["100"]})
        html = ps.render_html({"ops": ["100"]}, result, None, None)
        self.assertIn("BurstGPT", html)
        self.assertIn("TraceLab", html)
        self.assertIn("arxiv.org/abs/2401.17644", html)
        self.assertIn("arxiv.org/abs/2606.30560", html)

    def test_validate_exports_both_new_constants(self):
        keys = dict(ps.constant_numbers())
        self.assertEqual(keys["constants.human_session_ops_per_s_heavy"], 0.06)
        self.assertEqual(
            keys["constants.automated_client_ops_per_s_sustained"], 0.07)


class TestTheTwoReportedRatesEnterNoMachineCount(unittest.TestCase):
    """The point of the exercise: these two figures are printed and nothing
    else. Change either one to any value at all and every machine count, every
    demand figure and every storage figure must come out identical."""

    # Every machine, demand and storage number the model publishes, so that a
    # figure quietly reading one of the two new constants would be caught here
    # and not only in the machine counts.
    def snapshot(self):
        out = {}
        for name in ps.TIER_ORDER:
            r = ps.size_deployment(ps.TIER_OPS_PER_S[name], run_name=name)
            for key, value in ps.published_numbers(name, r):
                field = key.split(".", 1)[1]
                if field in ("human_sessions_heavy",
                             "automated_client_sessions_sustained"):
                    continue
                out[key] = value
        pop = ps.ops_for_population(humans=2500, automated=75,
                                    human_mix=ps.TrafficMix(90.0, 9.0, 1.0),
                                    automated_mix=ps.TrafficMix(10.0, 20.0,
                                                                70.0))
        for key, value in pop["sizing"]["machines"].items():
            out["population." + key] = value
        for key in ("ops_per_s_low", "ops_per_s_high", "human_ops_per_s_low",
                    "human_ops_per_s_high", "automated_ops_per_s",
                    "tier_for_low", "tier_for_high"):
            out["population." + key] = pop[key]
        out["population.blended_mix"] = tuple(sorted(
            pop["blended_mix"].items()))
        return out

    def test_changing_either_reported_rate_moves_nothing(self):
        before = self.snapshot()
        originals = (ps.HUMAN_SESSION_OPS_PER_S_HEAVY,
                     ps.AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED)
        try:
            for heavy, sustained in ((0.5, 3.0), (0.0001, 0.0001), (9.0, 9.0)):
                with self.subTest(heavy=heavy, sustained=sustained):
                    ps.HUMAN_SESSION_OPS_PER_S_HEAVY = heavy
                    ps.AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED = sustained
                    self.assertEqual(self.snapshot(), before)
        finally:
            (ps.HUMAN_SESSION_OPS_PER_S_HEAVY,
             ps.AUTOMATED_CLIENT_OPS_PER_S_SUSTAINED) = originals
        self.assertEqual(self.snapshot(), before)

    def test_the_two_sized_rates_still_do_move_the_answer(self):
        """The other direction, so the test above cannot pass by looking at
        nothing: the rates that ARE sized on change the population's demand."""
        pop = ps.ops_for_population(humans=2500, automated=75)
        original = ps.AUTOMATED_CLIENT_OPS_PER_S
        try:
            ps.AUTOMATED_CLIENT_OPS_PER_S = 0.8
            moved = ps.ops_for_population(humans=2500, automated=75)
        finally:
            ps.AUTOMATED_CLIENT_OPS_PER_S = original
        self.assertNotEqual(moved["ops_per_s_high"], pop["ops_per_s_high"])

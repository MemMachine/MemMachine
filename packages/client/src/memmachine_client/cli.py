"""CLI entry point for the memmachine command.

Provides `search` and `ingest` subcommands wired via [project.scripts].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import requests

from memmachine_client.client import MemMachineClient

REQUIRED_ENV_VARS = ("MEMMACHINE_URL", "MEMMACHINE_ORG_ID", "MEMMACHINE_PROJECT_ID")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with search and ingest subcommands."""
    parser = argparse.ArgumentParser(
        prog="memmachine",
        description="MemMachine CLI — search and ingest episodic memory",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search episodic memory",
    )
    search_parser.add_argument(
        "query",
        help="Query string to search for",
    )
    search_parser.add_argument(
        "--json",
        dest="json",
        action="store_true",
        default=False,
        help="Output results as JSON",
    )
    search_parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)",
    )

    # ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest an episode into episodic memory",
    )
    ingest_parser.add_argument(
        "--content",
        required=True,
        help="Content to ingest",
    )
    ingest_parser.add_argument(
        "--role",
        default="user",
        choices=["user", "assistant", "system"],
        help="Role of the content producer (default: user)",
    )
    ingest_parser.add_argument(
        "--session-id",
        dest="session_id",
        default=None,
        help="Session ID to associate with the episode",
    )
    ingest_parser.add_argument(
        "--producer-id",
        dest="producer_id",
        default=None,
        help="Producer ID to associate with the episode",
    )
    ingest_parser.add_argument(
        "--timestamp",
        dest="timestamp",
        default=None,
        help="ISO 8601 timestamp for the episode (default: now)",
    )

    return parser


def _get_client() -> tuple[MemMachineClient, str, str]:
    """Read env vars and return (client, org_id, project_id).

    Exits with code 1 and prints to stderr if any required env var is missing.
    """
    url = os.environ.get("MEMMACHINE_URL")
    if not url:
        sys.stderr.write("Error: MEMMACHINE_URL is not set\n")
        sys.exit(1)

    org_id = os.environ.get("MEMMACHINE_ORG_ID")
    if not org_id:
        sys.stderr.write("Error: MEMMACHINE_ORG_ID is not set\n")
        sys.exit(1)

    project_id = os.environ.get("MEMMACHINE_PROJECT_ID")
    if not project_id:
        sys.stderr.write("Error: MEMMACHINE_PROJECT_ID is not set\n")
        sys.exit(1)

    api_key = os.environ.get("MEMMACHINE_API_KEY") or None

    client = MemMachineClient(
        base_url=url,
        api_key=api_key,
    )
    return client, org_id, project_id


def cmd_search(args: argparse.Namespace) -> None:
    """Execute the search subcommand."""
    client, org_id, project_id = _get_client()
    try:
        project = client.get_or_create_project(org_id, project_id)
        memory = project.memory()
        results = memory.search(args.query, limit=args.limit)

        # Collect episodes from both short-term and long-term memory
        episodic_memory = results.content.episodic_memory
        if episodic_memory is None:
            episodes = []
        else:
            episodes = [
                *episodic_memory.short_term_memory.episodes,
                *episodic_memory.long_term_memory.episodes,
            ]

        if args.json:
            output = [{"score": e.score, "content": e.content} for e in episodes]
            sys.stdout.write(json.dumps(output) + "\n")
        else:
            for e in episodes:
                sys.stdout.write(f"{e.score:.3f}  {e.content}\n")
    except requests.RequestException as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Execute the ingest subcommand."""
    client, org_id, project_id = _get_client()
    try:
        metadata: dict[str, str] | None = None
        if args.session_id:
            metadata = {"session_id": args.session_id}

        project = client.get_or_create_project(org_id, project_id)
        memory = project.memory(metadata=metadata)

        ts = datetime.fromisoformat(args.timestamp) if args.timestamp else None

        memory.add(
            args.content,
            role=args.role,
            producer=args.producer_id,
            timestamp=ts,
        )
        sys.stderr.write("Ingested episode\n")
    except requests.RequestException as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


def main() -> None:
    """Entry point for the memmachine CLI."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.subcommand == "search":
        cmd_search(args)
    elif args.subcommand == "ingest":
        cmd_ingest(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

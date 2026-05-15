"""Command line interface for the MemMachine Python client."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, JsonValue

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memmachine_common.api import EpisodeType, MemoryType

from memmachine_client.client import MemMachineClient
from memmachine_client.project import Project

ENV_API_KEY = "MEMMACHINE_API_KEY"
ENV_BASE_URL = "MEMORY_BACKEND_URL"
ENV_MAX_RETRIES = "MEMMACHINE_MAX_RETRIES"
ENV_ORG_ID = "MEMMACHINE_ORG_ID"
ENV_PROJECT_ID = "MEMMACHINE_PROJECT_ID"
ENV_TIMEOUT = "MEMMACHINE_TIMEOUT"
DEFAULT_PROG = "mem-cli"


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value in {None, ""}:
        return default
    try:
        return int(raw_value)
    except ValueError as err:
        raise SystemExit(f"{name} must be an integer") from err


def _die(message: str, *, prog: str = DEFAULT_PROG) -> None:
    sys.stderr.write(f"{prog}: error: {message}\n")
    raise SystemExit(2)


def _to_jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel | Enum):
        return _to_jsonable(value)
    return str(value)


def print_json(value: object) -> None:
    """Print an object as stable, readable JSON."""
    value = _to_jsonable(value)
    sys.stdout.write(
        f"{json.dumps(value, default=_json_default, indent=2, sort_keys=True)}\n"
    )


def parse_json_object(raw_value: str, *, option_name: str) -> dict[str, JsonValue]:
    """Parse a JSON object argument."""
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as err:
        raise argparse.ArgumentTypeError(f"{option_name} must be valid JSON") from err
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"{option_name} must be a JSON object")
    return cast(dict[str, JsonValue], value)


def parse_json_string_object(raw_value: str, *, option_name: str) -> dict[str, str]:
    """Parse a JSON object whose keys and values are strings."""
    value = parse_json_object(raw_value, option_name=option_name)
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise argparse.ArgumentTypeError(
                f"{option_name} must be a JSON object with string values"
            )
    return cast(dict[str, str], value)


def parse_key_value_items(items: list[str] | None) -> dict[str, str]:
    """Parse repeated KEY=VALUE arguments into a dictionary."""
    parsed: dict[str, str] = {}
    for item in items or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"metadata values must use KEY=VALUE format: {item}")
        parsed[key] = value
    return parsed


def parse_memory_type(value: str) -> MemoryType:
    """Parse a memory type by enum value or name."""
    normalized = value.lower()
    for memory_type in MemoryType:
        if (
            memory_type.value.lower() == normalized
            or memory_type.name.lower() == normalized
        ):
            return memory_type
    choices = ", ".join(memory_type.value for memory_type in MemoryType)
    raise argparse.ArgumentTypeError(f"memory type must be one of: {choices}")


def parse_episode_type(value: str) -> EpisodeType:
    """Parse an episode type by enum value or name."""
    normalized = value.lower()
    for episode_type in EpisodeType:
        if (
            episode_type.value.lower() == normalized
            or episode_type.name.lower() == normalized
        ):
            return episode_type
    choices = ", ".join(episode_type.value for episode_type in EpisodeType)
    raise argparse.ArgumentTypeError(f"episode type must be one of: {choices}")


def add_project_context_args(parser: argparse.ArgumentParser) -> None:
    """Add project context arguments to a parser."""
    parser.add_argument("--org-id", help=f"Organization id. Defaults to ${ENV_ORG_ID}.")
    parser.add_argument(
        "--project-id",
        help=f"Project id. Defaults to ${ENV_PROJECT_ID}.",
    )


def get_project_context(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve org and project ids from args or environment variables."""
    org_id = args.org_id or os.environ.get(ENV_ORG_ID)
    project_id = args.project_id or os.environ.get(ENV_PROJECT_ID)
    if not org_id:
        _die(f"--org-id or {ENV_ORG_ID} is required", prog=args.prog)
    if not project_id:
        _die(f"--project-id or {ENV_PROJECT_ID} is required", prog=args.prog)
    return cast(str, org_id), cast(str, project_id)


def build_client(args: argparse.Namespace) -> MemMachineClient:
    """Create a MemMachineClient from command arguments and environment."""
    base_url = args.base_url or os.environ.get(ENV_BASE_URL)
    if not base_url:
        _die(f"--base-url or {ENV_BASE_URL} is required", prog=args.prog)

    return MemMachineClient(
        api_key=args.api_key or os.environ.get(ENV_API_KEY),
        base_url=base_url,
        timeout=args.timeout if args.timeout is not None else _env_int(ENV_TIMEOUT, 30),
        max_retries=args.max_retries
        if args.max_retries is not None
        else _env_int(ENV_MAX_RETRIES, 3),
    )


def _project(client: MemMachineClient, args: argparse.Namespace) -> Project:
    org_id, project_id = get_project_context(args)
    if getattr(args, "create", False):
        return client.get_or_create_project(
            org_id=org_id,
            project_id=project_id,
            timeout=args.request_timeout,
        )
    return client.get_project(
        org_id=org_id,
        project_id=project_id,
        timeout=args.request_timeout,
    )


def run_command(client: MemMachineClient, args: argparse.Namespace) -> int:
    """Run the parsed CLI command."""
    if args.command == "health":
        print_json(client.health_check(timeout=args.request_timeout))
        return 0

    if args.command == "metrics":
        sys.stdout.write(client.get_metrics(timeout=args.request_timeout))
        return 0

    if args.command == "config" and args.config_command == "resources":
        print_json(client.config().get_resources(timeout=args.request_timeout))
        return 0

    if args.command == "projects":
        return run_projects_command(client, args)

    if args.command == "memory":
        return run_memory_command(client, args)

    _die("a command is required", prog=args.prog)
    return 2


def run_projects_command(client: MemMachineClient, args: argparse.Namespace) -> int:
    """Run project subcommands."""
    if args.projects_command == "list":
        projects = [
            {
                "org_id": project.org_id,
                "project_id": project.project_id,
                "description": project.description,
            }
            for project in client.list_projects(timeout=args.request_timeout)
        ]
        print_json(projects)
        return 0

    if args.projects_command == "create":
        org_id, project_id = get_project_context(args)
        project = client.create_project(
            org_id=org_id,
            project_id=project_id,
            description=args.description,
            embedder=args.embedder,
            reranker=args.reranker,
            timeout=args.request_timeout,
        )
        print_json(_project_to_dict(project))
        return 0

    project = _project(client, args)
    if args.projects_command in {"get", "get-or-create"}:
        print_json(_project_to_dict(project))
        return 0

    if args.projects_command == "delete":
        print_json({"deleted": project.delete(timeout=args.request_timeout)})
        return 0

    if args.projects_command == "episode-count":
        print_json({"count": project.get_episode_count(timeout=args.request_timeout)})
        return 0

    _die("a projects subcommand is required", prog=args.prog)
    return 2


def run_memory_command(client: MemMachineClient, args: argparse.Namespace) -> int:
    """Run memory subcommands."""
    project = _project(client, args)
    metadata = parse_key_value_items(args.metadata)
    memory = project.memory(metadata=metadata)

    if args.memory_command == "add":
        result = memory.add(
            args.content,
            role=args.role,
            producer=args.producer,
            produced_for=args.produced_for,
            metadata=parse_json_string_object(
                args.extra_metadata, option_name="--extra-metadata"
            )
            if args.extra_metadata
            else None,
            timeout=args.request_timeout,
        )
        print_json(result)
        return 0

    if args.memory_command == "search":
        result = memory.search(
            args.query,
            limit=args.limit,
            expand_context=args.expand_context,
            score_threshold=args.score_threshold,
            filter_dict=parse_key_value_items(args.filter),
            set_metadata=parse_json_object(
                args.set_metadata, option_name="--set-metadata"
            )
            if args.set_metadata
            else None,
            agent_mode=args.agent_mode,
            timeout=args.request_timeout,
        )
        print_json(result)
        return 0

    if args.memory_command == "list":
        result = memory.list(
            memory_type=args.type,
            page_size=args.page_size,
            page_num=args.page_num,
            filter_dict=parse_key_value_items(args.filter),
            set_metadata=parse_json_object(
                args.set_metadata, option_name="--set-metadata"
            )
            if args.set_metadata
            else None,
            timeout=args.request_timeout,
        )
        print_json(result)
        return 0

    if args.memory_command == "delete-episodic":
        print_json(
            {
                "deleted": memory.delete_episodic(
                    episodic_id=args.id,
                    episodic_ids=args.ids,
                    timeout=args.request_timeout,
                )
            }
        )
        return 0

    if args.memory_command == "delete-semantic":
        print_json(
            {
                "deleted": memory.delete_semantic(
                    semantic_id=args.id,
                    semantic_ids=args.ids,
                    timeout=args.request_timeout,
                )
            }
        )
        return 0

    _die("a memory subcommand is required", prog=args.prog)
    return 2


def _project_to_dict(project: Project) -> dict[str, object]:
    return {
        "org_id": project.org_id,
        "project_id": project.project_id,
        "description": project.description,
        "config": project.config,
    }


def add_request_timeout_arg(parser: argparse.ArgumentParser) -> None:
    """Add per-request timeout override."""
    parser.add_argument(
        "--request-timeout",
        type=int,
        help="Override the client timeout for this request.",
    )


def build_parser(prog: str = DEFAULT_PROG) -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Command line client for the MemMachine HTTP API.",
    )
    parser.set_defaults(prog=parser.prog)
    parser.add_argument("--base-url", help=f"Server URL. Defaults to ${ENV_BASE_URL}.")
    parser.add_argument("--api-key", help=f"API key. Defaults to ${ENV_API_KEY}.")
    parser.add_argument(
        "--timeout", type=int, help=f"Client timeout. Defaults to ${ENV_TIMEOUT}."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help=f"Client retry count. Defaults to ${ENV_MAX_RETRIES}.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health", help="Check server health.")
    add_request_timeout_arg(health)

    metrics = subparsers.add_parser("metrics", help="Print Prometheus metrics.")
    add_request_timeout_arg(metrics)

    config = subparsers.add_parser("config", help="Inspect server configuration.")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)
    resources = config_subparsers.add_parser(
        "resources", help="List configured resources."
    )
    add_request_timeout_arg(resources)

    projects = subparsers.add_parser("projects", help="Manage projects.")
    project_subparsers = projects.add_subparsers(dest="projects_command", required=True)

    projects_list = project_subparsers.add_parser("list", help="List projects.")
    add_request_timeout_arg(projects_list)

    projects_create = project_subparsers.add_parser("create", help="Create a project.")
    add_project_context_args(projects_create)
    projects_create.add_argument("--description", default="")
    projects_create.add_argument("--embedder", default="")
    projects_create.add_argument("--reranker", default="")
    add_request_timeout_arg(projects_create)

    for name in ("get", "get-or-create", "delete", "episode-count"):
        project_parser = project_subparsers.add_parser(name, help=f"{name} a project.")
        add_project_context_args(project_parser)
        project_parser.set_defaults(create=name == "get-or-create")
        add_request_timeout_arg(project_parser)

    memory = subparsers.add_parser("memory", help="Manage memories.")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)

    memory_add = memory_subparsers.add_parser("add", help="Add a memory.")
    add_project_context_args(memory_add)
    memory_add.add_argument("content")
    memory_add.add_argument("--role", default="")
    memory_add.add_argument("--producer")
    memory_add.add_argument("--produced-for")
    memory_add.add_argument("--metadata", action="append", default=[])
    memory_add.add_argument("--extra-metadata")
    memory_add.add_argument(
        "--create", action="store_true", help="Create the project if missing."
    )
    add_request_timeout_arg(memory_add)

    memory_search = memory_subparsers.add_parser("search", help="Search memories.")
    add_project_context_args(memory_search)
    memory_search.add_argument("query")
    memory_search.add_argument("--metadata", action="append", default=[])
    memory_search.add_argument("--filter", action="append", default=[])
    memory_search.add_argument("--set-metadata")
    memory_search.add_argument("--limit", type=int)
    memory_search.add_argument("--expand-context", type=int, default=0)
    memory_search.add_argument("--score-threshold", type=float)
    memory_search.add_argument("--agent-mode", action="store_true")
    memory_search.add_argument(
        "--create", action="store_true", help="Create the project if missing."
    )
    add_request_timeout_arg(memory_search)

    memory_list = memory_subparsers.add_parser("list", help="List memories.")
    add_project_context_args(memory_list)
    memory_list.add_argument("--metadata", action="append", default=[])
    memory_list.add_argument("--filter", action="append", default=[])
    memory_list.add_argument("--set-metadata")
    memory_list.add_argument(
        "--type", type=parse_memory_type, default=MemoryType.Episodic
    )
    memory_list.add_argument("--page-size", type=int, default=100)
    memory_list.add_argument("--page-num", type=int, default=0)
    memory_list.add_argument(
        "--create", action="store_true", help="Create the project if missing."
    )
    add_request_timeout_arg(memory_list)

    for name in ("delete-episodic", "delete-semantic"):
        delete_parser = memory_subparsers.add_parser(name, help=f"{name} memory ids.")
        add_project_context_args(delete_parser)
        delete_parser.add_argument("--metadata", action="append", default=[])
        delete_parser.add_argument("--id", default="")
        delete_parser.add_argument("--ids", action="append", default=[])
        delete_parser.add_argument(
            "--create", action="store_true", help="Create the project if missing."
        )
        add_request_timeout_arg(delete_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser(prog=Path(sys.argv[0]).name if argv is None else DEFAULT_PROG)
    args = parser.parse_args(argv)
    client = build_client(args)
    try:
        return run_command(client, args)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

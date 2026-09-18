"""Point coding agents at a MemMachine server's memory tools.

Claude Code and Codex both speak MCP over streamable HTTP with static
headers, so one endpoint (`<server>/v1/mcp`) and one header
(`X-MemMachine-Tenant`) serve both, and the tools (`memory_query`,
`memory_expand`) live on the server. This module writes and removes that
entry in each agent's own configuration.

Capture is the other half: both agents run a `Stop` hook at the end of a
turn, with the same command line and the same handler shape, so this
module writes that hook into the file each agent reads its hooks from.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_NAMES = ("claude-code", "codex")
SCOPE_NAMES = ("user", "project")

MCP_SERVER_NAME = "memmachine"
MCP_ENDPOINT_PATH = "/v1/mcp"
TENANT_HEADER_NAME = "X-MemMachine-Tenant"

CLAUDE_EXECUTABLE_NAME = "claude"
CLAUDE_PROJECT_CONFIG_NAME = ".mcp.json"
CLAUDE_DIRECTORY_NAME = ".claude"
CLAUDE_SETTINGS_NAME = "settings.json"
CODEX_HOME_VARIABLE = "CODEX_HOME"
CODEX_DIRECTORY_NAME = ".codex"
CODEX_CONFIG_NAME = "config.toml"
CODEX_HOOKS_NAME = "hooks.json"
BACKUP_SUFFIX = ".bak"

HOOK_EVENT_NAME = "Stop"
# What the hook runs, and what a handler this installer wrote is
# recognized by, so a second install replaces it rather than adding one.
HOOK_COMMAND_MARKER = "memmachine_client.cli agent capture"
# Seconds a hook may take. Capture posts one batch at a time under a
# budget of its own, and this bounds what the agent waits for if the
# machine it runs on stalls.
HOOK_TIMEOUT_SECONDS = 10

CODEX_TABLE_NAMES = (
    ("mcp_servers", MCP_SERVER_NAME),
    ("mcp_servers", MCP_SERVER_NAME, "http_headers"),
)


class CodingAgentError(Exception):
    """Raised when an agent's configuration cannot be read or edited."""


def run_agent_command(args: argparse.Namespace) -> int:
    """Run an `agent` subcommand and return the process exit code."""
    try:
        if args.agent_command == "install":
            server = validated_server(args.server)
            tenant = validated_tenant(args.tenant)
            if args.agent_name == "claude-code":
                return install_claude_code(
                    server=server,
                    tenant=tenant,
                    scope=args.scope,
                    dry_run=args.dry_run,
                )
            return install_codex(
                server=server,
                tenant=tenant,
                scope=args.scope,
                dry_run=args.dry_run,
            )
        if args.agent_name == "claude-code":
            return disable_claude_code(scope=args.scope)
        return disable_codex(scope=args.scope)
    except CodingAgentError as error:
        sys.stderr.write(f"{args.prog}: error: {error}\n")
        return 1


def install_claude_code(*, server: str, tenant: str, scope: str, dry_run: bool) -> int:
    """Register the MemMachine MCP server and capture hook with Claude Code.

    The MCP entry of user scope is delegated to the `claude` executable,
    which owns that file; project scope writes `.mcp.json` in the
    current directory. The `Stop` hook goes into the scope's own
    settings file either way.
    """
    url = _mcp_endpoint_url(server)
    if scope == "project":
        entry: dict[str, object] = {
            "type": "http",
            "url": url,
            "headers": {TENANT_HEADER_NAME: tenant},
        }
        mcp_result = _edit_claude_project_config(entry, dry_run=dry_run)
    else:
        # `claude mcp add` refuses a name it already holds, so a
        # reinstall removes the earlier entry first and leaves exactly
        # one behind.
        mcp_result = _run_claude_executable(
            [
                "mcp",
                "add",
                "--transport",
                "http",
                "--scope",
                "user",
                MCP_SERVER_NAME,
                url,
                "--header",
                f"{TENANT_HEADER_NAME}: {tenant}",
            ],
            preceded_by=["mcp", "remove", "--scope", "user", MCP_SERVER_NAME],
            dry_run=dry_run,
        )
    hook_result = _edit_stop_hook(
        _claude_settings_path(scope),
        _capture_command("claude-code", server=server, tenant=tenant),
        dry_run=dry_run,
    )
    return mcp_result or hook_result


def disable_claude_code(*, scope: str) -> int:
    """Remove the MemMachine MCP server entry and capture hook from Claude Code."""
    if scope == "project":
        mcp_result = _edit_claude_project_config(None, dry_run=False)
    else:
        mcp_result = _run_claude_executable(
            ["mcp", "remove", "--scope", "user", MCP_SERVER_NAME],
            preceded_by=None,
            dry_run=False,
        )
    hook_result = _edit_stop_hook(_claude_settings_path(scope), None, dry_run=False)
    return mcp_result or hook_result


def _claude_settings_path(scope: str) -> Path:
    """The settings file a scope's Claude Code hooks live in.

    User scope is `~/.claude/settings.json`, and project scope is the
    `.claude` directory of the current directory.
    """
    if scope == "project":
        return Path.cwd() / CLAUDE_DIRECTORY_NAME / CLAUDE_SETTINGS_NAME
    return Path.home() / CLAUDE_DIRECTORY_NAME / CLAUDE_SETTINGS_NAME


def _run_claude_executable(
    arguments: list[str],
    *,
    preceded_by: list[str] | None,
    dry_run: bool,
) -> int:
    """Run `claude` with the given arguments, reporting the command run.

    `preceded_by` runs first and its exit code is ignored, which is what
    makes a reinstall idempotent. A dry run reports both commands and
    needs no executable; a run without a `claude` executable on PATH
    fails with an error naming the command to run by hand.
    """
    command_line = shlex.join([CLAUDE_EXECUTABLE_NAME, *arguments])
    if dry_run:
        if preceded_by is not None:
            preceding_line = shlex.join([CLAUDE_EXECUTABLE_NAME, *preceded_by])
            sys.stdout.write(f"would run: {preceding_line}\n")
        sys.stdout.write(f"would run: {command_line}\n")
        return 0
    executable = shutil.which(CLAUDE_EXECUTABLE_NAME)
    if executable is None:
        raise CodingAgentError(
            f"no {CLAUDE_EXECUTABLE_NAME} executable on PATH. Run this where "
            f"Claude Code is installed:\n  {command_line}"
        )
    if preceded_by is not None:
        subprocess.run(
            [executable, *preceded_by],
            capture_output=True,
            check=False,
            text=True,
        )
    result = subprocess.run(
        [executable, *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    sys.stdout.write(f"ran: {command_line}\n")
    return 0


def _edit_claude_project_config(
    entry: dict[str, object] | None,
    *,
    dry_run: bool,
) -> int:
    """Set or remove the MemMachine server in `.mcp.json` in this directory.

    Every other server and every other key of the file is carried over.
    """
    path = Path.cwd() / CLAUDE_PROJECT_CONFIG_NAME
    original_text = path.read_text(encoding="utf-8") if path.exists() else ""
    document = _loaded_json_object(original_text, path)

    servers = document.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise CodingAgentError(f'{path} has an "mcpServers" key that is not an object')
    servers = dict(servers)
    if entry is None:
        servers.pop(MCP_SERVER_NAME, None)
    else:
        servers[MCP_SERVER_NAME] = entry
    document["mcpServers"] = servers

    new_text = f"{json.dumps(document, indent=2)}\n"
    return _apply_config_change(path, original_text, new_text, dry_run=dry_run)


def install_codex(*, server: str, tenant: str, scope: str, dry_run: bool) -> int:
    """Write the MemMachine MCP server tables into Codex's `config.toml`."""
    path = _codex_config_path(scope)
    url = _mcp_endpoint_url(server)
    original_text = path.read_text(encoding="utf-8") if path.exists() else ""
    document = _loaded_toml(original_text, path)
    spans = _memmachine_table_spans(original_text.splitlines(keepends=True))
    if not spans and _codex_entry(document) is not None:
        raise CodingAgentError(
            f"{path} already defines mcp_servers.{MCP_SERVER_NAME} in a form this "
            "installer does not edit (an inline table or a dotted key). "
            "Remove it by hand and run this again."
        )

    # TOML basic strings take JSON's escapes, and JSON escapes every
    # character that a basic string may not carry raw.
    tables = (
        f"[mcp_servers.{MCP_SERVER_NAME}]\n"
        f"url = {json.dumps(url)}\n"
        "\n"
        f"[mcp_servers.{MCP_SERVER_NAME}.http_headers]\n"
        f"{TENANT_HEADER_NAME} = {json.dumps(tenant)}\n"
    )
    new_text = _codex_text_with_tables(original_text, spans, tables)
    written = _codex_entry(_loaded_toml(new_text, path))
    if written != {"url": url, "http_headers": {TENANT_HEADER_NAME: tenant}}:
        raise CodingAgentError(
            f"the edit of {path} would not have produced the intended "
            f"mcp_servers.{MCP_SERVER_NAME} table. The file is unchanged."
        )
    mcp_result = _apply_config_change(path, original_text, new_text, dry_run=dry_run)
    hook_result = _edit_stop_hook(
        _codex_hooks_path(scope),
        _capture_command("codex", server=server, tenant=tenant),
        dry_run=dry_run,
    )
    return mcp_result or hook_result


def disable_codex(*, scope: str) -> int:
    """Remove the MemMachine MCP server tables from Codex's `config.toml`."""
    path = _codex_config_path(scope)
    if not path.exists():
        sys.stdout.write(f"{path} does not exist\n")
        return _edit_stop_hook(_codex_hooks_path(scope), None, dry_run=False)
    original_text = path.read_text(encoding="utf-8")
    document = _loaded_toml(original_text, path)
    spans = _memmachine_table_spans(original_text.splitlines(keepends=True))
    if not spans and _codex_entry(document) is not None:
        raise CodingAgentError(
            f"{path} defines mcp_servers.{MCP_SERVER_NAME} in a form this installer "
            "does not edit (an inline table or a dotted key). Remove it by hand."
        )
    new_text = _codex_text_without_tables(original_text, spans)
    if _codex_entry(_loaded_toml(new_text, path)) is not None:
        raise CodingAgentError(
            f"the edit of {path} would have left mcp_servers.{MCP_SERVER_NAME} "
            "behind. The file is unchanged."
        )
    mcp_result = _apply_config_change(path, original_text, new_text, dry_run=False)
    hook_result = _edit_stop_hook(_codex_hooks_path(scope), None, dry_run=False)
    return mcp_result or hook_result


def _codex_config_path(scope: str) -> Path:
    """Return the `config.toml` a scope owns.

    User scope is Codex's own directory; project scope is the `.codex`
    directory of the current directory.
    """
    if scope == "project":
        return Path.cwd() / CODEX_DIRECTORY_NAME / CODEX_CONFIG_NAME
    return codex_home_directory() / CODEX_CONFIG_NAME


def _codex_hooks_path(scope: str) -> Path:
    """Return the `hooks.json` a scope owns.

    Codex reads hooks from `hooks.json` beside the configuration it
    reads the MCP servers from, so the two scopes are the same two
    directories.
    """
    if scope == "project":
        return Path.cwd() / CODEX_DIRECTORY_NAME / CODEX_HOOKS_NAME
    return codex_home_directory() / CODEX_HOOKS_NAME


def codex_home_directory() -> Path:
    """Codex's own directory, `$CODEX_HOME` where it is set and `~/.codex` where it is not."""
    codex_home = os.environ.get(CODEX_HOME_VARIABLE)
    return Path(codex_home) if codex_home else Path.home() / CODEX_DIRECTORY_NAME


def _codex_text_with_tables(
    original_text: str,
    spans: list[tuple[int, int]],
    tables: str,
) -> str:
    """Return the file's text with our tables in it, once.

    The first table this installer owns is replaced where it stands, any
    later one is dropped, and a file without one gets the tables appended
    after a blank line. Every line outside those spans is carried over
    unchanged.
    """
    if not spans:
        if not original_text:
            return tables
        text = original_text if original_text.endswith("\n") else f"{original_text}\n"
        if not text.endswith("\n\n"):
            text += "\n"
        return text + tables
    lines = original_text.splitlines(keepends=True)
    for start, end in reversed(spans[1:]):
        del lines[start:end]
    first_start, first_end = spans[0]
    # A span ends where the next table's header begins, so tables that
    # something follows keep the blank line that separates them from it.
    separated = tables if first_end >= len(lines) else f"{tables}\n"
    lines[first_start:first_end] = [separated]
    return "".join(lines)


def _codex_text_without_tables(
    original_text: str,
    spans: list[tuple[int, int]],
) -> str:
    """Return the file's text with the tables this installer owns removed.

    Every line outside those spans is carried over unchanged, except the
    blank line an append put before tables that run to the end of the
    file, which goes with them.
    """
    lines = original_text.splitlines(keepends=True)
    removed_to_end = bool(spans) and spans[-1][1] == len(lines)
    for start, end in reversed(spans):
        del lines[start:end]
    if removed_to_end and lines and not lines[-1].strip():
        # The blank line an append left in front of them goes too.
        del lines[-1]
    return "".join(lines)


def _memmachine_table_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Return the half-open line ranges of the tables this installer owns.

    A table runs from its header to the line before the next header, so a
    span carries the table's own comments and blank lines with it. Lines
    inside a multi-line string are not read as headers.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    string_delimiter: str | None = None
    for index, line in enumerate(lines):
        if string_delimiter is not None:
            if line.count(string_delimiter) % 2 == 1:
                string_delimiter = None
            continue
        stripped = line.lstrip()
        if stripped.startswith("["):
            if start is not None:
                spans.append((start, index))
                start = None
            if _table_header_parts(stripped) in CODEX_TABLE_NAMES:
                start = index
            continue
        for delimiter in ('"""', "'''"):
            if line.count(delimiter) % 2 == 1:
                string_delimiter = delimiter
                break
    if start is not None:
        spans.append((start, len(lines)))
    return spans


def _table_header_parts(line: str) -> tuple[str, ...]:
    """Return the dotted parts of a table header, or nothing for other lines.

    Quoted key parts are returned unquoted. An array-of-tables header is
    never one of ours, so it returns nothing.
    """
    text = line.strip()
    if not text.startswith("[") or text.startswith("[["):
        return ()
    parts: list[str] = []
    current = ""
    quote: str | None = None
    for character in text[1:]:
        if quote is not None:
            if character == quote:
                quote = None
            else:
                current += character
        elif character in {'"', "'"}:
            quote = character
        elif character == ".":
            parts.append(current.strip())
            current = ""
        elif character == "]":
            parts.append(current.strip())
            return tuple(parts)
        else:
            current += character
    return ()


def _codex_entry(document: dict[str, object]) -> object:
    """Return the `mcp_servers.memmachine` value of a parsed config, or None."""
    servers = document.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    return servers.get(MCP_SERVER_NAME)


def _edit_stop_hook(path: Path, command: str | None, *, dry_run: bool) -> int:
    """Set or remove the MemMachine `Stop` hook in an agent's hooks file.

    Claude Code and Codex read the same shape, a `hooks` object whose
    `Stop` key holds groups of handlers, so one edit serves both. Every
    other hook and every other key of the file is carried over, and a
    handler this installer wrote is replaced where it stands, so a
    second install leaves one behind. A file that already reads the way
    this edit would leave it is not written.
    """
    if command is None and not path.exists():
        sys.stdout.write(f"{path} does not exist\n")
        return 0
    original_text = path.read_text(encoding="utf-8") if path.exists() else ""
    document = _loaded_json_object(original_text, path)
    edited = _document_with_stop_hook(document, command, path)
    if edited == document:
        held = (
            "already holds this hook"
            if command is not None
            else "holds no hook of ours"
        )
        sys.stdout.write(f"{path} {held}\n")
        return 0
    new_text = f"{json.dumps(edited, indent=2)}\n"
    return _apply_config_change(path, original_text, new_text, dry_run=dry_run)


def _document_with_stop_hook(
    document: dict[str, object],
    command: str | None,
    path: Path,
) -> dict[str, object]:
    """The hooks file with our `Stop` handler set to `command`, or removed.

    A `Stop` key left holding nothing goes, and so does a `hooks` key
    left holding nothing, so disabling leaves the file as it was before
    the install.
    """
    hooks = document.get("hooks", {})
    if not isinstance(hooks, dict):
        raise CodingAgentError(f'{path} has a "hooks" key that is not an object')
    groups = hooks.get(HOOK_EVENT_NAME, [])
    if not isinstance(groups, list):
        raise CodingAgentError(
            f'{path} has a "hooks.{HOOK_EVENT_NAME}" key that is not a list'
        )
    edited_hooks = dict(hooks)
    edited_groups = _stop_hook_groups(groups, command)
    if edited_groups:
        edited_hooks[HOOK_EVENT_NAME] = edited_groups
    else:
        edited_hooks.pop(HOOK_EVENT_NAME, None)
    edited = dict(document)
    if edited_hooks:
        edited["hooks"] = edited_hooks
    else:
        edited.pop("hooks", None)
    return edited


def _stop_hook_groups(groups: list[object], command: str | None) -> list[object]:
    """The `Stop` groups with our handler set to `command`, or removed.

    Every group and every handler that is not ours is carried over in
    the order it was found, a group left holding no handler goes, and a
    file that held none of ours gets a group of its own.
    """
    edited: list[object] = []
    written = False
    for group in groups:
        handlers = group.get("hooks") if isinstance(group, dict) else None
        if not isinstance(group, dict) or not isinstance(handlers, list):
            edited.append(group)
            continue
        kept: list[object] = []
        for handler in handlers:
            if not _is_capture_handler(handler):
                kept.append(handler)
            elif command is not None and not written:
                kept.append(_capture_handler(command))
                written = True
        if kept:
            edited.append({**group, "hooks": kept})
    if command is not None and not written:
        edited.append({"hooks": [_capture_handler(command)]})
    return edited


def _is_capture_handler(handler: object) -> bool:
    """Whether a handler is one this installer wrote.

    A handler is ours if it runs the capture command, whatever server,
    tenant and interpreter it names, so an install written by an earlier
    version of this client is replaced rather than left beside its
    replacement.
    """
    if not isinstance(handler, dict):
        return False
    return HOOK_COMMAND_MARKER in str(handler.get("command", ""))


def _capture_handler(command: str) -> dict[str, object]:
    """The handler that runs capture when a turn ends."""
    return {"type": "command", "command": command, "timeout": HOOK_TIMEOUT_SECONDS}


def _capture_command(agent_name: str, *, server: str, tenant: str) -> str:
    """The command line a `Stop` hook runs to capture a session.

    Both agents run a handler's command through a shell, whose PATH need
    not hold the interpreter this client is installed in, so the command
    names that interpreter by its own path and reaches the client as a
    module of it.
    """
    return shlex.join(
        [
            sys.executable,
            "-m",
            "memmachine_client.cli",
            "agent",
            "capture",
            agent_name,
            "--server",
            server,
            "--tenant",
            tenant,
        ]
    )


def _loaded_json_object(text: str, path: Path) -> dict[str, object]:
    """Parse a JSON object, naming the file that does not hold one.

    A file that holds nothing yet holds no keys.
    """
    if not text.strip():
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as error:
        raise CodingAgentError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise CodingAgentError(f"{path} does not hold a JSON object")
    return loaded


def _loaded_toml(text: str, path: Path) -> dict[str, object]:
    """Parse TOML text, naming the file that failed to parse."""
    if sys.version_info < (3, 11):
        raise CodingAgentError(
            "reading a Codex configuration needs Python 3.11 or newer for "
            "tomllib; install this client on 3.11+ to configure Codex"
        )
    import tomllib

    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CodingAgentError(f"{path} is not valid TOML: {error}") from error


def _apply_config_change(
    path: Path,
    original_text: str,
    new_text: str,
    *,
    dry_run: bool,
) -> int:
    """Write a configuration file, keeping `<file>.bak` of what it replaced.

    A file that already holds the intended text is left alone, so a
    second install neither backs up nor writes.
    """
    if new_text == original_text:
        sys.stdout.write(f"{path} already holds this entry\n")
        return 0
    if dry_run:
        difference = "".join(
            difflib.unified_diff(
                original_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=str(path),
                tofile=str(path),
            )
        )
        if not difference.endswith("\n"):
            difference += "\n"
        sys.stdout.write(f"would write {path}\n{difference}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup_path = path.with_name(path.name + BACKUP_SUFFIX)
        backup_path.write_text(original_text, encoding="utf-8")
        sys.stdout.write(f"backed up {path} to {backup_path}\n")
    path.write_text(new_text, encoding="utf-8")
    sys.stdout.write(f"wrote {path}\n")
    return 0


def _mcp_endpoint_url(server: str) -> str:
    """Return the MCP endpoint of a server's base URL."""
    return server.rstrip("/") + MCP_ENDPOINT_PATH


def validated_server(server: str) -> str:
    """Return the server base URL, rejecting one no agent could reach."""
    if not server.startswith(("http://", "https://")):
        raise CodingAgentError(f"--server must be an http or https URL: {server}")
    return server


def validated_tenant(tenant: str) -> str:
    """Return the tenant name, rejecting one no HTTP header could carry."""
    if not tenant or any(
        character < " " or character == "\x7f" for character in tenant
    ):
        raise CodingAgentError(
            "--tenant must be a non-empty name without control characters"
        )
    return tenant

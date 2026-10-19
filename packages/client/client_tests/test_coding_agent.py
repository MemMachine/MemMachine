"""Tests for the coding agent installer."""

import json
import shlex
import subprocess
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from memmachine_client import cli, coding_agent

SERVER = "http://memmachine.test:8080"
ENDPOINT = "http://memmachine.test:8080/v1/mcp"
TENANT = "alice"

CODEX_TABLES = (
    "[mcp_servers.memmachine]\n"
    f'url = "{ENDPOINT}"\n'
    "\n"
    "[mcp_servers.memmachine.http_headers]\n"
    f'X-MemMachine-Tenant = "{TENANT}"\n'
)

UNRELATED_CODEX_CONFIG = """# Codex configuration
model = "gpt-5-codex"

[shell_environment_policy]
inherit = "core"

[mcp_servers.other]
command = "other-server"

[profiles.notes]
instructions = \"\"\"
Not a table header: [mcp_servers.memmachine]
and neither is this one: [mcp_servers.memmachine.http_headers]
\"\"\"
"""


@pytest.fixture(autouse=True)
def home_directory(tmp_path, monkeypatch):
    """Keep every file an install writes inside a temporary home directory."""
    directory = tmp_path / "home"
    directory.mkdir()
    monkeypatch.setenv("HOME", str(directory))
    monkeypatch.setenv("USERPROFILE", str(directory))
    return directory


@pytest.fixture
def claude_commands(monkeypatch):
    """Record the claude commands an install or disable would run."""
    commands = []

    def record(command, **keywords):
        commands.append(list(command))
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(coding_agent.shutil, "which", lambda name: f"/opt/bin/{name}")
    monkeypatch.setattr(coding_agent.subprocess, "run", record)
    return commands


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    """Point Codex's user scope at a temporary home directory."""
    home = tmp_path / "codex-home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


def run_cli(*arguments):
    """Run the command line the way a console script entry point does."""
    return cli.main(list(arguments))


def test_claude_code_user_scope_runs_the_documented_command(claude_commands, capsys):
    exit_code = run_cli(
        "agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 0
    assert claude_commands == [
        ["/opt/bin/claude", "mcp", "remove", "--scope", "user", "memmachine"],
        [
            "/opt/bin/claude",
            "mcp",
            "add",
            "--transport",
            "http",
            "--scope",
            "user",
            "memmachine",
            ENDPOINT,
            "--header",
            f"X-MemMachine-Tenant: {TENANT}",
        ],
    ]
    assert "mcp add --transport http" in capsys.readouterr().out


def test_claude_code_user_scope_install_twice_leaves_one_entry(claude_commands):
    for _ in range(2):
        run_cli(
            "agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT
        )

    # Each install removes the earlier entry before adding its own, which
    # is what leaves exactly one behind.
    assert [command[2] for command in claude_commands] == [
        "remove",
        "add",
        "remove",
        "add",
    ]


def test_claude_code_user_scope_disable_removes_the_entry(claude_commands):
    exit_code = run_cli("agent", "disable", "claude-code")

    assert exit_code == 0
    assert claude_commands == [
        ["/opt/bin/claude", "mcp", "remove", "--scope", "user", "memmachine"]
    ]


def test_claude_code_user_scope_dry_run_runs_nothing(
    claude_commands, home_directory, capsys
):
    exit_code = run_cli(
        "agent",
        "install",
        "claude-code",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--dry-run",
    )

    assert exit_code == 0
    assert claude_commands == []
    assert capsys.readouterr().out.startswith(
        "would run: claude mcp remove --scope user memmachine\n"
        "would run: claude mcp add --transport http --scope user memmachine "
        f"{ENDPOINT} --header 'X-MemMachine-Tenant: {TENANT}'\n"
    )


def test_claude_code_dry_run_needs_no_executable(home_directory, monkeypatch, capsys):
    monkeypatch.setenv("PATH", "")

    exit_code = run_cli(
        "agent",
        "install",
        "claude-code",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--dry-run",
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "would run: claude mcp remove --scope user memmachine\n" in output
    assert "would run: claude mcp add --transport http" in output


def test_claude_code_without_the_executable_names_the_command(monkeypatch, capsys):
    monkeypatch.setenv("PATH", "")

    exit_code = run_cli(
        "agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 1
    error_output = capsys.readouterr().err
    assert "no claude executable on PATH" in error_output
    assert (
        "claude mcp add --transport http --scope user memmachine "
        f"{ENDPOINT} --header 'X-MemMachine-Tenant: {TENANT}'" in error_output
    )


def test_claude_code_project_scope_writes_the_documented_shape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = run_cli(
        "agent",
        "install",
        "claude-code",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--scope",
        "project",
    )

    assert exit_code == 0
    assert json.loads((tmp_path / ".mcp.json").read_text()) == {
        "mcpServers": {
            "memmachine": {
                "type": "http",
                "url": ENDPOINT,
                "headers": {"X-MemMachine-Tenant": TENANT},
            }
        }
    }
    assert not (tmp_path / ".mcp.json.bak").exists()


def test_claude_code_project_scope_merges_and_backs_up(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".mcp.json"
    original = json.dumps({"mcpServers": {"other": {"command": "other-server"}}})
    config_path.write_text(original)

    for _ in range(2):
        assert (
            run_cli(
                "agent",
                "install",
                "claude-code",
                "--server",
                SERVER,
                "--tenant",
                TENANT,
                "--scope",
                "project",
            )
            == 0
        )

    document = json.loads(config_path.read_text())
    assert set(document["mcpServers"]) == {"other", "memmachine"}
    assert document["mcpServers"]["other"] == {"command": "other-server"}
    # The backup holds what the first install replaced; the second one
    # found the file already right and left it, and the backup, alone.
    assert (tmp_path / ".mcp.json.bak").read_text() == original
    assert "already holds this entry" in capsys.readouterr().out


def test_claude_code_project_scope_disable_keeps_other_servers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_cli(
        "agent",
        "install",
        "claude-code",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--scope",
        "project",
    )
    config_path = tmp_path / ".mcp.json"
    document = json.loads(config_path.read_text())
    document["mcpServers"]["other"] = {"command": "other-server"}
    document["someOtherKey"] = {"kept": True}
    config_path.write_text(json.dumps(document, indent=2))

    exit_code = run_cli("agent", "disable", "claude-code", "--scope", "project")

    assert exit_code == 0
    assert json.loads(config_path.read_text()) == {
        "mcpServers": {"other": {"command": "other-server"}},
        "someOtherKey": {"kept": True},
    }


def test_claude_code_project_scope_dry_run_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)

    exit_code = run_cli(
        "agent",
        "install",
        "claude-code",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--scope",
        "project",
        "--dry-run",
    )

    assert exit_code == 0
    assert not (tmp_path / ".mcp.json").exists()
    output = capsys.readouterr().out
    assert "would write" in output
    assert '+      "url": "http://memmachine.test:8080/v1/mcp",' in output


def test_codex_user_scope_writes_the_two_tables(codex_home):
    exit_code = run_cli(
        "agent", "install", "codex", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 0
    assert (codex_home / "config.toml").read_text() == CODEX_TABLES


def test_codex_user_scope_defaults_to_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    exit_code = run_cli(
        "agent", "install", "codex", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 0
    assert (tmp_path / ".codex" / "config.toml").read_text() == CODEX_TABLES


def test_codex_project_scope_writes_the_project_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = run_cli(
        "agent",
        "install",
        "codex",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--scope",
        "project",
    )

    assert exit_code == 0
    assert (tmp_path / ".codex" / "config.toml").read_text() == CODEX_TABLES


def test_codex_install_twice_leaves_one_entry(codex_home, capsys):
    for _ in range(2):
        assert (
            run_cli("agent", "install", "codex", "--server", SERVER, "--tenant", TENANT)
            == 0
        )

    text = (codex_home / "config.toml").read_text()
    assert text.count("[mcp_servers.memmachine]") == 1
    assert text.count("[mcp_servers.memmachine.http_headers]") == 1
    assert "already holds this entry" in capsys.readouterr().out


def test_codex_install_and_disable_preserve_unrelated_content(codex_home):
    config_path = codex_home / "config.toml"
    config_path.write_text(UNRELATED_CODEX_CONFIG)

    assert (
        run_cli("agent", "install", "codex", "--server", SERVER, "--tenant", TENANT)
        == 0
    )

    installed = config_path.read_text()
    assert installed == f"{UNRELATED_CODEX_CONFIG}\n{CODEX_TABLES}"
    document = tomllib.loads(installed)
    assert document["mcp_servers"]["memmachine"] == {
        "url": ENDPOINT,
        "http_headers": {"X-MemMachine-Tenant": TENANT},
    }
    assert document["mcp_servers"]["other"] == {"command": "other-server"}
    assert "[mcp_servers.memmachine]" in document["profiles"]["notes"]["instructions"]

    assert run_cli("agent", "disable", "codex") == 0

    assert config_path.read_text() == UNRELATED_CODEX_CONFIG
    assert (codex_home / "config.toml.bak").read_text() == installed


def test_codex_replaces_an_earlier_entry_where_it_stands(codex_home):
    config_path = codex_home / "config.toml"
    config_path.write_text(
        "[mcp_servers.memmachine]\n"
        'url = "http://old.test/v1/mcp"\n'
        "\n"
        "[mcp_servers.memmachine.http_headers]\n"
        'X-MemMachine-Tenant = "someone-else"\n'
        "\n"
        "[mcp_servers.other]\n"
        'command = "other-server"\n'
    )

    assert (
        run_cli("agent", "install", "codex", "--server", SERVER, "--tenant", TENANT)
        == 0
    )

    assert config_path.read_text() == (
        f'{CODEX_TABLES}\n[mcp_servers.other]\ncommand = "other-server"\n'
    )


def test_codex_refuses_an_entry_it_cannot_edit(codex_home, capsys):
    config_path = codex_home / "config.toml"
    original = '[mcp_servers]\nmemmachine = { url = "http://old.test/v1/mcp" }\n'
    config_path.write_text(original)

    exit_code = run_cli(
        "agent", "install", "codex", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 1
    assert config_path.read_text() == original
    assert "does not edit" in capsys.readouterr().err


def test_codex_refuses_a_file_that_is_not_toml(codex_home, capsys):
    config_path = codex_home / "config.toml"
    config_path.write_text("this is not toml\n")

    exit_code = run_cli(
        "agent", "install", "codex", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 1
    assert config_path.read_text() == "this is not toml\n"
    assert "is not valid TOML" in capsys.readouterr().err


def test_codex_dry_run_writes_nothing(codex_home, capsys):
    config_path = codex_home / "config.toml"
    config_path.write_text(UNRELATED_CODEX_CONFIG)

    exit_code = run_cli(
        "agent",
        "install",
        "codex",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--dry-run",
    )

    assert exit_code == 0
    assert config_path.read_text() == UNRELATED_CODEX_CONFIG
    assert not (codex_home / "config.toml.bak").exists()
    output = capsys.readouterr().out
    assert f"would write {config_path}" in output
    assert "+[mcp_servers.memmachine]" in output


def test_codex_disable_without_a_config_file(codex_home, capsys):
    exit_code = run_cli("agent", "disable", "codex")

    assert exit_code == 0
    assert not (codex_home / "config.toml").exists()
    assert "does not exist" in capsys.readouterr().out


def test_agent_install_needs_no_server_url_for_the_client(codex_home, monkeypatch):
    monkeypatch.delenv("MEMORY_BACKEND_URL", raising=False)

    exit_code = run_cli(
        "agent", "install", "codex", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 0


def test_a_tenant_no_header_could_carry_is_refused(codex_home, capsys):
    exit_code = run_cli(
        "agent", "install", "codex", "--server", SERVER, "--tenant", "two\nlines"
    )

    assert exit_code == 1
    assert not (codex_home / "config.toml").exists()
    assert "control characters" in capsys.readouterr().err


def test_a_server_no_agent_could_reach_is_refused(codex_home, capsys):
    exit_code = run_cli(
        "agent", "install", "codex", "--server", "memmachine.test", "--tenant", TENANT
    )

    assert exit_code == 1
    assert not (codex_home / "config.toml").exists()
    assert "must be an http or https URL" in capsys.readouterr().err


def capture_command(agent_name, server=SERVER, tenant=TENANT):
    """The command line the installer writes into a `Stop` hook."""
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


def capture_group(agent_name, server=SERVER):
    """The `Stop` group the installer writes."""
    return {
        "hooks": [
            {
                "type": "command",
                "command": capture_command(agent_name, server=server),
                "timeout": 10,
            }
        ]
    }


def test_claude_code_user_scope_registers_the_capture_hook(
    claude_commands, home_directory
):
    exit_code = run_cli(
        "agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 0
    settings = json.loads((home_directory / ".claude" / "settings.json").read_text())
    assert settings == {"hooks": {"Stop": [capture_group("claude-code")]}}
    # The hook names the interpreter this client is installed in, so a
    # shell whose PATH does not hold it still runs the capture.
    command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert command.startswith(shlex.quote(sys.executable))
    assert "-m memmachine_client.cli agent capture claude-code" in command


def test_claude_code_project_scope_registers_the_capture_hook(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = run_cli(
        "agent",
        "install",
        "claude-code",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--scope",
        "project",
    )

    assert exit_code == 0
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert settings == {"hooks": {"Stop": [capture_group("claude-code")]}}


def test_codex_user_scope_registers_the_capture_hook(codex_home):
    exit_code = run_cli(
        "agent", "install", "codex", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 0
    hooks = json.loads((codex_home / "hooks.json").read_text())
    assert hooks == {"hooks": {"Stop": [capture_group("codex")]}}


def test_codex_project_scope_registers_the_capture_hook(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = run_cli(
        "agent",
        "install",
        "codex",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--scope",
        "project",
    )

    assert exit_code == 0
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    assert hooks == {"hooks": {"Stop": [capture_group("codex")]}}


def test_installing_twice_leaves_one_hook(claude_commands, home_directory, capsys):
    for _ in range(2):
        assert (
            run_cli(
                "agent",
                "install",
                "claude-code",
                "--server",
                SERVER,
                "--tenant",
                TENANT,
            )
            == 0
        )

    settings_path = home_directory / ".claude" / "settings.json"
    assert json.loads(settings_path.read_text()) == {
        "hooks": {"Stop": [capture_group("claude-code")]}
    }
    assert "already holds this hook" in capsys.readouterr().out
    # The second install found the file right and left it, so there is
    # nothing for a backup to hold.
    assert not settings_path.with_suffix(".json.bak").exists()


def test_the_hook_of_an_earlier_install_is_replaced_where_it_stands(
    claude_commands, home_directory
):
    settings_path = home_directory / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    original = {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {"type": "command", "command": "chime.sh"},
                        {
                            "type": "command",
                            "command": capture_command(
                                "claude-code", server="http://old.test:8080"
                            ),
                            "timeout": 10,
                        },
                    ]
                }
            ]
        }
    }
    settings_path.write_text(json.dumps(original, indent=2))

    assert (
        run_cli(
            "agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT
        )
        == 0
    )

    assert json.loads(settings_path.read_text()) == {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {"type": "command", "command": "chime.sh"},
                        capture_group("claude-code")["hooks"][0],
                    ]
                }
            ]
        }
    }
    assert json.loads(settings_path.with_suffix(".json.bak").read_text()) == original


def test_installing_and_disabling_preserve_every_other_hook(
    claude_commands, home_directory
):
    settings_path = home_directory / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    original = {
        "permissions": {"allow": ["Bash(ls:*)"]},
        "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": "note.sh"}]}],
            "Stop": [{"hooks": [{"type": "command", "command": "chime.sh"}]}],
        },
    }
    settings_path.write_text(f"{json.dumps(original, indent=2)}\n")

    assert (
        run_cli(
            "agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT
        )
        == 0
    )

    installed = json.loads(settings_path.read_text())
    assert installed["permissions"] == {"allow": ["Bash(ls:*)"]}
    assert installed["hooks"]["SessionStart"] == original["hooks"]["SessionStart"]
    assert installed["hooks"]["Stop"] == [
        {"hooks": [{"type": "command", "command": "chime.sh"}]},
        capture_group("claude-code"),
    ]

    assert run_cli("agent", "disable", "claude-code") == 0

    assert json.loads(settings_path.read_text()) == original


def test_disabling_leaves_a_file_that_held_only_our_hook_empty(
    claude_commands, home_directory
):
    run_cli("agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT)

    assert run_cli("agent", "disable", "claude-code") == 0

    settings_path = home_directory / ".claude" / "settings.json"
    assert json.loads(settings_path.read_text()) == {}


def test_disabling_codex_removes_the_hook_it_wrote(codex_home):
    run_cli("agent", "install", "codex", "--server", SERVER, "--tenant", TENANT)
    hooks_path = codex_home / "hooks.json"
    document = json.loads(hooks_path.read_text())
    document["description"] = "kept"
    hooks_path.write_text(json.dumps(document, indent=2))

    assert run_cli("agent", "disable", "codex") == 0

    assert json.loads(hooks_path.read_text()) == {"description": "kept"}


def test_disabling_a_hook_that_is_not_there_writes_nothing(
    claude_commands, home_directory, capsys
):
    settings_path = home_directory / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    original = '{"permissions": {"allow": []}}'
    settings_path.write_text(original)

    assert run_cli("agent", "disable", "claude-code") == 0

    assert settings_path.read_text() == original
    assert not settings_path.with_suffix(".json.bak").exists()
    assert "holds no hook of ours" in capsys.readouterr().out


def test_disabling_without_a_settings_file_writes_none(
    claude_commands, home_directory, capsys
):
    assert run_cli("agent", "disable", "claude-code") == 0

    assert not (home_directory / ".claude" / "settings.json").exists()
    assert "does not exist" in capsys.readouterr().out


def test_a_dry_run_writes_no_hook(claude_commands, home_directory, capsys):
    exit_code = run_cli(
        "agent",
        "install",
        "claude-code",
        "--server",
        SERVER,
        "--tenant",
        TENANT,
        "--dry-run",
    )

    assert exit_code == 0
    assert not (home_directory / ".claude" / "settings.json").exists()
    output = capsys.readouterr().out
    assert "would write" in output
    assert '+    "Stop": [' in output


def test_a_settings_file_that_is_not_json_is_refused(
    claude_commands, home_directory, capsys
):
    settings_path = home_directory / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("half a file")

    exit_code = run_cli(
        "agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 1
    assert settings_path.read_text() == "half a file"
    assert "is not valid JSON" in capsys.readouterr().err


def test_hooks_that_are_not_a_list_are_refused(claude_commands, home_directory, capsys):
    settings_path = home_directory / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"hooks": {"Stop": "chime.sh"}}))

    exit_code = run_cli(
        "agent", "install", "claude-code", "--server", SERVER, "--tenant", TENANT
    )

    assert exit_code == 1
    assert "is not a list" in capsys.readouterr().err

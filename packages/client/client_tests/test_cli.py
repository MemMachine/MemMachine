"""Tests for the MemMachine command line client."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from memmachine_client import cli


class CliResult(BaseModel):
    """Small Pydantic test result."""

    uid: str


def test_pyproject_installs_command_name():
    """Editable installs should expose the documented mem-cli command."""
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )

    assert pyproject["project"]["scripts"]["mem-cli"] == ("memmachine_client.cli:main")


def test_build_client_uses_args_over_environment(monkeypatch):
    """Command arguments should override environment variables."""
    monkeypatch.setenv("MEMORY_BACKEND_URL", "http://env:8080")
    monkeypatch.setenv("MEMMACHINE_API_KEY", "env-key")
    monkeypatch.setenv("MEMMACHINE_TIMEOUT", "10")
    monkeypatch.setenv("MEMMACHINE_MAX_RETRIES", "2")

    args = cli.build_parser().parse_args(
        [
            "--base-url",
            "http://arg:8080",
            "--api-key",
            "arg-key",
            "--timeout",
            "20",
            "--max-retries",
            "4",
            "health",
        ]
    )

    client = cli.build_client(args)

    assert client.base_url == "http://arg:8080"
    assert client.api_key == "arg-key"
    assert client.timeout == 20
    assert client.max_retries == 4


def test_build_client_requires_base_url(monkeypatch):
    """The CLI should fail clearly when base_url is unavailable."""
    monkeypatch.delenv("MEMORY_BACKEND_URL", raising=False)

    args = cli.build_parser().parse_args(["health"])

    with pytest.raises(SystemExit) as exc:
        cli.build_client(args)

    assert exc.value.code == 2


def test_parser_help_uses_installed_command_name(capsys):
    """Help output should match the documented console script."""
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("usage: mem-cli ")


def test_custom_errors_use_parser_program_name(monkeypatch, capsys):
    """Custom errors should use the same program name as argparse."""
    monkeypatch.delenv("MEMORY_BACKEND_URL", raising=False)

    args = cli.build_parser().parse_args(["health"])

    with pytest.raises(SystemExit) as exc:
        cli.build_client(args)

    assert exc.value.code == 2
    assert capsys.readouterr().err == (
        "mem-cli: error: --base-url or MEMORY_BACKEND_URL is required\n"
    )


def test_health_command_prints_json(capsys):
    """The health command should call the underlying client."""
    client = Mock()
    client.health_check.return_value = {"status": "healthy"}

    result = cli.run_command(client, cli.build_parser().parse_args(["health"]))

    assert result == 0
    client.health_check.assert_called_once_with(timeout=None)
    assert capsys.readouterr().out == '{\n  "status": "healthy"\n}\n'


def test_memory_add_uses_env_project_context(monkeypatch, capsys):
    """Memory commands should accept project context from the environment."""
    monkeypatch.setenv("MEMMACHINE_ORG_ID", "env-org")
    monkeypatch.setenv("MEMMACHINE_PROJECT_ID", "env-project")
    client = Mock()
    project = Mock()
    memory = Mock()
    memory.add.return_value = [CliResult(uid="mem-1")]
    project.memory.return_value = memory
    client.get_project.return_value = project

    args = cli.build_parser().parse_args(
        ["memory", "add", "hello", "--metadata", "user_id=user-1"]
    )

    exit_code = cli.run_command(client, args)

    assert exit_code == 0
    client.get_project.assert_called_once_with(
        org_id="env-org",
        project_id="env-project",
        timeout=None,
    )
    project.memory.assert_called_once_with(metadata={"user_id": "user-1"})
    memory.add.assert_called_once_with(
        "hello",
        role="",
        producer=None,
        produced_for=None,
        metadata=None,
        timeout=None,
    )
    assert capsys.readouterr().out == '[\n  {\n    "uid": "mem-1"\n  }\n]\n'


@pytest.mark.parametrize(
    ("argv", "memory_method"),
    [
        (["memory", "search", "hello"], "search"),
        (["memory", "list"], "list"),
        (["memory", "delete-episodic", "--id", "episode-1"], "delete_episodic"),
        (["memory", "delete-semantic", "--id", "semantic-1"], "delete_semantic"),
    ],
)
def test_memory_subcommands_use_env_project_context(
    argv, memory_method, monkeypatch, capsys
):
    """Memory subcommands should resolve project context through the shared helper."""
    monkeypatch.setenv("MEMMACHINE_ORG_ID", "env-org")
    monkeypatch.setenv("MEMMACHINE_PROJECT_ID", "env-project")
    client = Mock()
    project = Mock()
    memory = Mock()
    getattr(memory, memory_method).return_value = (
        True if memory_method.startswith("delete_") else []
    )
    project.memory.return_value = memory
    client.get_project.return_value = project

    args = cli.build_parser().parse_args(argv)

    assert cli.run_command(client, args) == 0
    client.get_project.assert_called_once_with(
        org_id="env-org",
        project_id="env-project",
        timeout=None,
    )
    project.memory.assert_called_once_with(metadata={})
    assert capsys.readouterr().out


def test_projects_create_uses_env_project_context(monkeypatch, capsys):
    """Project create should accept project context from the environment."""
    monkeypatch.setenv("MEMMACHINE_ORG_ID", "env-org")
    monkeypatch.setenv("MEMMACHINE_PROJECT_ID", "env-project")
    client = Mock()
    project = Mock(
        org_id="env-org",
        project_id="env-project",
        description="",
        config={},
    )
    client.create_project.return_value = project

    args = cli.build_parser().parse_args(["projects", "create"])

    exit_code = cli.run_command(client, args)

    assert exit_code == 0
    client.create_project.assert_called_once_with(
        org_id="env-org",
        project_id="env-project",
        description="",
        embedder="",
        reranker="",
        timeout=None,
    )
    assert capsys.readouterr().out == (
        "{\n"
        '  "config": {},\n'
        '  "description": "",\n'
        '  "org_id": "env-org",\n'
        '  "project_id": "env-project"\n'
        "}\n"
    )


@patch("memmachine_client.cli.MemMachineClient")
def test_main_closes_client(mock_client_class, monkeypatch):
    """The top-level entry point should close the client after command execution."""
    monkeypatch.setenv("MEMORY_BACKEND_URL", "http://localhost:8080")
    client = Mock()
    client.health_check.return_value = {"status": "healthy"}
    mock_client_class.return_value = client

    exit_code = cli.main(["health"])

    assert exit_code == 0
    client.close.assert_called_once_with()

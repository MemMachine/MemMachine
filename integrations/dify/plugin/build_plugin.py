#!/usr/bin/env python3

"""
Build and package the MemMachine Dify plugin.

This script renders provider YAML/Python from Jinja templates, then packages the
plugin using the official `dify` CLI.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing build dependency 'jinja2'. Install it with: pip install jinja2"
    ) from e


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildOptions:
    """Build options controlling template output."""

    enable_base_url_credential: bool
    plugin_dir: Path


def _require_dify_plugin_cli() -> str:
    """Return the dify CLI path or raise if missing."""
    cmd = shutil.which("dify")
    if not cmd:
        raise RuntimeError(
            "Missing Dify plugin CLI: 'dify'.\n"
            "Install it first, then re-run.\n"
            "Expected command: dify plugin package <plugin_dir>"
        )
    return cmd


def _render_template(
    env: Environment,
    template_relpath: str,
    output_path: Path,
    context: dict[str, object],
) -> None:
    """Render a template into the requested output path."""
    template = env.get_template(template_relpath)
    rendered = template.render(**context)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def build_generated_files(opts: BuildOptions) -> None:
    """Render provider files from templates into the plugin directory."""
    templates_dir = opts.plugin_dir / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )

    context: dict[str, object] = {
        "enable_base_url_credential": opts.enable_base_url_credential,
        "default_base_url": "https://api.memmachine.ai/v2",
    }

    _render_template(
        env,
        "provider/memmachine-plugin.yaml.j2",
        opts.plugin_dir / "provider" / "memmachine-plugin.yaml",
        context,
    )
    _render_template(
        env,
        "provider/memmachine-plugin.py.j2",
        opts.plugin_dir / "provider" / "memmachine-plugin.py",
        context,
    )


def package_plugin(plugin_cli: str, plugin_dir: Path) -> None:
    """Package the plugin using the dify CLI."""
    subprocess.run(
        [plugin_cli, "plugin", "package", str(plugin_dir)],
        check=True,
    )


def parse_args(argv: list[str]) -> BuildOptions:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build MemMachine Dify plugin files (via Jinja templates) and package via dify CLI."
        )
    )
    parser.add_argument(
        "--enable-base-url-credential",
        action="store_true",
        help=(
            "If set, add an optional provider credential 'memmachine_base_url' to override the API base URL."
        ),
    )

    ns = parser.parse_args(argv)
    return BuildOptions(
        enable_base_url_credential=bool(ns.enable_base_url_credential),
        plugin_dir=Path(__file__).resolve().parent,
    )


def main(argv: list[str]) -> int:
    """Entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        opts = parse_args(argv)
        plugin_cli = _require_dify_plugin_cli()
        build_generated_files(opts)
        package_plugin(plugin_cli, opts.plugin_dir)
    except subprocess.CalledProcessError:
        logger.exception("Packaging failed")
        return 1
    except Exception:
        logger.exception("Build failed")
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

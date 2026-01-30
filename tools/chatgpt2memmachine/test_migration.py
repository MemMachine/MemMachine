# Simple test harness for `migration.py` using sample OpenAI export data.
# 1. cd ~/MemMachine; ./memmachine-compose.sh        # start MemMachine
# 2. cd ~/MemMachine/tools/chatgpt2memmachine        # change to this dir
# 3. uv run python3 test_migration.py                # run test

import os
import pytest

# Disable this file from pytest collection
pytestmark = pytest.mark.skip(reason="Manual test script, not automated pytest")

from migration import MigrationHack


def test_migration(dry_run: bool = True) -> None:
    """
    Run a sample migration using the bundled OpenAI conversations JSON.

    This uses the same parameters and filters that the CLI supports in `migration.py`,
    but hard-codes sensible defaults for quick testing.
    """
    base_url = "http://localhost:8080"

    # Change working directory to this script's directory so relative paths work
    my_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(my_dir)

    # Sample input file and parser source
    input_file = "data/conversations-chatgpt-sample.json"
    source = "openai"  # or "locomo" if you point to a Locomo-style file

    # Optional MemMachine identifiers (left empty in dry-run mode)
    org_id = ""
    project_id = ""

    # Example filters matching the parser/filter API
    filters: dict = {
        # "since": 0,                # uncomment to filter by timestamp
        # "limit": 100,              # uncomment to cap messages per conversation
        # "chat_title": "My Chat",   # uncomment to restrict to a single chat title (OpenAI only)
    }

    migration_hack = MigrationHack(
        base_url=base_url,
        org_id=org_id,
        project_id=project_id,
        input=input_file,
        source=source,
        filters=filters or None,
        dry_run=dry_run,
        verbose=True,
    )

    migration_hack.migrate()
    print("== All completed successfully")


if __name__ == "__main__":
    test_migration(dry_run=False)

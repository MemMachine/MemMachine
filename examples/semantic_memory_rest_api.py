"""Semantic Memory REST API Example

This script demonstrates how to use the MemMachine semantic memory API.

Semantic memory stores structured knowledge (facts, preferences, traits)
as features organized into sets and categories, enabling LLM-friendly
semantic search.

Prerequisites:
    - MemMachine server running (default: http://localhost:8080)
    - memmachine_client package installed

Configuration:
    The client can be configured via:
    1. Environment variable: MEMORY_BACKEND_URL (recommended)
       export MEMORY_BACKEND_URL="http://localhost:8080"
    2. Explicit parameter: base_url="http://localhost:8080"

Features demonstrated:
    - Creating semantic set types and categories
    - Adding semantic features (structured knowledge entries)
    - Searching semantic memory with filters
    - Listing and displaying semantic memory results
    - Cleanup of semantic resources
"""

from __future__ import annotations

import json
import os
import sys
import time

from memmachine_client import MemMachineClient

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"
{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_semantic_features(features, label: str = "Semantic Features") -> None:
    """Pretty-print a list of semantic features."""
    print(f"
  {label}:")
    if not features:
        print("    (none)")
        return
    for i, f in enumerate(features, 1):
        print(f"    [{i}] category={f.category}, tag={f.tag}, "
              f"feature={f.feature_name}, value={f.value}")
        if f.metadata and f.metadata.id:
            print(f"         id: {f.metadata.id}")


def print_search_result(result) -> None:
    """Pretty-print a combined episodic + semantic search result."""
    content = result.content

    # Episodic memories
    if content.episodic_memory:
        lt = content.episodic_memory.long_term_memory
        st = content.episodic_memory.short_term_memory
        if lt.episodes:
            print(f"
  Long-term episodes ({len(lt.episodes)}):")
            for ep in lt.episodes[:3]:
                print(f"    - {ep.content}")
        if st.episodes:
            print(f"
  Short-term episodes ({len(st.episodes)}):")
            for ep in st.episodes[:3]:
                print(f"    - {ep.content}")

    # Semantic memories
    if content.semantic_memory:
        print_semantic_features(content.semantic_memory, "Semantic Matches")

def demo_semantic_memory() -> None:
    """Run the semantic memory API demonstration."""
    print_section("Setup: Client and Project")

    base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
    print(f"Connecting to MemMachine at {base_url}")

    client = MemMachineClient(base_url=base_url, timeout=30)

    # Check server health
    try:
        health = client.health_check()
        print(f"Server health: {health}")
    except Exception as e:
        print(f"ERROR: Server not reachable at {base_url}: {e}")
        print("Start the MemMachine server first.")
        sys.exit(1)

    # Get or create a project for this demo
    org_id = "demo_org"
    project_id = "semantic_demo"

    try:
        project = client.get_project(org_id=org_id, project_id=project_id)
        print(f"Using existing project: {project_id}")
    except Exception:
        project = client.create_project(org_id=org_id, project_id=project_id)
        print(f"Created project: {project_id}")

    # Create a memory instance scoped to this demo
    memory = project.memory(
        group_id="semantic_demo_group",
        agent_id="semantic_demo_agent",
        user_id="demo_user",
        session_id="semantic_demo_session",
    )

    # ------------------------------------------------------------------
    print_section("1. Create a Semantic Set Type")

    # A set type defines the metadata schema that identifies semantic sets.
    # The metadata_tags are ordered key names that every set of this type
    # must supply when being created/retrieved.
    set_type_id = memory.create_semantic_set_type(
        metadata_tags=["domain", "language"],
        name="User Knowledge",
        description="Stores structured user knowledge across domains",
    )
    print(f"Created set type: {set_type_id}")

    # List available set types
    set_types = memory.list_semantic_set_types()
    print(f"Available set types: {len(set_types)}")
    for st in set_types:
        print(f"  - {st.set_type_id} (tags: {st.metadata_tags})")

    # ------------------------------------------------------------------
    print_section("2. Add Category Templates to the Set Type")

    # Category templates are inherited by all sets mapped to this type.
    # Each category has a prompt template the server uses to extract
    # features from unstructured text.
    category_template_id = memory.add_semantic_category_template(
        set_type_id=set_type_id,
        category_name="user_preference",
        prompt="Extract user preferences and likes from the conversation.",
        description="Captures user preferences and tastes",
    )
    print(f"Added category template: {category_template_id}")

    # ------------------------------------------------------------------
    print_section("3. Get (or Create) a Semantic Set ID")

    # A semantic set is identified by its type's metadata_tags plus the
    # concrete metadata values.  get_semantic_set_id() creates the set
    # on first call and returns the existing ID on subsequent calls.
    set_id = memory.get_semantic_set_id(
        metadata_tags=["domain", "language"],
        set_metadata={
            "domain": "programming",
            "language": "python",
        },
    )
    print(f"Semantic set ID: {set_id}")

    # ------------------------------------------------------------------
    print_section("4. Add Semantic Categories to the Set")

    # Categories group related features within a set.  They can be added
    # directly to a set (in addition to any templates inherited from the
    # set type).
    category_id = memory.add_semantic_category(
        set_id=set_id,
        category_name="developer_profile",
        prompt="Extract developer skill and experience information.",
        description="Developer skills and experience traits",
    )
    print(f"Added category: {category_id}")

    # ------------------------------------------------------------------
    print_section("5. Add Semantic Features (Structured Knowledge)")

    # Features are the atomic units of semantic memory.  Each feature
    # belongs to a category and carries a tag, a name, and a value.
    features_to_add = [
        {
            "category_name": "developer_profile",
            "tag": "skills",
            "feature": "language",
            "value": "Python",
        },
        {
            "category_name": "developer_profile",
            "tag": "skills",
            "feature": "framework",
            "value": "FastAPI",
        },
        {
            "category_name": "developer_profile",
            "tag": "experience",
            "feature": "years",
            "value": "5",
        },
        {
            "category_name": "developer_profile",
            "tag": "experience",
            "feature": "role",
            "value": "backend_engineer",
        },
        {
            "category_name": "user_preference",
            "tag": "tools",
            "feature": "editor",
            "value": "VS Code",
        },
        {
            "category_name": "user_preference",
            "tag": "tools",
            "feature": "os",
            "value": "Linux",
        },
        {
            "category_name": "user_preference",
            "tag": "learning",
            "feature": "topic",
            "value": "machine_learning",
        },
    ]

    feature_ids: list[str] = []
    for feat in features_to_add:
        try:
            fid = memory.add_feature(
                set_id=set_id,
                category_name=feat["category_name"],
                tag=feat["tag"],
                feature=feat["feature"],
                value=feat["value"],
            )
            feature_ids.append(fid)
            print(f"  Added: {feat['tag']}:{feat['feature']} = {feat['value']}  ->  {fid}")
        except Exception as e:
            print(f"  FAILED to add {feat['feature']}: {e}")

    # ------------------------------------------------------------------
    print_section("6. Search Semantic Memory with Filters")

    # The search() method queries both episodic and semantic memory.
    # Semantic results appear in result.content.semantic_memory.
    print('
  >>> Search: "What does the user know about Python?"')
    try:
        result = memory.search(
            query="What does the user know about Python?",
            limit=5,
        )
        print_search_result(result)
    except Exception as e:
        print(f"  Search failed: {e}")

    # Search with set_metadata to scope the query to a specific set
    print('
  >>> Search (scoped to programming/python set): "skills and experience"')
    try:
        result = memory.search(
            query="skills and experience",
            limit=5,
            set_metadata={
                "domain": "programming",
                "language": "python",
            },
        )
        print_search_result(result)
    except Exception as e:
        print(f"  Scoped search failed: {e}")

    # ------------------------------------------------------------------
    print_section("7. List Semantic Memories Directly")

    # The list() method can target semantic memory directly.
    try:
        list_result = memory.list(
            memory_type="Semantic",
            page_size=20,
        )
        print(f"Total semantic features: {list_result.total}")
        print_semantic_features(list_result.features)
    except Exception as e:
        print(f"  List failed: {e}")

    # ------------------------------------------------------------------
    print_section("8. Retrieve a Specific Feature by ID")

    if feature_ids:
        try:
            feature = memory.get_feature(feature_id=feature_ids[0])
            if feature:
                print(f"  Feature: {feature.feature_name} = {feature.value}")
                print(f"  Category: {feature.category}, Tag: {feature.tag}")
                print(f"  Set ID: {feature.set_id}")
                if feature.metadata and feature.metadata.id:
                    print(f"  Feature ID: {feature.metadata.id}")
        except Exception as e:
            print(f"  Get feature failed: {e}")

    # ------------------------------------------------------------------
    print_section("9. Cleanup")

    print("Cleaning up semantic resources...")

    # Delete features (in reverse order to avoid dependency issues)
    for fid in reversed(feature_ids):
        try:
            memory.delete_semantic(semantic_id=fid)
            print(f"  Deleted feature: {fid}")
        except Exception as e:
            print(f"  Failed to delete feature {fid}: {e}")

    # Delete the set type (this also cleans up associated sets and categories)
    try:
        memory.delete_semantic_set_type(set_type_id=set_type_id)
        print(f"  Deleted set type: {set_type_id}")
    except Exception as e:
        print(f"  Failed to delete set type {set_type_id}: {e}")

    print_section("Demo Complete")


def main() -> None:
    """Entry point for the semantic memory REST API example."""
    print("MemMachine Semantic Memory REST API Example")
    print("============================================")
    print()
    print("This example demonstrates the semantic memory API in MemMachine.")
    print("Semantic memory stores structured facts and enables semantic search.")
    print()
    print("Configuration:")
    base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
    print(f"  MEMORY_BACKEND_URL: {os.getenv('MEMORY_BACKEND_URL', 'not set (using default)')}")
    print(f"  Using base_url: {base_url}")
    print()
    print(f"Make sure MemMachine server is running on {base_url}")
    print()

    try:
        demo_semantic_memory()
    except KeyboardInterrupt:
        print("
Demo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"
Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

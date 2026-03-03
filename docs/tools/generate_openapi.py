"""Generate the OpenAPI specification from FastAPI route metadata.

Usage:
    uv run python docs/tools/generate_openapi.py
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from copy import deepcopy

from fastapi import FastAPI

from memmachine_server.common.api.version import get_version
from memmachine_server.server.api_v2.router import load_v2_api_router

UNTAGGED_LABEL = "Untagged"

TAG_DESCRIPTIONS = {
    "Projects": "Lifecycle management for isolated memory namespaces.",
    "Memories": "Core operations for episodic and semantic memory ingestion and retrieval.",
    "Configuration": "System overview and memory subsystem configuration.",
    "Resources": "Embedder, language model, and reranker lifecycle management.",
    "Episodic Configuration": "Per-project episodic memory subsystem configuration.",
    "Semantic Memory: Features": "Add, retrieve, and update individual semantic features.",
    "Semantic Memory: Sets": "Set type and set ID lifecycle, listing, and configuration.",
    "Semantic Memory: Categories": "Category, template, and tag management for semantic sets.",
    "System": "Infrastructure, health, and observability.",
    UNTAGGED_LABEL: "Endpoints missing an explicit tag — please add one.",
}

REQUEST_BODY_EXAMPLES: dict[str, dict] = {
    "create_project": {
        "org_id": "acme-corp",
        "project_id": "agent-smith-v1",
        "description": "Primary memory store for customer support agent",
        "config": {
            "embedder": "openai-text-3-small",
            "reranker": "cohere-rerank-english-v3.0",
        },
    },
}


def _clean_operation_id(operation_id: str) -> str:
    """Strip FastAPI's auto-generated suffixes like '_api_v2_..._post'."""
    # Remove the _api_v2_..._method suffix that FastAPI appends
    cleaned = re.sub(r"_api_v2_\w*_(?:post|get|put|delete|patch)$", "", operation_id)
    # Strip trailing _endpoint from config router function names
    cleaned = re.sub(r"_endpoint$", "", cleaned)
    return cleaned


def _sort_paths(paths: dict, tag_order: list[str]) -> dict:
    """Sort paths by tag group order, then alphabetically within each group."""
    tag_index = {tag: i for i, tag in enumerate(tag_order)}
    default_index = len(tag_order)

    def sort_key(item: tuple) -> tuple:
        path, methods = item
        # Collect all tags across every method on this path, then pick the
        # one with the lowest tag_order index.  This is deterministic
        # regardless of HTTP-method iteration order or per-method tag
        # differences.
        all_tags = {tag for op in methods.values() for tag in op.get("tags", [])}
        best = min(
            (tag_index.get(t, default_index) for t in all_tags),
            default=default_index,
        )
        return (best, path)

    return dict(sorted(paths.items(), key=sort_key))


def generate() -> dict:
    """Build the OpenAPI spec dict."""
    app = FastAPI()
    load_v2_api_router(app, with_config_api=True)
    spec = deepcopy(app.openapi())

    # Strip SCM dev metadata (e.g. "0.2.7.dev69+g71322db" -> "0.2.7") so the
    # spec only changes on tagged releases, not every commit.
    raw_version = get_version().server_version
    stable_version = re.sub(r"\.dev\d+.*$", "", raw_version)

    spec["info"] = {
        "title": "MemMachine API",
        "version": stable_version,
        "description": (
            "Architectural Memory Systems for AI Agents. "
            "Specialized in episodic and semantic memory management "
            "with strict namespace isolation."
        ),
    }
    spec["servers"] = [{"url": "http://localhost:8080"}]

    # Clean operationIds and inject request body examples.
    # Examples are keyed on the cleaned operationId (pre-disambiguation),
    # so injection happens here before disambiguation may alter the id.
    for _path, methods in spec.get("paths", {}).items():
        for _method, operation in methods.items():
            if "operationId" in operation:
                clean_id = _clean_operation_id(operation["operationId"])
                operation["operationId"] = clean_id

                if clean_id in REQUEST_BODY_EXAMPLES:
                    rb = operation.get("requestBody", {})
                    content = rb.get("content", {})
                    json_ct = content.get("application/json", {})
                    if json_ct:
                        json_ct["example"] = REQUEST_BODY_EXAMPLES[clean_id]

    # Disambiguate duplicate operationIds by appending a tag-derived suffix.
    # If the suffix itself collides (same tag), append an incrementing counter.
    seen: dict[str, list[tuple[str, dict]]] = {}
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            oid = operation.get("operationId", "")
            if oid:
                seen.setdefault(oid, []).append((path, operation))

    for oid, entries in seen.items():
        if len(entries) <= 1:
            continue
        assigned: dict[str, int] = {}
        for _path, operation in entries:
            tag = (operation.get("tags") or [""])[0]
            suffix = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
            candidate = f"{oid}_{suffix}"
            count = assigned.get(candidate, 0)
            assigned[candidate] = count + 1
            operation["operationId"] = (
                candidate if count == 0 else f"{candidate}_{count}"
            )

    # Warn about and auto-tag untagged endpoints
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if not operation.get("tags"):
                op_id = operation.get("operationId", "unknown")
                warnings.warn(
                    f"Untagged endpoint: {method.upper()} {path} "
                    f"(operationId={op_id!r}). "
                    f"Assigned to {UNTAGGED_LABEL!r} — please add an explicit tag.",
                    stacklevel=2,
                )
                operation["tags"] = [UNTAGGED_LABEL]

    # Determine tag order, sort paths, and set tag metadata
    all_tags: set[str] = set()
    for methods in spec.get("paths", {}).values():
        for operation in methods.values():
            all_tags.update(operation.get("tags", []))

    tag_order = sorted(all_tags)

    spec["paths"] = _sort_paths(spec.get("paths", {}), tag_order)
    spec["tags"] = [
        {"name": tag, "description": TAG_DESCRIPTIONS.get(tag, "")} for tag in tag_order
    ]

    # Emit keys in canonical OpenAPI order
    key_order = ["openapi", "info", "servers", "tags", "paths", "components"]
    ordered: dict = {k: spec[k] for k in key_order if k in spec}
    # Preserve any unexpected keys at the end
    for k, v in spec.items():
        if k not in ordered:
            ordered[k] = v

    return ordered


def main() -> None:
    spec = generate()
    json.dump(spec, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

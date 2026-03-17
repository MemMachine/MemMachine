from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

from memmachine_server.common.configuration import Configuration
from memmachine_server.common.resource_manager.resource_manager import ResourceManagerImpl
from memmachine_server.semantic_memory.semantic_ingestion import IngestionService

from evaluation.semantic_memory.ingest import build_episode_entries
from evaluation.semantic_memory.search import format_feature_context
from evaluation.semantic_memory.semantic_harness import (
    build_run_config,
    maybe_start_pg_container,
)


ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with answering questions using
semantic features.

# CONTEXT:
You have access to semantic features extracted from a conversation.

# INSTRUCTIONS:
1. Answer using only the provided features.
2. If no feature is relevant, reply "unknown".
3. Keep the answer under 6 words.

<FEATURES>
{feature_context}
</FEATURES>

Question: {question}

Answer:
"""


def build_variant_plan() -> dict[str, float]:
    return {
        "clustered": 0.3,
        "no_cluster": 1.0,
    }


def _load_yaml(path: str) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _apply_db_overrides(config: dict[str, Any], db_info: dict[str, Any]) -> None:
    resources = config.get("resources", {})
    databases = resources.get("databases", {})
    for db in databases.values():
        db_config = db.get("config")
        if not isinstance(db_config, dict):
            continue
        db_config.update(
            {
                "host": db_info["host"],
                "port": db_info["port"],
                "user": db_info["user"],
                "password": db_info["password"],
                "db_name": db_info["db_name"],
            }
        )


def _get_answer_model_config(
    config: dict[str, Any],
) -> tuple[str, str | None, str | None]:
    resources = config.get("resources", {})
    language_models = resources.get("language_models", {})
    semantic_conf = config.get("semantic_memory", {})
    llm_model_id = semantic_conf.get("llm_model")

    model_name = "gpt-4o-mini"
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = None

    if llm_model_id and llm_model_id in language_models:
        llm_config = language_models[llm_model_id].get("config", {})
        model_name = llm_config.get("model", model_name)
        api_key = llm_config.get("api_key") or api_key
        base_url = llm_config.get("base_url") or base_url

    return model_name, api_key, base_url


def _within_range(idx: int, conv_start: int | None, conv_stop: int | None) -> bool:
    if conv_start is not None and idx < conv_start - 1:
        return False
    if conv_stop is not None and idx > conv_stop - 1:
        return False
    return True


async def _ingest_conversations(
    *,
    semantic_service,
    ingestion_service: IngestionService,
    locomo_data: list[dict[str, Any]],
    conv_start: int | None,
    conv_stop: int | None,
) -> None:
    episode_storage = semantic_service._episode_storage

    for idx, item in enumerate(locomo_data):
        if not _within_range(idx, conv_start, conv_stop):
            continue
        if "conversation" not in item:
            continue

        conversation = item["conversation"]
        group_id = f"group_{idx}"

        session_idx = 0
        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"
            if session_id not in conversation:
                break

            session = conversation[session_id]
            session_date_time = conversation.get(f"{session_id}_date_time")
            for message in session:
                speaker = message["speaker"]
                content = message["text"]

                entries = build_episode_entries(
                    session_id=session_id,
                    speaker=speaker,
                    content=content,
                    timestamp=session_date_time,
                )
                episodes = await episode_storage.add_episodes(
                    session_key=group_id,
                    episodes=entries,
                )
                for episode in episodes:
                    await semantic_service.add_message_to_sets(episode.uid, [group_id])

        await ingestion_service.process_set_ids([group_id])


async def _search_questions(
    *,
    semantic_service,
    openai_client: AsyncOpenAI,
    answer_model: str,
    locomo_data: list[dict[str, Any]],
    conv_start: int | None,
    conv_stop: int | None,
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}

    for idx, item in enumerate(locomo_data):
        if not _within_range(idx, conv_start, conv_stop):
            continue
        if "conversation" not in item:
            continue

        conversation = item["conversation"]
        qa_list = item.get("qa", [])
        group_id = f"group_{idx}"

        for qa in qa_list:
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa.get("evidence", "")
            adversarial_answer = qa.get("adversarial_answer", "")

            features = []
            async for feature in semantic_service.search(
                set_ids=[group_id],
                query=question,
                limit=30,
                load_citations=True,
            ):
                features.append(
                    {
                        "feature_name": feature.feature_name,
                        "value": feature.value,
                        "category": feature.category,
                        "tag": feature.tag,
                        "metadata": {
                            "citations": list(feature.metadata.citations or [])
                        },
                    }
                )

            feature_context = format_feature_context(features)
            prompt = ANSWER_PROMPT.format(
                feature_context=feature_context,
                question=question,
            )
            rsp = await openai_client.responses.create(
                model=answer_model,
                max_output_tokens=256,
                temperature=0.0,
                top_p=1,
                input=[{"role": "user", "content": prompt}],
            )

            result = {
                "question": question,
                "locomo_answer": answer,
                "model_answer": rsp.output_text,
                "category": category,
                "evidence": evidence,
                "adversarial_answer": adversarial_answer,
                "conversation_memories": feature_context,
            }
            category_result = results.get(category, [])
            category_result.append(result)
            results[category] = category_result

    return results


async def _run_variant(
    *,
    base_config: dict[str, Any],
    data_path: str,
    run_dir: Path,
    similarity_threshold: float,
    conv_start: int | None,
    conv_stop: int | None,
    use_testcontainer: bool,
) -> None:
    locomo_data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    run_config = build_run_config(
        base_config, similarity_threshold=similarity_threshold
    )

    if use_testcontainer:
        with maybe_start_pg_container(True) as db_info:
            if db_info is not None:
                _apply_db_overrides(run_config, db_info)
            await _execute_run(
                run_config=run_config,
                locomo_data=locomo_data,
                run_dir=run_dir,
                conv_start=conv_start,
                conv_stop=conv_stop,
            )
    else:
        await _execute_run(
            run_config=run_config,
            locomo_data=locomo_data,
            run_dir=run_dir,
            conv_start=conv_start,
            conv_stop=conv_stop,
        )


async def _execute_run(
    *,
    run_config: dict[str, Any],
    locomo_data: list[dict[str, Any]],
    run_dir: Path,
    conv_start: int | None,
    conv_stop: int | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "semantic_config.yaml"
    _write_yaml(run_config, run_config_path)

    config = Configuration.load_yml_file(str(run_config_path))
    resource_manager = ResourceManagerImpl(config)
    await resource_manager.build()
    semantic_service = await resource_manager.get_semantic_service()

    ingestion_service = IngestionService(
        params=IngestionService.Params(
            semantic_storage=semantic_service._semantic_storage,
            resource_retriever=semantic_service._set_id_resource,
            history_store=semantic_service._episode_storage,
            ingestion_trigger_messages=semantic_service._cluster_ingestion_message_limit,
            ingestion_trigger_age=semantic_service._cluster_ingestion_time_limit,
            cluster_idle_ttl=semantic_service._cluster_idle_ttl,
            cluster_state_storage=semantic_service._cluster_state_storage,
            cluster_params=semantic_service._cluster_params,
            cluster_splitter=semantic_service._build_cluster_splitter(),
            consolidated_threshold=semantic_service._consolidation_threshold,
            debug_fail_loudly=semantic_service._debug_fail_loudly,
        )
    )

    await _ingest_conversations(
        semantic_service=semantic_service,
        ingestion_service=ingestion_service,
        locomo_data=locomo_data,
        conv_start=conv_start,
        conv_stop=conv_stop,
    )

    model_name, api_key, base_url = _get_answer_model_config(run_config)
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set for semantic evaluation")

    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    search_results = await _search_questions(
        semantic_service=semantic_service,
        openai_client=openai_client,
        answer_model=model_name,
        locomo_data=locomo_data,
        conv_start=conv_start,
        conv_stop=conv_stop,
    )

    search_results_path = run_dir / "search_results.json"
    search_results_path.write_text(
        json.dumps(search_results, indent=4),
        encoding="utf-8",
    )

    eval_path = run_dir / "evaluation_metrics.json"
    eval_script = Path(__file__).with_name("locomo_evaluate.py")
    subprocess.run(
        [
            sys.executable,
            str(eval_script),
            "--data-path",
            str(search_results_path),
            "--target-path",
            str(eval_path),
        ],
        check=True,
    )

    scores_script = Path(__file__).with_name("generate_scores.py")
    scores = subprocess.run(
        [sys.executable, str(scores_script)],
        check=True,
        cwd=run_dir,
        capture_output=True,
        text=True,
    )
    (run_dir / "score_summary.txt").write_text(scores.stdout, encoding="utf-8")

    await resource_manager.close()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the source data")
    parser.add_argument("--config", required=True, help="Path to semantic config yaml")
    parser.add_argument("--run-id", required=True, help="Run identifier for outputs")
    parser.add_argument("--conv-start", type=int)
    parser.add_argument("--conv-stop", type=int)
    parser.add_argument(
        "--no-testcontainer",
        action="store_true",
        help="Disable testcontainers Postgres",
    )

    args = parser.parse_args()

    base_config = _load_yaml(args.config)
    variants = build_variant_plan()
    run_root = Path("runs") / args.run_id

    for variant_name, similarity_threshold in variants.items():
        run_dir = run_root / variant_name
        await _run_variant(
            base_config=base_config,
            data_path=args.data_path,
            run_dir=run_dir,
            similarity_threshold=similarity_threshold,
            conv_start=args.conv_start,
            conv_stop=args.conv_stop,
            use_testcontainer=not args.no_testcontainer,
        )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.retrieval_skill.wikimultihop_ingest import load_data  # noqa: E402
from evaluation.utils import skill_utils  # noqa: E402

# Citation: Luo et al. (2025), "Agent Lightning: Train ANY AI Agents with
# Reinforcement Learning", arXiv:2508.03680.
ANSWER_PROMPT = """You are asked to answer `{question}` using `{memories}` as the primary source when they contain sufficient evidence; otherwise use general world knowledge.

<instructions>
1. Normalize inputs before deciding anything:
   - Treat `{memories}` as possibly empty.
   - Normalize entity spellings/case/ordinals/titles and common aliases (e.g., “10Th” → “10th”; honorific variants).
   - If `{question}` is malformed, underspecified, or missing key constraints, ask exactly one concise clarifying question instead of answering.

2. Choose the evidence basis using this strict priority:
   (a) **Memory-explicit**: Use when `{memories}` contain at least one explicit statement that answers the question or provides all necessary facts.
   (b) **Memory-determined inference**: Use when explicit memory facts, taken together, *fully determine* the answer unambiguously (show minimal reasoning).
   (c) **Open-domain fallback**: Use general world knowledge when memories are empty/irrelevant/too vague OR do not fully determine the answer.

2.1 Attribute-target resolution rules:
   - For questions phrased as "work at"/employer/organization, output organization names (not role titles). If both appear, prefer organization entities. If an intergovernmental organization appears in memory for the resolved person, include that organization in the answer.
   - For place-of-death questions, if no explicit "died in/at" location exists but a compact lifespan line exists with a single location token (e.g., `[birth_year] [city] - [death_year]`), use that location as best-available answer.
   - For relation-chain questions (parent/spouse/child/grandparent and similar), resolve each hop and return only the final requested entity/attribute, not an intermediate hop entity.
   - If memory contains `[StageResult ...] Query: ... Answer: ...` lines:
     - `reliability=high` (or confidence >= 0.85): treat as strong distilled evidence unless contradicted.
     - `reliability=tentative` (or confidence < 0.85): treat as a hypothesis; require explicit corroboration from memory or use open-domain fallback.
     - If stage text includes uncertainty cues (e.g., "if", "likely", "inferred", "unknown", "traditional"), do not treat it as final by itself.
   - If memory contains `Sub-skill provisional answer candidate (...)`, treat it as weak evidence only; never treat it as final without explicit corroboration.
   - For any candidate tagged `reliability=tentative` or `Status: unverified`, do not copy it verbatim unless explicit memory evidence directly supports the final asked attribute.
   - For yes/no same-country or same-nationality questions, compare normalized country/nationality sets and answer **yes** if they share at least one country/nationality (e.g., "British-American" overlaps with "American").
   - For same-country/same-nationality questions, before answering **no**, expand each side with commonly known dual/multiple nationalities from open-domain knowledge and re-check overlap.
   - For country/nationality questions, do not infer from industry labels or context words alone (e.g., "Bollywood", "Hollywood", language, genre); require explicit country/nationality evidence or use open-domain fallback.
   - If question asks for a country and evidence gives only a demonym/adjectival nationality (e.g., "American"), normalize to the corresponding country ("United States").
   - For place/location questions, reject date-only or non-location candidates as insufficient and continue reasoning/fallback.

3. Uncertainty rule:
   - Do **not** say “unknown/not mentioned” if open-domain knowledge can reasonably answer.
   - Prefer a best-supported concrete answer from open-domain knowledge when memory evidence is sparse/conflicting but a widely accepted answer exists.
   - If neither memories nor general knowledge allow a confident answer, say “I don’t know” (optionally add a brief reason).

4. Ambiguity handling:
   - If multiple plausible entities/answers remain after normalization, provide the top candidates and note the ambiguity briefly.
   - If multiple valid answers are genuinely possible, enumerate them (comma-separated or short bullets).

5. Computation and counting:
   - For counts or time intervals, compute explicitly (brief enumeration or numeric subtraction) to avoid mistakes.

6. Output requirements (concise, auditable):
   - Provide the **Answer** only, without additional commentary.
   - Keep the total response to **max 2 sentences**, except when enumeration/computation is required; then use **up to 4 short lines** (bullets allowed) while staying as brief as possible.
</instructions>

<memories>
{memories}
</memories>

Question: {question}
"""


async def run_wiki(
    dpath: str | None = None,
    epath: str | None = None,
    data_path: str | None = None,
    eval_result_path: str | None = None,
    length: int | None = None,
    model_name: str = "gpt-5.2",
    runner_kwargs: dict | None = None,
    session_id: str = "group1",
) -> tuple[str, dict[str, Any]]:
    if data_path is not None:
        _data_path = data_path
        _eval_result_path = eval_result_path
        _length = length or 100
        _test_target = "retrieval_skill"
    else:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--data-path", required=True, help="Path to the source data file"
        )
        parser.add_argument(
            "--eval-result-path",
            required=True,
            help="Path to save evaluation results",
            default=None,
        )
        parser.add_argument(
            "--length", type=int, default=500, help="Number of questions to search"
        )
        parser.add_argument(
            "--test-target",
            required=True,
            help="Testing with retrieval_skill or pure llm",
            choices=["retrieval_skill", "llm"],
        )
        parser.add_argument(
            "--session-id",
            required=False,
            default="group1",
            help="Evaluation session/project identifier for REST-backed runs",
        )

        args = parser.parse_args()
        _data_path = dpath or args.data_path
        _eval_result_path = epath or args.eval_result_path
        _length = args.length
        _test_target = args.test_target
        session_id = args.session_id

    print("Starting WikiMultiHop test...")
    print(f"Data path: {_data_path}")
    print(f"Evaluation result path: {_eval_result_path}")
    print(f"Length: {_length}")
    print(f"Test target: {_test_target}")
    effective_runner_kwargs = dict(runner_kwargs or {})
    if _test_target == "retrieval_skill":
        effective_runner_kwargs.setdefault("stage_result_mode", True)
        effective_runner_kwargs.setdefault(
            "omit_episode_text_on_confident_stage_result", True
        )
        effective_runner_kwargs.setdefault("use_answer_prompt_template", True)

    vector_graph_store = skill_utils.init_vector_graph_store(
        neo4j_uri="bolt://localhost:7687"
    )
    _, model, query_skill = await skill_utils.init_memmachine_params(
        vector_graph_store=vector_graph_store,
        session_id=session_id,
        model_name=model_name,
        runner_config=effective_runner_kwargs,
        build_runner=_test_target == "retrieval_skill",
    )
    if _test_target == "retrieval_skill" and query_skill is None:
        raise RuntimeError("WikiMultiHop benchmark requires an initialized SkillRunner.")

    contexts, questions, answers, types, supporting_facts = load_data(
        data_path=_data_path, start_line=1, end_line=_length, randomize="NONE"
    )
    print(f"Loaded {len(questions)} questions, start querying...")

    tasks = []
    results: dict[str, Any] = {}
    attribute_matrix = skill_utils.init_attribute_matrix()
    full_content = "\n".join(contexts)
    num_processed = 0
    question_batch_size = 2 if _test_target == "retrieval_skill" else 25
    for q, a, t, f_list in zip(
        questions, answers, types, supporting_facts, strict=True
    ):
        tasks.append(
            skill_utils.process_question_with_runner(
                answer_prompt=ANSWER_PROMPT,
                runner=query_skill,
                model=model,
                question=q,
                answer=a,
                category=t,
                supporting_facts=f_list,
                adversarial_answer="",
                model_name=model_name,
                full_content=full_content if _test_target == "llm" else None,
            )
        )

        if len(tasks) % question_batch_size == 0 or (q == questions[-1]):
            responses = await asyncio.gather(*tasks)
            tasks = []
            skill_utils.update_results(responses, attribute_matrix, results)
            num_processed += len(responses)
            print(f"Completed searching {num_processed}/{len(questions)} questions...")

    skill_utils.update_final_attribute_matrix(
        "wiki",
        attribute_matrix,
        results,
    )
    return _eval_result_path, results


async def main():
    eval_result_path, results = await run_wiki()
    with open(eval_result_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())

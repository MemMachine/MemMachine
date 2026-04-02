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
ANSWER_PROMPT = """Answer `{question}` using `{memories}` as the primary source. This is an offline evaluation: do not ask the user a clarifying question, do not request more context, and do not answer with another question.

<instructions>
1. Normalize inputs before deciding anything:
   - Treat `{memories}` as possibly empty.
   - Normalize entity spellings/case/ordinals/titles and common aliases (e.g., “10Th” → “10th”; honorific variants).
   - If `{question}` is slightly underspecified, infer the most likely target from the question anchors and retrieved memories; do not ask for clarification.

2. Choose the evidence basis using this strict priority:
   (a) **Memory-explicit**: Use when `{memories}` contain at least one explicit statement that answers the question or provides all necessary facts.
   (b) **Memory-determined inference**: Use when explicit memory facts, taken together, fully determine the answer unambiguously.
   (c) **Open-domain fallback**: Use general world knowledge only when memories do not identify or determine the final asked attribute.

3. Attribute-target resolution rules:
   - For questions phrased as "work at"/employer/organization, output organization names (not role titles). If both appear, prefer organization entities. If an intergovernmental organization appears in memory for the resolved person, include that organization in the answer.
   - For place-of-death questions, if no explicit "died in/at" location exists but a compact lifespan line exists with a single location token (e.g., `[birth_year] [city] - [death_year]`), use that location as best-available answer.
   - For relation-chain questions (parent/spouse/child/grandparent/in-law and similar), resolve each hop and return only the final requested entity/attribute, not an intermediate hop entity.
   - Decompose composite kinship relations step-by-step: maternal grandfather = mother -> her father; paternal grandmother = father -> his mother; father-in-law/mother-in-law = spouse -> their father/mother.
   - For creator/performer relations, first resolve the person (director, performer, singer, author, etc.), then answer the asked attribute for that exact person.
   - If a previous hop resolved a person with disambiguating context, preserve that exact identity. Do not switch to a more famous namesake with the same name.
   - If memory contains `[StageResult ...] Query: ... Answer: ...` lines:
     - `reliability=high` (or confidence >= 0.9): treat as strong distilled evidence unless contradicted.
     - `reliability=tentative` (or confidence < 0.9): treat as a hypothesis; require explicit corroboration from memory or open-domain fallback.
     - If stage text includes uncertainty cues (e.g., "if", "likely", "inferred", "unknown", "traditional"), do not treat it as final by itself.
   - If memory contains `Sub-skill provisional answer candidate (...)`, treat it as weak evidence only; never treat it as final without explicit corroboration.
   - For any candidate tagged `reliability=tentative` or `Status: unverified`, do not copy it verbatim unless explicit memory evidence directly supports the final asked attribute.
   - For country/nationality questions, do not infer from industry labels or context words alone (e.g., "Bollywood", "Hollywood", language, genre); require explicit country/nationality evidence or use open-domain fallback.
   - For country/nationality questions, prefer the polity/country explicitly named in evidence. Do not replace an explicitly named kingdom, state, or polity with a modern country unless the question clearly asks for the modern country.
   - If question asks for a country and evidence gives only a demonym/adjectival nationality (e.g., "American"), normalize to the corresponding country ("United States").
   - For place/location questions, reject date-only or non-location candidates as insufficient and continue reasoning/fallback.

4. Comparison and uncertainty rules:
   - For earlier/later, born first, died first, older/younger, more recent, and similar comparisons, explicitly determine both sides before choosing. Do not answer until both comparison values are resolved from memory or widely known world knowledge.
   - For yes/no same-country or same-nationality questions, compare normalized country/nationality sets and answer **yes** if they share at least one country/nationality.
   - Do **not** say “unknown/not mentioned” if open-domain knowledge can reasonably answer.
   - Prefer a best-supported concrete answer from open-domain knowledge when memory evidence is sparse/conflicting but a widely accepted answer exists.
   - If neither memories nor general knowledge allow a confident answer, say “I don’t know” (optionally add a brief reason).

5. Ambiguity handling:
   - If multiple plausible entities/answers remain after normalization, provide the top candidates and note the ambiguity briefly.
   - If multiple valid answers are genuinely possible, enumerate them (comma-separated or short bullets).

6. Computation and counting:
   - For counts or time intervals, compute explicitly (brief enumeration or numeric subtraction) to avoid mistakes.

7. Output requirements:
   - Provide the final answer only, without additional commentary.
   - Never return a clarifying question.
   - Keep the total response to **max 2 sentences**, except when enumeration/computation is required; then use **up to 4 short lines** (bullets allowed) while staying as brief as possible.
</instructions>

<memories>
{memories}
</memories>

Question: {question}
"""


async def run_wiki(  # noqa: C901
    dpath: str | None = None,
    epath: str | None = None,
    data_path: str | None = None,
    eval_result_path: str | None = None,
    length: int | None = None,
    model_name: str = "gpt-5.2",
    runner_kwargs: dict | None = None,
    session_id: str = "group1",
    concurrency: int = 10,
    answer_llm: object | None = None,
) -> tuple[str, dict[str, Any]]:
    if data_path is not None:
        _data_path = data_path
        _eval_result_path = eval_result_path
        _length = length or 100
        _test_target = "retrieval_skill"
        _concurrency = concurrency
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
        parser.add_argument(
            "--concurrency",
            type=int,
            default=10,
            help="Maximum number of concurrent search requests (default: 10)",
        )
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to benchmark_config.yml for answer model",
        )

        args = parser.parse_args()
        _data_path = dpath or args.data_path
        _eval_result_path = epath or args.eval_result_path
        _length = args.length
        _test_target = args.test_target
        session_id = args.session_id
        _concurrency = args.concurrency

        if answer_llm is None and args.config:
            from evaluation.retrieval_skill.benchmark_config import (
                LLMClient,
                load_benchmark_config,
            )

            cfg = load_benchmark_config(args.config)
            answer_llm = LLMClient(cfg.answer_model)

    if answer_llm is not None:
        model_name = answer_llm.model_name

    print("Starting WikiMultiHop test...")
    print(f"Data path: {_data_path}")
    print(f"Evaluation result path: {_eval_result_path}")
    print(f"Length: {_length}")
    print(f"Test target: {_test_target}")
    print(f"Concurrency: {_concurrency}")
    if answer_llm is not None:
        print(f"Answer model: {answer_llm.provider} / {answer_llm.model_name}")
    effective_runner_kwargs = dict(runner_kwargs or {})
    if _test_target == "retrieval_skill":
        effective_runner_kwargs.setdefault("stage_result_mode", True)
        effective_runner_kwargs.setdefault(
            "omit_episode_text_on_confident_stage_result", True
        )
        effective_runner_kwargs.setdefault("use_answer_prompt_template", False)

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
        raise RuntimeError(
            "WikiMultiHop benchmark requires an initialized SkillRunner."
        )

    contexts, questions, answers, types, supporting_facts = load_data(
        data_path=_data_path, start_line=1, end_line=_length, randomize="NONE"
    )
    warmup_query = next((question.strip() for question in questions if question), "")
    if _test_target == "retrieval_skill" and warmup_query:
        elapsed = await skill_utils.warmup_rest_evaluation_search(
            session_id=session_id,
            query=warmup_query,
            raise_on_failure=True,
        )
        if elapsed is not None:
            print(f"Search backend ready for {session_id} in {elapsed:.3f}s")
    print(f"Loaded {len(questions)} questions, start querying...")

    tasks = []
    results: dict[str, Any] = {}
    attribute_matrix = skill_utils.init_attribute_matrix()
    full_content = "\n".join(contexts)
    num_processed = 0
    question_batch_size = _concurrency
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
                answer_llm=answer_llm,
            )
        )

        if len(tasks) >= question_batch_size:
            responses = await asyncio.gather(*tasks)
            tasks = []
            skill_utils.update_results(responses, attribute_matrix, results)
            num_processed += len(responses)
            print(f"Completed searching {num_processed}/{len(questions)} questions...")

    if tasks:
        responses = await asyncio.gather(*tasks)
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

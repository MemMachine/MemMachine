# based on Edwin PR#651 on Dec 4
# modified to use restapiv2
# ruff: noqa: PTH118, RUF059, TRY400, G004, C901, PERF403, UP031

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass

import openai
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    function_tool,
    trace,
)
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

if True:
    # find path to other scripts and modules
    my_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir = os.path.abspath(os.path.join(my_dir, ".."))
    utils_dir = os.path.join(top_dir, "utils")
    sys.path.insert(1, utils_dir)
    sys.path.insert(1, top_dir)
    from memmachine_helper import MemmachineHelper


def convert_for_json(obj):
    """Recursively convert objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return {key: convert_for_json(value) for key, value in obj.__dict__.items()}
    if isinstance(obj, str):
        try:
            return convert_for_json(json.loads(obj))
        except Exception:
            return obj
    else:
        # For non-serializable types, convert to string
        return str(obj)


# from https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide
LOCOMO_INSTRUCTIONS1 = """
<context_gathering>
Goal: Get enough context fast. Parallelize discovery and stop as soon as you can act.

Method:
- Start broad, then fan out to focused subqueries.
- In parallel, launch varied queries; read top hits per query. Deduplicate paths and cache; don’t repeat queries.
- Avoid over searching for context. If needed, run targeted searches in one parallel batch.

Early stop criteria:
- You can name exact content to change.
- Top hits converge (~70%) on one area/path.

Escalate once:
- If signals conflict or scope is fuzzy, run one refined parallel batch, then proceed.

Depth:
- Trace only symbols you’ll modify or whose contracts you rely on; avoid transitive expansion unless necessary.

Loop:
- Batch search → minimal plan → complete task.
- Search again only if validation fails or new unknowns appear. Prefer acting over more searching.
</context_gathering>
<persistence>
- You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
- Only terminate your turn when you are sure that the problem is solved.
- Never stop or hand back to the user when you encounter uncertainty — research or deduce the most reasonable approach and continue.
- Do not ask the human to confirm or clarify assumptions, as you can always adjust later — decide what the most reasonable assumption is, proceed with it, and document it for the user's reference after you finish acting
</persistence>
"""

LOCOMO_INSTRUCTIONS2 = """
You are asked to answer a question based on your memories of a conversation.

<procedure>
1. First, the question has been used directly as a contextual cue to retrieve some relevant base memories.
2. Reason about the base memories to break down the question into sub-questions or identify specific details that need to be recalled.
3. Use these sub-questions and details to come up with new cues and follow-up questions to retrieve more memories using the retrieve_memories tool.
4. You may use the retrieve_memories tool as many times as necessary to retrieve all relevant memories.
5. Finally, synthesize all the retrieved memories to formulate a concise and accurate answer to the original question.
</procedure>

<guidelines>
1. Prioritize memories that answer the question directly. Be meticulous about identifying details.
2. When there may be multiple answers to the question, use effort to retrieve memories to list all possible answers. Do not become satisfied with just the first few answers retrieved.
3. When asked about time intervals or to count items, do not rush to answer immediately. Instead, carefully compute the answer.
4. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
5. The question may contain typos or be based on the asker's own unreliable memories. Do your best to answer the question using the most relevant information in your memories.
6. Your memories may include small or large jumps in time or context. Do not assume that the information is complete.
7. Your memories are ordered from earliest to latest for each time you retrieve more memories using the retrieve_memories tool.
8. Your final response to the question should be no more than a couple of sentences.
</guidelines>

<base>
{memories}
</base>

<question>
{question}
</question>
"""


@dataclass
class MmaiCtx:
    mmai: MemmachineHelper
    locomo_agent: Agent
    mem_type: str
    top_k: int
    conv_num: int
    q_num: int
    qa_info: dict
    stat: dict


def init_stat():
    stat = {
        "i_tokens": [],
        "o_tokens": [],
        "call_id": [],
        "cue": [],
        "ctx_usage": [],
        "search_secs": 0.0,
        "cb_count": 0,
        "cb_i_tokens": 0,
        "cb_o_tokens": 0,
        "agent_requests": 0,
        "agent_i_tokens": 0,
        "agent_o_tokens": 0,
        "answer": "",
    }
    return stat


def count_token_usage(stat):
    try:
        for i_tokens in stat["i_tokens"]:
            stat["cb_i_tokens"] += i_tokens
        for o_tokens in stat["o_tokens"]:
            stat["cb_o_tokens"] += o_tokens
        if not stat["agent_i_tokens"]:
            stat["agent_i_tokens"] = stat["cb_i_tokens"]
        if not stat["agent_o_tokens"]:
            stat["agent_o_tokens"] = stat["cb_o_tokens"]
    except Exception:
        pass


@function_tool(name_override="retrieve_memories")
async def retrieve_memories(wrapper: RunContextWrapper[MmaiCtx], cue: str) -> str:
    """
    Retrieve relevant memories based on the provided cue.
    The cue should be a complete question or phrase to help retrieve relevant memories.

    Args:
        cue (str): A cue used to retrieve relevant memories.
    """
    # agent called our tool to get context
    try:
        mmai = wrapper.context.mmai
        mem_type = wrapper.context.mem_type
        top_k = wrapper.context.top_k
        conv_num = wrapper.context.conv_num
        q_num = wrapper.context.q_num
        stat = wrapper.context.stat
        timeout = 30 + (3 * top_k)
        types = None
        if mem_type:
            mem_type = mem_type.lower()
            types = [mem_type]
        ctx = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        memory_start = time.time()
        mmai.log.debug(f"agent callback conv={conv_num} q={q_num} starts")
        data = await mmai.async_search_memory(
            cue, top_k=top_k, types=types, timeout=timeout
        )
        memory_end = time.time()
        memory_time = f"{memory_end - memory_start:.2f} secs."

        num_types, le_len, se_len, ss_len, sm_len = mmai.split_data_count(data)
        ctx_usage = f"le={le_len} se={se_len} ss={ss_len} sm={sm_len}"
        i_tokens = 0
        o_tokens = 0
        call_id = "UNKNOWN"
        if hasattr(wrapper, "usage"):
            i_tokens = wrapper.usage.input_tokens
            o_tokens = wrapper.usage.output_tokens
        else:
            print(
                f"rm:ERROR: agent callback conv={conv_num} q={q_num} get usage failed"
            )
        if hasattr(wrapper, "tool_call") and hasattr(wrapper.tool_call, "call_id"):
            call_id = f"call_id:{wrapper.tool_call.call_id}"
        elif hasattr(wrapper, "tool_call_id"):
            call_id = f"tool_call_id:{wrapper.tool_call_id}"
        stat["ctx_usage"].append(ctx_usage)
        stat["cb_count"] += 1
        stat["i_tokens"].append(i_tokens)
        stat["o_tokens"].append(o_tokens)
        stat["call_id"].append(call_id)
        stat["cue"].append(cue)
        stat["search_secs"] += memory_end - memory_start

        if mem_type == "episodic":
            ctx = mmai.build_episodic_ctx(data)
            if sm_len:
                print(
                    f"rm:ERROR: agent callback episodic memory search returned {sm_len} semantic memories "
                    f"conv={conv_num} q={q_num}"
                )
        else:
            ctx = mmai.build_ctx(data)
            if not sm_len:
                print(
                    f"rm:ERROR: agent callback semantic memory search returned no semantic memories "
                    f"conv={conv_num} q={q_num}"
                )
        mmai.log.debug(
            f"agent callback conv={conv_num} q={q_num} duration={memory_time}"
        )
        mmai.log.debug(
            f"agent callback conv={conv_num} q={q_num} completed stat={stat}"
        )
    except Exception as ex:
        print(
            f"rm:ERROR: agent callback failed conv={conv_num} question={q_num} ex={ex}"
        )
        print(f"rm:{traceback.format_exc()}")
        mmai.log.error(
            f"rm:ERROR: agent callback failed conv={conv_num} question={q_num} ex={ex}"
        )
        mmai.log.error(f"rm:{traceback.format_exc()}")
    return ctx


async def process_question(mmai_ctx):
    mmai = mmai_ctx.mmai
    locomo_agent = mmai_ctx.locomo_agent
    mem_type = mmai_ctx.mem_type
    top_k = mmai_ctx.top_k

    conv_num = mmai_ctx.conv_num
    q_num = mmai_ctx.q_num
    qa_info = mmai_ctx.qa_info
    stat = mmai_ctx.stat

    question = qa_info["question"]
    answer = qa_info.get("answer", "")
    category = qa_info["category"]
    evidence = qa_info["evidence"]
    adversarial_answer = qa_info.get("adversarial_answer", "")

    final_result = {
        "question": "",
        "locomo_answer": "",
        "model_answer": "",
        "category": 0,
        "evidence": "",
        "adversarial_answer": "",
        "agent_time": 0.0,
        "agent_trace": "",
    }
    ctx_usage = ""
    timeout = 30 + (3 * top_k)
    agent_start = 0.0
    agent_end = 0.0

    try:
        # create initial context to pass to agent
        types = None
        if mem_type:
            mem_type = mem_type.lower()
            types = [mem_type]
        ctx = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        memory_start = time.time()
        mmai.log.debug(f"search memory conv={conv_num} q={q_num} starts")
        data = await mmai.async_search_memory(
            question, top_k=top_k, types=types, timeout=timeout
        )
        memory_end = time.time()
        memory_time = f"{memory_end - memory_start:.2f} secs."

        num_types, le_len, se_len, ss_len, sm_len = mmai.split_data_count(data)
        ctx_usage = f"le={le_len} se={se_len} ss={ss_len} sm={sm_len}"
        stat["ctx_usage"].append(ctx_usage)
        stat["search_secs"] += memory_end - memory_start

        if mem_type == "episodic":
            ctx = mmai.build_episodic_ctx(data)
            if sm_len:
                print(
                    f"pc:ERROR: episodic memory search returned {sm_len} semantic memories "
                    f"conv={conv_num} q={q_num} ctx_usage={ctx_usage} duration={memory_time}"
                )
        else:
            ctx = mmai.build_ctx(data)
            if not sm_len:
                print(
                    f"pc:ERROR: semantic memory search returned no semantic memories "
                    f"conv={conv_num} q={q_num} ctx_usage={ctx_usage} duration={memory_time}"
                )
        mmai.log.debug(
            f"search memory conv={conv_num} q={q_num} initial ctx_usage={ctx_usage} duration={memory_time}"
        )
        prefetched_context = ctx
    except Exception as ex:
        print(f"pq:ERROR: search memory failed ex={ex}")
        print(f"pq:{traceback.format_exc()}")
        mmai.log.error(f"pq:ERROR: search memory failed ex={ex}")
        mmai.log.error(f"pq:{traceback.format_exc()}")
        return final_result

    try:
        msg = LOCOMO_INSTRUCTIONS2.format(
            memories=prefetched_context, question=question
        )
        agent_start = time.monotonic()

        with trace("locomo"):
            try:
                mmai.log.debug(f"search memory conv={conv_num} q={q_num} agent starts")
                run_result = await Runner.run(
                    starting_agent=locomo_agent,
                    input=msg,
                    max_turns=20,
                    context=mmai_ctx,
                )
                mmai.log.debug(
                    f"search memory conv={conv_num} q={q_num} agent completes"
                )
                run_usage = {}
                try:
                    run_context = run_result.context_wrapper
                    usage = run_context.usage
                    run_usage = {
                        "requests": usage.requests,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                except Exception:
                    pass

                agent_trace = [
                    {str(type(item).__name__): convert_for_json(item.raw_item)}
                    for item in run_result.new_items
                ]

                results = {
                    "response": run_result.final_output.strip(),
                    "trace": agent_trace,
                    "run_usage": run_usage,
                }
            except Exception as e:
                print(f"ERROR: agent runner failed ex={e}")
                traceback.print_exc()
                mmai.log.debug(
                    f"search memory conv={conv_num} q={q_num} agent error ex={e}"
                )
                results = {"response": "Error", "trace": "None", "run_usage": {}}

        agent_end = time.monotonic()

        result_response = results.get("response")
        run_usage = results.get("run_usage", {})
        count_token_usage(stat)
        stat["agent_requests"] = run_usage.get("requests", 0)
        stat["agent_i_tokens"] = run_usage.get("input_tokens", 0)
        stat["agent_o_tokens"] = run_usage.get("output_tokens", 0)
        stat["answer"] = result_response
        mmai.log.debug(
            f"search memory conv={conv_num} q={q_num} final run_usage={run_usage}"
        )
        mmai.log.debug(f"search memory conv={conv_num} q={q_num} final stat={stat}")
    except Exception as ex:
        print(f"pq:ERROR: agent loop failed conv={conv_num} question={q_num} ex={ex}")
        print(f"pq:{traceback.format_exc()}")
        mmai.log.error(
            f"pq:ERROR: agent loop failed conv={conv_num} question={q_num} ex={ex}"
        )
        mmai.log.error(f"pq:{traceback.format_exc()}")
        count_token_usage(stat)
        return final_result

    final_result = {
        "question": question,
        "locomo_answer": answer,
        "model_answer": results["response"],
        "category": category,
        "evidence": evidence,
        "adversarial_answer": adversarial_answer,
        "agent_time": agent_end - agent_start,
        "agent_trace": results["trace"],
        "run_usage": results["run_usage"],
    }
    return final_result


async def respond_question(mmai_ctx, semaphore):
    conv_num = mmai_ctx.conv_num
    q_num = mmai_ctx.q_num
    category = mmai_ctx.qa_info["category"]
    async with semaphore:
        response = await process_question(mmai_ctx)
        response["conv_num"] = conv_num
        response["question_num"] = q_num
    return category, response


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--conv-start", type=int, default=1, help="start at this conversation"
    )
    parser.add_argument(
        "--conv-stop", type=int, default=1, help="stop at this conversation"
    )
    parser.add_argument(
        "--top-k", type=int, default=30, help="return this many hints per question"
    )
    parser.add_argument("--mem-type", help="<episodic, semantic>, default is both")
    parser.add_argument(
        "--max-workers", type=int, default=10, help="number of simultaneous queries"
    )
    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    # agent code from https://openai.github.io/openai-agents-python/models/
    openai_agent_client = openai.AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    openai_agent_model = OpenAIChatCompletionsModel(
        model=openai_model_name,
        openai_client=openai_agent_client,
    )
    locomo_agent = Agent[MmaiCtx](
        name="agent",
        instructions=LOCOMO_INSTRUCTIONS1,
        model=openai_agent_model,
        model_settings=ModelSettings(max_tokens=2000, temperature=0.2, store=False),
        tools=[retrieve_memories],
    )

    mmai = MemmachineHelper.factory("restapiv2")
    health = mmai.get_health()
    print("mmai health:")
    for k, v in health.items():
        print(f"k={k} v={v}")

    metrics_before = mmai.get_metrics()
    with open(f"search_metrics_before_{os.getpid()}.json", "w") as fp:
        json.dump(metrics_before, fp, indent=4)

    print(f"top_k={args.top_k} max_workers={args.max_workers} mem_type={args.mem_type}")
    results = {}
    stats = {}
    semaphore = asyncio.Semaphore(args.max_workers)
    for idx, item in enumerate(locomo_data):
        # outer conversation loop
        conv_num = idx + 1
        if conv_num < args.conv_start or conv_num > args.conv_stop:
            continue

        if "conversation" not in item:
            continue

        print(f"Processing questions for conversation {conv_num}...")
        qa_list = item["qa"]
        response_tasks = []
        for q_idx, qa in enumerate(qa_list):
            # inner question loop
            # only do category 1 to 4, skip cat5 adversarial
            try:
                category = int(qa["category"])
                if category < 1 or category > 4:
                    continue
            except Exception:
                continue
            q_num = q_idx + 1
            stat = init_stat()
            mmai_ctx = MmaiCtx(
                mmai, locomo_agent, args.mem_type, args.top_k, conv_num, q_num, qa, stat
            )
            if conv_num not in stats:
                stats[conv_num] = {}
            stats[conv_num][q_num] = mmai_ctx
            task = respond_question(mmai_ctx, semaphore)
            response_tasks.append(task)

        # tasks created for all questions, now wait for responses with progress reporting
        responses = await tqdm_asyncio.gather(
            *response_tasks, desc=f"Conv {conv_num} answer questions"
        )

        for category, response in responses:
            if category not in results:
                results[category] = []
            results[category].append(response)

    # count token usage
    cb_requests = 0
    cb_i_tokens = 0
    cb_o_tokens = 0
    for conv_num, q_stats in stats.items():
        for q_num, mmai_ctx in q_stats.items():
            stat = mmai_ctx.stat
            count_token_usage(stat)
            cb_requests += stat["cb_count"] + 1
            cb_i_tokens += stat["agent_i_tokens"] + 1
            cb_o_tokens += stat["agent_o_tokens"] + 1

    # workaround intermittent crash
    # TypeError: Object of type set is not JSON serializable
    clean_results = {}
    categories = list(results.keys())
    categories = sorted(categories)
    print(f"save: validate results categories={categories}")
    num_items = 0
    num_errors = 0
    total_requests = 0
    total_i_tokens = 0
    total_o_tokens = 0
    for category in categories:
        clean_results[category] = []
        results_list = results[category]
        cat_num_items = 0
        cat_num_errors = 0
        for result_item in results_list:
            try:
                json.dumps(result_item)
                if not result_item["model_answer"]:
                    cat_num_errors += 1
                clean_results[category].append(result_item)
                if "run_usage" in result_item:
                    run_usage = result_item["run_usage"]
                    requests = run_usage["requests"]
                    i_tokens = run_usage["input_tokens"]
                    o_tokens = run_usage["output_tokens"]
                    total_requests += requests
                    total_i_tokens += i_tokens
                    total_o_tokens += o_tokens
            except Exception as ex:
                print(f"save:ERROR: json dump failed result={result_item} ex={ex}")
                cat_num_errors += 1
            cat_num_items += 1
        num_items += cat_num_items
        num_errors += cat_num_errors
        print(
            f"save: category={category} items={cat_num_items} errors={cat_num_errors}"
        )

    print(f"save: total items={num_items} errors={num_errors}")
    print(
        f"save: agent reported: "
        f"requests={total_requests} i_tokens={total_i_tokens} o_tokens={total_o_tokens}"
    )
    print(
        f"save: callback stats: "
        f"requests={cb_requests} i_tokens={cb_i_tokens} o_tokens={cb_o_tokens}"
    )

    with open(target_path, "w") as f:
        json.dump(clean_results, f, indent=4)

    metrics_after = mmai.get_metrics()
    with open(f"search_metrics_after_{os.getpid()}.json", "w") as fp:
        json.dump(metrics_after, fp, indent=4)
    metrics_delta = mmai.diff_metrics()
    new_delta = {}
    for k, v in metrics_delta.items():
        if not k.endswith("_created"):
            new_delta[k] = v  # remove timestamps
    metrics_filename = f"search_metrics_delta_{os.getpid()}.json"
    with open(metrics_filename, "w") as fp:
        json.dump(new_delta, fp, indent=4)
    i_tokens = 0
    o_tokens = 0
    e_tokens = 0
    rerank_queries = 0
    if "language_model_openai_usage_input_tokens_total" in metrics_delta:
        i_tokens = int(metrics_delta["language_model_openai_usage_input_tokens_total"])
    if "language_model_openai_usage_output_tokens_total" in metrics_delta:
        o_tokens = int(metrics_delta["language_model_openai_usage_output_tokens_total"])
    if "embedder_openai_usage_prompt_tokens_total" in metrics_delta:
        e_tokens = int(metrics_delta["embedder_openai_usage_prompt_tokens_total"])
    if "amazon_bedrock_reranker_score_calls_total" in metrics_delta:
        rerank_queries = int(metrics_delta["amazon_bedrock_reranker_score_calls_total"])
    tokens_str = (
        f"chat i_tokens={i_tokens} o_tokens={o_tokens} embedder tokens={e_tokens}"
    )
    print(f"save: memmachine {tokens_str} rerank_queries={rerank_queries}")
    vm_before = 0.0
    vm_after = 0.0
    rss_before = 0.0
    rss_after = 0.0
    if "process_virtual_memory_bytes" in metrics_before:
        vm_before = metrics_before["process_virtual_memory_bytes"]
    if "process_virtual_memory_bytes" in metrics_after:
        vm_after = metrics_after["process_virtual_memory_bytes"]
    if "process_resident_memory_bytes" in metrics_before:
        rss_before = metrics_before["process_resident_memory_bytes"]
    if "process_resident_memory_bytes" in metrics_after:
        rss_after = metrics_after["process_resident_memory_bytes"]
    vm_before /= 1073741824.0
    vm_after /= 1073741824.0
    rss_before /= 1073741824.0
    rss_after /= 1073741824.0
    mem_str = "VM_before "
    mem_str += "%8.4f GiB " % vm_before
    mem_str += "VM_after "
    mem_str += "%8.4f GiB " % vm_after
    mem_str += "RSS_before "
    mem_str += "%8.4f GiB " % rss_before
    mem_str += "RSS_after "
    mem_str += "%8.4f GiB " % rss_after
    print(f"save: memmachine memory {mem_str}")
    print(f"metrics_filename={metrics_filename}")


if __name__ == "__main__":
    load_dotenv()
    # load test params from env
    openai_api_key = os.getenv("OPENAI_API_KEY", "none")
    openai_api_base = os.getenv("OPENAI_API_BASE", None)
    openai_base_url = os.getenv("OPENAI_BASE_URL", None)
    openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    openai_no_think = os.getenv("OPENAI_NO_THINK")
    openai_strip_think = os.getenv("OPENAI_STRIP_THINK")
    openai_embedder_base = os.getenv("OPENAI_EMBEDDER_BASE", None)
    openai_embedder_name = os.getenv("OPENAI_EMBEDDER_NAME", "text-embedding-3-small")
    openai_embedder_dims = os.getenv("OPENAI_EMBEDDER_DIMS")
    openai_embedder_dims_use_default = os.getenv("OPENAI_EMBEDDER_DIMS_USE_DEFAULT")
    if not openai_api_base:
        openai_api_base = openai_base_url
    if not openai_embedder_base:
        openai_embedder_base = openai_api_base
    if openai_embedder_dims:
        openai_embedder_dims = int(openai_embedder_dims)

    if openai_api_base:
        os.environ["OPENAI_API_BASE"] = openai_api_base
        os.environ["OPENAI_BASE_URL"] = openai_api_base

    asyncio.run(main())

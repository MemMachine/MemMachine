# ruff: noqa: PTH118, C901, G004, PERF403, RUF059, UP031
import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

if True:
    # find path to other scripts and modules
    my_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir = os.path.abspath(os.path.join(my_dir, ".."))
    utils_dir = os.path.join(top_dir, "utils")
    sys.path.insert(1, utils_dir)
    sys.path.insert(1, top_dir)
    from memmachine_helper import MemmachineHelper


language = "english"
stop_words = stopwords.words(language)


def default_tokenize(text: str) -> list[str]:
    """
    Preprocess the input text
    by removing non-alphanumeric characters,
    converting to lowercase,
    word-tokenizing,
    and removing stop words.

    Args:
        text (str): The input text to preprocess.

    Returns:
        list[str]: A list of tokens for use in BM25 scoring.
    """
    alphanumeric_text = re.sub(r"\W+", " ", text)
    lower_text = alphanumeric_text.lower()
    words = word_tokenize(lower_text, language)
    tokens = [word for word in words if word and word not in stop_words]
    return tokens


ANSWER_PROMPT = """
You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Prioritize memories that answer the question directly. Be meticulous about recalling details.
2. When there may be multiple answers to the question, think hard to remember and list all possible answers. Do not become satisfied with just the first few answers you remember.
3. When asked about time intervals or to count items, do not rush to answer immediately. Instead, carefully enumerate the items or subtract the times using numbers.
4. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
5. The question may contain typos or be based on the asker's own unreliable memories. Do your best to answer the question using the most relevant information in your memories.
6. Your memories may include small or large jumps in time or context. You are not confused by this. You just did not bother to remember everything in between.
7. Your memories are ordered from earliest to latest.
</instructions>

<memories>
{memories}
</memories>

Question: {question}
Your short response to the question without fluff (no more than a couple of sentences):
"""


def process_question(
    mem_type,
    top_k,
    mmai,
    model,
    question,
    answer,
    category,
    evidence,
    adversarial_answer,
    conv_num,
    q_num,
):
    # TOM1
    result = {
        "question": "",
        "locomo_answer": "",
        "model_answer": "",
        "category": 0,
        "evidence": "",
        "adversarial_answer": "",
        "conversation_memories": "",
    }
    ctx_usage = ""
    timeout = 30 + (3 * top_k)
    try:
        types = None
        if mem_type:
            mem_type = mem_type.lower()
            types = [mem_type]
        memory_start = time.time()
        mmai.log.debug(f"search memory conv={conv_num} q={q_num}")
        data = mmai.search_memory(question, top_k=top_k, types=types, timeout=timeout)
        num_types, le_len, se_len, ss_len, sm_len = mmai.split_data_count(data)
        ctx_usage = f"le={le_len} se={se_len} ss={ss_len} sm={sm_len}"
        if mem_type == "episodic":
            ctx = mmai.build_episodic_ctx(data)
            if sm_len:
                print(
                    f"pc:ERROR: episodic memory search returned {sm_len} semantic memories "
                    f"conv={conv_num} q={q_num}"
                )
        else:
            ctx = mmai.build_ctx(data)
            if not sm_len:
                print(
                    f"pc:ERROR: semantic memory search returned no semantic memories "
                    f"conv={conv_num} q={q_num}"
                )
        memory_end = time.time()
    except Exception as ex:
        print(f"pc:ERROR: memory search failed ex={ex}")
        print(f"pc:{traceback.format_exc()}")
        return result

    prompt = ANSWER_PROMPT.format(memories=ctx, question=question)

    run_usage = {}
    try:
        llm_start = time.time()
        rsp = model.responses.create(
            model=openai_model_name,
            max_output_tokens=4096,
            temperature=0.0,
            top_p=1,
            input=[{"role": "user", "content": prompt}],
        )
        llm_end = time.time()
        # TOM1 get token usage
        run_usage = {}
        try:
            usage = rsp.usage
            run_usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
        except Exception:
            pass
    except Exception as ex:
        print(f"pc:ERROR: chat failed conv={conv_num} question={q_num} ex={ex}")
        print(f"pc:{traceback.format_exc()}")
        return result

    rsp_text = rsp.output_text

    print(
        f"conv={conv_num} question={q_num} category={category}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Response: {rsp_text}\n"
        f"Ctx usage: {ctx_usage}\n"
        f"Memory retrieval time: {memory_end - memory_start:.2f} seconds\n"
        f"LLM response time: {llm_end - llm_start:.2f} seconds\n"
        f"run_usage: {run_usage}\n\n"
    )
    # TOM1 too much to print f"MEMORIES START\n{formatted_context}MEMORIES END\n"
    result = {
        "question": question,
        "locomo_answer": answer,
        "model_answer": rsp_text,
        "category": category,
        "evidence": evidence,
        "adversarial_answer": adversarial_answer,
        "conversation_memories": ctx,
        "run_usage": run_usage,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    # TOM1
    parser.add_argument(
        "--conv-start", type=int, default=1, help="start at this conversation"
    )
    parser.add_argument(
        "--conv-stop", type=int, default=1, help="stop at this conversation"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="return this many hints per question"
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

    # TOM1
    openai_eval_chat_client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    mmai = MemmachineHelper.factory("restapiv2")
    health = mmai.get_health()
    print("mmai health:")
    for k, v in health.items():
        print(f"k={k} v={v}")

    metrics_before = mmai.get_metrics()
    with open(f"search_metrics_before_{os.getpid()}.json", "w") as fp:
        json.dump(metrics_before, fp, indent=4)

    print(f"top_k={args.top_k} max_workers={args.max_workers}")
    results = {}
    for idx, item in enumerate(locomo_data):
        # TOM1
        conv_num = idx + 1
        if conv_num < args.conv_start or conv_num > args.conv_stop:
            continue

        if "conversation" not in item:
            continue

        qa_list = item["qa"]

        print(f"Processing questions for group {idx}...")

        def respond_question(qa, conv_num, q_num):
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa["evidence"]

            adversarial_answer = qa.get("adversarial_answer", "")

            question_response = process_question(
                args.mem_type,
                args.top_k,
                mmai,
                openai_eval_chat_client,
                question,
                answer,
                category,
                evidence,
                adversarial_answer,
                conv_num,
                q_num,
            )
            question_response["conv_num"] = conv_num
            question_response["question_num"] = q_num
            return (
                category,
                question_response,
            )

        responses = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # submit tasks
            futures = []
            for q_idx, qa in enumerate(qa_list):
                # TOM1 only do category 1 to 4, skip cat5 adversarial
                try:
                    category = int(qa["category"])
                    if category < 1 or category > 4:
                        continue
                except Exception:
                    continue
                future = executor.submit(respond_question, qa, idx + 1, q_idx + 1)
                futures.append(future)
            # wait for task completion
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Answering Questions"
            ):
                result = future.result()
                responses.append(result)

        for category, response in responses:
            if category not in results:
                results[category] = []
            results[category].append(response)

    # TOM1 workaround intermittent crash
    # TypeError: Object of type set is not JSON serializable
    clean_results = {}
    categories = list(results.keys())
    categories = sorted(categories)
    print(f"save:TOM1: validate results categories={categories}")
    num_items = 0
    num_errors = 0
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
                    if run_usage:
                        i_tokens = run_usage["input_tokens"]
                        o_tokens = run_usage["output_tokens"]
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

    print(
        f"save: total items={num_items} errors={num_errors} "
        f"i_tokens={total_i_tokens} o_tokens={total_o_tokens}"
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
    print(f"metrics_filename={metrics_filename}")
    i_tokens = 0
    o_tokens = 0
    e_tokens = 0
    if "language_model_openai_usage_input_tokens_total" in metrics_delta:
        i_tokens = int(metrics_delta["language_model_openai_usage_input_tokens_total"])
    if "language_model_openai_usage_output_tokens_total" in metrics_delta:
        o_tokens = int(metrics_delta["language_model_openai_usage_output_tokens_total"])
    if "embedder_openai_usage_prompt_tokens_total" in metrics_delta:
        e_tokens = int(metrics_delta["embedder_openai_usage_prompt_tokens_total"])
    tokens_str = (
        f"chat i_tokens={i_tokens} o_tokens={o_tokens} embedder tokens={e_tokens}"
    )
    print(f"save: memmachine {tokens_str}")
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


if __name__ == "__main__":
    load_dotenv()
    # TOM1
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

    main()

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, cast

from dotenv import load_dotenv
from openai import AsyncOpenAI

from memmachine.episodic_memory.episodic_memory import EpisodicMemory
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
)
from memmachine.knowledge_graph.re_gpt4_1 import KnowledgeGraph
from memmachine.common.vector_graph_store import Node

# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/prompts.py).
# It is modified to work with MemMachine.
ANSWER_PROMPT = """
You are an analytical AI that reasons deeply about context before answering questions. Your task is to:

1. FIRST: Look for direct, explicit answers in the context
2. ANALYZE the context thoroughly for relevant information
3. IDENTIFY patterns, connections, and implications 
4. REASON about what the context suggests or implies
5. ANSWER based on direct evidence OR analysis

<reasoning>
- Scan through ALL episodes and facts completely before answering
- Look for every explicit statement that relates to the question
- NEVER stop after finding the first answer - continue scanning for more
- When asking "what did X show Y", look for ALL items X showed Y on that date
- Collect multiple items, events, or details that answer the same question
- If not found directly, identify all context elements related to the question
- Look for patterns, themes, and implicit information in the context
- Consider what the context suggests beyond explicit statements
- Note any contradictions or missing information that affects the answer
- Pay close attention to temporal information and dates (validAt timestamps)
- For time-sensitive questions, prioritize more recent information
- Consider the chronological sequence of events when relevant
- CRITICAL: Ensure completeness by including ALL relevant items found
- If you find 2+ items for the same question, mention them all in your answer
- Be precise with details (specific types, colors, descriptions when available)
- Draw logical conclusions based on available evidence
- Don't give reasoning in the output
</reasoning>

**Output Format** (JSON dict, don't give the JSON with ```json):
{"answer" : "Your direct, short(max 2 sentences) answer based on your analysis"}
"""

USER_PROMPT = """<context>
${context}
</context>

<question>
Question: ${question}
</question>
"""

WIKI_ANSWER_PROMPT="""
    You are an intelligent memory assistant tasked with retrieving accurate information from context memories to answer a question.

    # CONTEXT:
    You have access to memories of contexts. These memories contain
    information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. If the question asks about a specific event or fact, look for direct evidence in the memories
    3. If the memories contain contradictory information, prioritize the most recent memory
    4. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the context. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    5. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    6. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question, pay attention to FACTS section if available
    2. Examine the content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    <MEMORIES>

    {conversation_memories}

    </MEMORIES>

    Question: {question}

    Answer:
    """


def format_memory(episodes, summary, kg_episodes, fmt, include_timestamp: bool = True) -> str:
    kg_final = []
    uuids = set()

    episode_nodes = []
    for e in episodes:
        episode_nodes.append(Node(
            uuid=e.uuid,
            properties={
                "timestamp": datetime.strptime(e.user_metadata['source_timestamp'], fmt),
                "content": f"{e.user_metadata['source_speaker']}: {e.content}{f' [Image Caption: {e.user_metadata["blip_caption"]}]' if e.user_metadata.get('blip_caption') else ''}"
            },
        ))
        uuids.add(e.uuid)

    num_filtered = 0
    for e in kg_episodes:
        if e.uuid in uuids:
            num_filtered += 1
            continue
        # Drop microseconds and timezone info for consistency
        ts_str = e.properties['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        episode_nodes.append(Node(
            uuid=e.uuid,
            properties={
                "timestamp": datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"),
                "content": e.properties['content']
            },
        ))
    # print(f"Filtered out {num_filtered} duplicate episodes from KG results")

    final = sorted(episode_nodes, key=lambda x: x.properties["timestamp"])

    episode_context = (
        # "<LONG TERM MEMORY EPISODES>\n"
        # + "\n".join(
        #     [
        #         f"[{episode.properties['timestamp']}] {episode.properties['content']}"
        #         for episode in kg_episodes
        #     ]
        # )
        # + "\n".join(
        #     [
        #         f"[{episode.user_metadata['source_timestamp']}] {episode.user_metadata['source_speaker']}: {episode.content}{f' [ATTACHED: {episode.user_metadata["blip_caption"]}]' if episode.user_metadata.get('blip_caption') else ''}"
        #         for episode in episodes
        #     ]
        # )
        # + "\n</LONG TERM MEMORY EPISODES>\n"
        "<LONG TERM MEMORY EPISODES>\n"
        + "\n".join(
            [
                f"[{episode.properties['timestamp']}] {episode.properties['content']}" if include_timestamp else f"{episode.properties['content']}"
                for episode in final 
            ]
        )
        + "\n</LONG TERM MEMORY EPISODES>"
    )
    summary_context = (
        f"<WORKING MEMORY SUMMARY>\n{summary}\n</WORKING MEMORY SUMMARY>"
        if summary
        else ""
    )
    # return episode_context + "\n" + summary_context
    return episode_context

def get_bedrock_reranker():
    import boto3
    from memmachine.common.reranker.amazon_bedrock_reranker import (
        AmazonBedrockReranker,
        AmazonBedrockRerankerParams,
    )

    region = "us-west-2"

    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        aws_access_key_id="key_id",
        aws_secret_access_key="key",
    )

    return AmazonBedrockReranker(
        AmazonBedrockRerankerParams(
            client=client,
            region=region,
            model_id="amazon.rerank-v1:0"
        )
    )

async def process_question(
    memory_manager: EpisodicMemoryManager,
    model: AsyncOpenAI,
    group_id,
    user,
    question,
    answer,
    category,
    evidence,
    adversarial_answer,
    limit,
):
    memory_start = time.time()
    memory = cast(
        EpisodicMemory,
        await memory_manager.get_episodic_memory_instance(
            group_id=group_id,
            session_id=group_id,
            user_id=[user],
        ),
    )

    (
        short_term_episodes,
        long_term_episodes,
        summaries,
        kg_episodes,
    ) = await memory.query_memory(query=question, limit=limit)

    episodes = long_term_episodes + short_term_episodes
    summary = summaries[0] if summaries else ""
    memory_end = time.time()

    return episodes, summary

    # llm_start = time.time()
    # rsp = await model.responses.create(
    #     model="gpt-4o-mini",
    #     max_output_tokens=4096,
    #     temperature=0.0,
    #     top_p=1,
    #     input=[{"role": "user", "content": prompt}],
    # )
    # llm_end = time.time()

    # rsp_text = rsp.output_text

    # print_info = (
    #     f"Question: {question}\n"
    #     f"Answer: {answer}\n"
    #     f"Response: {rsp_text}\n"
    #     f"Memory retrieval time: {memory_end - memory_start:.2f} seconds\n"
    #     f"LLM response time: {llm_end - llm_start:.2f} seconds\n"
    #     f"MEMORIES START\n{formatted_context}MEMORIES END\n"
    # )

    # return {
    #     "question": question,
    #     "locomo_answer": answer,
    #     "model_answer": rsp_text,
    #     "category": category,
    #     "evidence": evidence,
    #     "adversarial_answer": adversarial_answer,
    #     "conversation_memories": formatted_context,
    #     "print_info": print_info,
    # }

async def get_model_answer(
    model: AsyncOpenAI,
    group_id,
    user,
    qa,
    formatted_context,
    answer_prompt=USER_PROMPT,
    perf_matrix={},
):
    question = qa["question"]
    answer = qa.get("answer", "")
    category = qa["category"]
    evidence = qa.get("evidence", "")

    adversarial_answer = qa.get("adversarial_answer", "")

    prompt = answer_prompt.format(
        context=formatted_context, question=question
    )
    
    
    llm_start = time.time()
    rsp = await model.responses.create(
        model="gpt-4.1-mini",
        max_output_tokens=4096,
        temperature=0.0,
        top_p=1,
        input=[
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    llm_end = time.time()

    # Remove leading <output>\n and trailing \n</output> from rsp.output_text
    output_text = ""
    for line in rsp.output_text.split("\n"):
        if line == "```json" or line == "```":
            continue
        output_text += line + "\n"

    rsp_dict = {}
    try:
        if output_text.startswith("{"):
            rsp_dict = json.loads(output_text)
        else:
            rsp_dict = {"answer": output_text}
            print(f"WARNING: LLM response is not JSON:\n{rsp.output_text}\nUsing the string directly:\n{output_text}")
    except Exception as e:
        print(f"Parse LLM response\n:{output_text}\ngot error: {e}")
        raise e
    rsp_text = rsp_dict["answer"]

    print_info = (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Response: {rsp_text}\n"
        # f"Memory retrieval time: {memory_end - memory_start:.2f} seconds\n"
        f"LLM response time: {llm_end - llm_start:.2f} seconds\n"
        f"MEMORIES START\n{formatted_context}MEMORIES END\n"
    )

    question_response = {
        "question": question,
        "locomo_answer": answer,
        "model_answer": rsp_text,
        "category": category,
        "evidence": evidence,
        "evidence_text": qa["evidence_text"],
    }

    for key, value in perf_matrix.items():
        question_response[key] = value

    question_response["conversation_memories"] = formatted_context

    return (
        category,
        question_response,
    )

def load_data(
    start_line: int = 1,
    num_cases: int = 100,
    randomize: bool = True,
):
    # dataset = "/home/tomz/MemMachine/evaluation/locomo/wikimultihop.json"
    # dataset = "/home/tomz/MemMachine/evaluation/locomo/wiki-filter-gpt-4o-mini.json"
    dataset = "/home/tomz/MemMachine/evaluation/locomo/wiki-filter-gpt-4.1.json"
    print(f"Loading data from {dataset} line {start_line} to {num_cases}, randomize={randomize}")
    contexts = []
    supporting_facts = []
    questions = []
    answers = []
    types = []
    i = 1
    with open(dataset, "r", encoding="utf-8") as f:
        for line in f:
            if i < start_line:
                i += 1
                continue
            if i > num_cases:
                break

            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["context"] = json.loads(obj["context"])
            c_list = []
            key_to_sentences = {}
            for key, sentences in obj["context"]:
                key_to_sentences[key] = sentences
                for s in sentences:
                    c = f"{key}: {s}"
                    if randomize:
                        insert_index = random.randrange(len(contexts) + 1)  # 0..len inclusive
                        c_list.insert(insert_index, c)
                    else:
                        c_list.append(c)
            contexts.append(c_list)
            questions.append(obj["question"])
            answers.append(obj["answer"])
            types.append(obj["type"])
            golden_facts = json.loads(obj["supporting_facts"])
            fact_sents = []
            for fact in golden_facts:
                key = fact[0]
                sentence_idx = int(fact[1])
                fact_sents.append(key_to_sentences[key][sentence_idx])
            supporting_facts.append(fact_sents)
                
            i += 1
    return contexts, questions, answers, types, supporting_facts

async def search_wikimultihop(
    target_path: str,
    limit: int,
):
    memory_manager = EpisodicMemoryManager.create_episodic_memory_manager(
        "locomo_config.yaml"
    )

    model = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=600,
    )

    reranker = get_bedrock_reranker()

    em_total_time = 0.0
    kg_total_time = 0.0
    answer_total_time = 0.0

    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:9999",
        auth=(
            "neo4j",
            "password",
        ),
        connection_timeout=3600,
        connection_acquisition_timeout=3600,
        max_transaction_retry_time=3600,
        max_connection_lifetime=3600,
        keep_alive=True,
    )

    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import Neo4jVectorGraphStore, Neo4jVectorGraphStoreParams
    store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=driver,
            max_concurrent_transactions=200,
            force_exact_similarity_search=False,
        )
    )

    await store.create_fulltext_index()

    from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder(
        {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    )

    contexts, questions, answers, types, supporting_facts = load_data(start_line=1, num_cases=250, randomize=False)

    import uuid
    num_batch = 30
    kg_tasks = []
    em_tasks = []
    kg_episode_list = []
    triple_list = []
    l_episode_list = []
    summaries = []
    total_facts = 0
    hit_facts = 0
    ts = datetime.now()
    for c_list, q, a, t in zip(contexts, questions, answers, types):
        # ep_list = []
        # for c in c_list:
        #     ep_list.append(
        #         Node(
        #             uuid=uuid.uuid4(),
        #             properties={
        #                 "timestamp": ts + timedelta(seconds=len(ep_list)),
        #                 "content": c,
        #             }
        #         )
        #     )
        # kg_episode_list.append(ep_list)
        # triple_list.append([])
        # l_episode_list.append([])
        # summaries.append("")
        # ====== baseline above ======

        kg_tasks.append(
            search_kg(reranker, model, driver, store, embedder, q, 1, limit=limit)
        )

        em_tasks.append(
            process_question(
                memory_manager,
                model,
                "1",
                "user",
                q,
                a,
                t,
                "",
                "",
                limit=limit,
            )
        )

        if len(kg_tasks) >= num_batch or (q == questions[-1]):
            ts = time.perf_counter()
            print(f"Async gathering {len(kg_tasks)} KG tasks...")
            r_kg = await asyncio.gather(*kg_tasks)
            # r_kg = [([], []) for _ in kg_tasks]  # IGNORE KG
            for t_list, e_list in r_kg:
                kg_episode_list.append(e_list)
                triple_list.append(t_list)
            print(f"Gathered {len(kg_tasks)} KG tasks in {time.perf_counter() - ts:.2f}s")
            kg_total_time += time.perf_counter() - ts
            kg_tasks = []

            ts = time.perf_counter()
            print(f"Async gathering {len(em_tasks)} EM tasks...")
            # r_em = await asyncio.gather(*em_tasks)
            r_em = [([], "") for _ in em_tasks]  # IGNORE EM
            for episodes, summary in r_em:
                l_episode_list.append(episodes)
                summaries.append(summary)
            print(f"Gathered {len(em_tasks)} EM tasks in {time.perf_counter() - ts:.2f}s")
            em_total_time += time.perf_counter() - ts
            em_tasks = []
            
            print(f"Processed {len(kg_episode_list) if len(kg_episode_list) != 0 else len(l_episode_list)} / {len(questions)} questions.")

    # Run all response generation in parallel for current conversation
    r_batch = 50
    r_tasks = []
    responses = []
    for (
        l_episodes,
        summary,
        triple_texts,
        kg_episodes,
        q,
        a,
        t,
        facts
    ) in zip(l_episode_list,
            summaries,
            triple_list,
            kg_episode_list,
            questions,
            answers,
            types,
            supporting_facts):
        formatted_context = format_memory(l_episodes, summary, triple_texts, kg_episodes, fmt="%Y-%m-%d %H:%M:%S", include_timestamp=False)
        qa = {
            "question": q,
            "answer": a,
            "category": t,
        }

        total_facts += len(facts)
        for fact in facts:
            if fact in formatted_context:
                hit_facts += 1

        r_tasks.append(
            get_model_answer(
                model,
                "1",
                "1",
                qa,
                formatted_context,
                answer_prompt=WIKI_ANSWER_PROMPT,
            )
        )
        if len(r_tasks) >= r_batch or (q == questions[-1]):
            ts = time.perf_counter()
            print(f"Async gathering {len(r_tasks)} response tasks...")
            responses.extend(await asyncio.gather(*r_tasks))
            print(f"Gathered {len(r_tasks)} response tasks in {time.perf_counter() - ts:.2f}s, {len(responses)}/{len(questions)} total.")
            answer_total_time += time.perf_counter() - ts
            r_tasks = []

    results: dict[str, Any] = {}
    for category, response in responses:
        # print(f"---\n{response["print_info"][:300]} ---\n")
        category_result = results.get(category, [])
        category_result.append(response)
        results[category] = category_result

    print(f"Total Episodic Memory retrieval time: {em_total_time:.2f} seconds")
    print(f"Total Knowledge Graph retrieval time: {kg_total_time:.2f} seconds")
    print(f"Total Answer generation time: {answer_total_time:.2f} seconds")
    print(f"Recall of golden sentences: {hit_facts}/{total_facts} = {hit_facts/total_facts*100:.2f}%")
    with open(target_path, "a") as f:
        json.dump(results, f, indent=4)

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--limit", required=False, help="Path to the target data file"
    )

    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path
    limit = int(args.limit) if args.limit else 30

    # await search_wikimultihop(target_path, limit)
    # return

    memory_manager = EpisodicMemoryManager.create_episodic_memory_manager(
        "locomo_config.yaml"
    )

    model = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=600,
    )

    em_total_time = 0.0
    kg_total_time = 0.0
    answer_total_time = 0.0
    total_input_tolens = 0
    total_output_tokens = 0
    total_num_sufficiency_checks = 0

    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:9999",
        auth=(
            "neo4j",
            "password",
        ),
        connection_timeout=3600,                    # seconds
        # how long to wait for a pooled connection
        connection_acquisition_timeout=3600,        # seconds
        # built-in retry window for transient failures
        max_transaction_retry_time=3600,
        max_connection_lifetime=3600,
        keep_alive=True,
    )

    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import Neo4jVectorGraphStore, Neo4jVectorGraphStoreParams
    store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=driver,
            max_concurrent_transactions=200,
            force_exact_similarity_search=False,
        )
    )

    await store.create_fulltext_index()

    from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder(
        {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    )

    reranker = get_bedrock_reranker()

    model_name = "gpt-4.1-mini"
    print(f"Using KnowledgeGraph with model {model_name}...")
    kg = KnowledgeGraph(
        model_name=model_name,
        model=model,
        embedder=embedder,
        store=store,
        reranker=reranker,
    )

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    recall_hit = 0
    num_facts = 0
    num_episodes_returned = 0
    num_used_kg = 0
    num_used_longterm_only = 0
    num_used_both = 0
    num_questions = 0
    num_processed = 0
    skip_to_index = 0
    run_until_index = 20
    results: dict[str, Any] = {}
    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            continue
        
        if num_processed < skip_to_index:
            num_processed += 1
            continue

        print(f"Processing questions for group {idx}...")

        conversation = item["conversation"]
        user = conversation["speaker_a"]

        break_session = 0
        break_sentence = 0
        full_text = ""
        evidence_to_text = {}
        session_idx = 0
        num_msg = 0
        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"
            if session_id not in conversation:
                break

            session = conversation[session_id]
            session_date_time = conversation[f"{session_id}_date_time"]

            for message in session:
                num_msg += 1
                speaker = message["speaker"]
                dia_id = message["dia_id"]
                text = message["text"]
                img_url = message.get("img_url")
                blip_caption = message.get("blip_caption")

                text = f"{speaker}: {text}"
                # if img_url:
                #     text += f" [Image URL: {img_url}]\n"
                # if blip_caption:
                #     text += f" [Image Caption: {blip_caption}]\n"

                evidence_to_text[dia_id] = text
                full_text += text
                if num_msg == 300:
                    session, sentence = dia_id.split(":")
                    break_session = int(session[1:])
                    break_sentence = int(sentence)
                    # print(f"Breaking at dia_id {dia_id}, session {break_session}, sentence {break_sentence}")
                    break

        qas = item["qa"]
        qa_list = []
        for qa in qas:
            # if qa["category"] != 3:
            #     continue
            if qa["category"] == 5:
                continue
            ev_ids = []
            for ev in qa["evidence"]:
                if "," in ev:
                    ids = ev.split(",")
                    ev_ids.extend(ids)
                elif ";" in ev:
                    ids = ev.split(";")
                    ev_ids.extend(ids)
                else:
                    ev_ids.append(ev)

            if len(ev_ids) == 0:
                continue
            qa_list.append(qa)

        print(f"Testing on {len(qa_list)} questions.")

        group_id = f"group_{idx}"

        # qa_list = [{
        #     "question": "When did Melanie go camping in June?",
        #     "answer": "The week before 27 June 2023",
        #     "category": "2",
        #     "evidence": ["D4:6"],
        # }]
        # group_id = "group_0"

        # qa_list = qa_list[50:55]

        async def respond_question(qa, limit):
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa["evidence"]

            adversarial_answer = qa.get("adversarial_answer", "")

            return await process_question(
                memory_manager,
                model,
                group_id,
                user,
                question,
                answer,
                category,
                evidence,
                adversarial_answer,
                limit=limit,
            )

        responses = []
        em_converted_list = []
        summaries = []
        contexts_tasks = []
        em_batch = 50
        # Get EM responses first
        for qa in qa_list:
            # Always limit higher, rerank and truncate later
            contexts_tasks.append(respond_question(qa, limit*3))
            if len(contexts_tasks) >= em_batch or qa['question'] == qa_list[-1]['question']:
                ts = time.perf_counter()
                print(f"Async gathering {len(contexts_tasks)} contexts tasks...")
                r = await asyncio.gather(*contexts_tasks)
                # r = [([], "") for _ in contexts_tasks]  # IGNORE EM
                for episodes, summary in r:
                    em_convert = []
                    # Convert EM episodes to KG episodes
                    for e in episodes:
                        em_convert.append(Node(
                            uuid=e.uuid,
                            properties={
                                "timestamp": datetime.strptime(e.user_metadata['source_timestamp'], "%I:%M %p on %d %B, %Y"),
                                "content": f"{e.user_metadata['source_speaker']}: {e.content}{f' [Image Caption: {e.user_metadata["blip_caption"]}]' if e.user_metadata.get('blip_caption') else ''}"
                            },
                        ))
                    em_converted_list.append(em_convert)
                    summaries.append(summary)
                print(f"Gathered {len(contexts_tasks)} contexts tasks in {time.perf_counter() - ts:.2f}s")
                contexts_tasks = []
                em_total_time += time.perf_counter() - ts

        # # Read em_converted_list from binary file to reuse
        # import pickle
        # with open(f"em_converted_group_{idx}.pkl", "rb") as f:
        #     em_converted_list = pickle.load(f)
        
        suff_tasks = []
        for em_convert, qa in zip(em_converted_list, qa_list):
            suff_tasks.append(kg.check_sufficiency_batch(em_convert, [], qa["question"]))
            total_num_sufficiency_checks += len(em_convert) // limit + (1 if len(em_convert) % limit != 0 else 0)
        
        print(f"Async gathering {len(suff_tasks)} sufficiency check tasks for EM...")
        t = time.perf_counter()
        r_suff = await asyncio.gather(*suff_tasks)
        print(f"Gathered {len(suff_tasks)} sufficiency check tasks in {time.perf_counter() - t:.2f}s")

        # # Dump em_converted_list as binary file for reuse
        # import pickle
        # with open(f"em_converted_group_{idx}.pkl", "wb") as f:
        #     pickle.dump(em_converted_list, f)\

        # # Dump r_suff as binary file for reuse
        # import pickle
        # with open(f"r_suff_group_{idx}.pkl", "wb") as f:
        #     pickle.dump(r_suff, f)


        # # USE EM ONLY
        # r_suff = [(True, [], [], "", 0, 0) for em_convert in em_converted_list]

        # Get KG response only if EM is insufficient
        kg_batch = 50
        kg_tasks = []
        cur_real_kg_task = 0
        perf_list = []
        result_episodes_list = []
        em_suff_list = []
        kg_suff_list = []
        for (is_em_sufficient, sorted_suff_em_episodes, possible_relevant, reasoning_str, itoken, otoken), qa, em_convert in zip(r_suff, qa_list, em_converted_list):
            total_input_tolens += itoken
            total_output_tokens += otoken
            em_suff_list.append(is_em_sufficient)

            # if EM == True, skip KG and return EM
            # if EM == False, search KG and return filtered EM + KG
            if is_em_sufficient:
                res_e_list = em_convert
                if len(em_convert) > limit:
                    cohere_res = await kg.cohere_rerank(em_convert, score_threshold=0.0, query=qa["question"], limit=limit)
                    res_e_list = [e for e, _ in cohere_res]
                kg_tasks.append(
                    asyncio.sleep(0, result=(res_e_list, {"used_em": True, "used_kg": False, "reasoning": reasoning_str}, False))
                )
            else:
                possible = sorted_suff_em_episodes + list(possible_relevant)
                kg_tasks.append(
                    kg.search(query=qa["question"], possible_episodes=possible, session_id=group_id, limit=limit)
                )
                # if len(em_convert) > limit:
                #     cohere_res = await kg.cohere_rerank(em_convert, score_threshold=0.0, query=qa["question"], limit=limit)
                #     res_e_list = [e for e, _ in cohere_res]
                # kg_tasks.append(
                #     asyncio.sleep(0, result=(res_e_list, {"reasoning": reasoning_str}, False))
                # )
                num_used_kg += 1
                cur_real_kg_task += 1
            
            if cur_real_kg_task >= kg_batch or qa['question'] == qa_list[-1]['question']:
                print(f"Async gathering {cur_real_kg_task} KG tasks...")
                ts = time.perf_counter()
                r = await asyncio.gather(*kg_tasks)
                kg_total_time += time.perf_counter() - ts
                print(f"Gathered {cur_real_kg_task} KG tasks in {time.perf_counter() - ts:.2f}s")

                for e_list, perf_matrix, is_kg_sufficient in r:
                    result_episodes_list.append(e_list)
                    perf_list.append(perf_matrix)
                    kg_suff_list.append(is_kg_sufficient)
                    total_input_tolens += perf_matrix.get("num_llm_input_tokens", 0)
                    total_output_tokens += perf_matrix.get("num_llm_output_tokens", 0)
                    total_num_sufficiency_checks += perf_matrix.get("num_sufficiency_checks", 0)
                kg_tasks = []
                cur_real_kg_task = 0

        # Run all response generation in parallel for current conversation
        r_tasks = []
        for qa, result_episodes_kg_formatted, perf_matrix in zip(qa_list, result_episodes_list, perf_list):
            num_questions += 1
            evidence_strs = []

            num_episodes_returned += len(result_episodes_kg_formatted)
            formatted_context = format_memory([], None, result_episodes_kg_formatted, fmt="%Y-%m-%d %H:%M:%S")

            num_cur_facts = 0
            num_cur_hits = 0
            for ev in qa["evidence"]:
                if "," in ev:
                    ids = ev.split(",")
                    for id in ids:
                        num_cur_facts += 1
                        evidence_strs.append(evidence_to_text.get(id.strip(), ""))
                        if evidence_strs[-1] in formatted_context:
                            num_cur_hits += 1
                elif ";" in ev:
                    ids = ev.split(";")
                    for id in ids:
                        num_cur_facts += 1
                        evidence_strs.append(evidence_to_text.get(id.strip(), ""))
                        if evidence_strs[-1] in formatted_context:
                            num_cur_hits += 1
                else:
                    num_cur_facts += 1
                    evidence_strs.append(evidence_to_text.get(ev.strip(), ""))
                    if evidence_strs[-1] in formatted_context:
                        num_cur_hits += 1
            num_facts += num_cur_facts
            recall_hit += num_cur_hits
            perf_matrix["recall"] = f"{num_cur_hits}/{num_cur_facts} = {num_cur_hits/num_cur_facts*100:.2f}%" if num_cur_facts > 0 else "N/A"

            qa["evidence_text"] = evidence_strs
            r_tasks.append(
                get_model_answer(
                    model,
                    group_id,
                    user,
                    qa,
                    formatted_context,
                    USER_PROMPT,
                    perf_matrix,
                )
            )

        # # Baseline: use ground-truth evidence only
        # r_tasks = []
        # for qa in qa_list:
        #     context = f"Evidence:\n"
        #     for ev in qa["evidence"]:
        #         if "," in ev:
        #             ids = ev.split(",")
        #             for id in ids:
        #                 context += evidence_to_text.get(id.strip(), "") + "\n"
        #         elif ";" in ev:
        #             ids = ev.split(";")
        #             for id in ids:
        #                 context += evidence_to_text.get(id.strip(), "") + "\n"
        #         else:
        #             context += evidence_to_text.get(ev.strip(), "") + "\n"
        #     context += "\nFull Conversation:\n" + full_text
            
        #     r_tasks.append(
        #         get_model_answer(
        #             model,
        #             group_id,
        #             user,
        #             qa,
        #             context,
        #         )
        #     )

        print(f"Async gathering {len(r_tasks)} response tasks...")
        ts = time.perf_counter()
        responses.extend(await asyncio.gather(*r_tasks))
        print(f"Gathered {len(r_tasks)} response tasks in {time.perf_counter() - ts:.2f}s")
        answer_total_time += time.perf_counter() - ts

        for category, response in responses:
            # print(f"---\n{response["print_info"][:300]} ---\n")
            category_result = results.get(category, [])
            category_result.append(response)
            results[category] = category_result
        
        if num_processed >= run_until_index:
            break
        num_processed += 1
        # break
    
    final_matrix = f"""Total Episodic Memory retrieval time: {em_total_time:.2f} seconds
Total Knowledge Graph retrieval time: {kg_total_time:.2f} seconds
Average question response time: {(em_total_time + kg_total_time) / num_questions:.2f} seconds
Total Answer generation time: {answer_total_time:.2f} seconds
Total LLM input tokens: {total_input_tolens}
Average LLM input tokens per question: {total_input_tolens}/{num_questions} = {total_input_tolens/num_questions:.2f}
Total LLM output tokens: {total_output_tokens}
Average LLM output tokens per question: {total_output_tokens}/{num_questions} = {total_output_tokens/num_questions:.2f}
Total LLM tokens: {total_input_tolens + total_output_tokens}
Average LLM tokens per question: {total_input_tolens + total_output_tokens}/{num_questions} = {(total_input_tolens + total_output_tokens)/num_questions:.2f}
Total number of sufficiency checks: {total_num_sufficiency_checks}
Average number of sufficiency checks per question: {total_num_sufficiency_checks}/{num_questions} = {total_num_sufficiency_checks/num_questions:.2f}
Overall Evidence Recall: {recall_hit}/{num_facts} = {recall_hit/num_facts*100:.2f}%
Overall Evidence Precision: {recall_hit}/{num_episodes_returned} = {recall_hit/num_episodes_returned*100:.2f}%
Average episodes returned per question: {num_episodes_returned}/{num_questions} = {num_episodes_returned/num_questions:.2f}
Number of questions used KG: {num_used_kg}/{num_questions} = {num_used_kg/num_questions*100:.2f}%
"""
    for cat, res_list in results.items():
        res_list[0]["final_matrix"] = final_matrix
        break

    # print(f"Total Episodic Memory retrieval time: {em_total_time:.2f} seconds")
    # print(f"Total Knowledge Graph retrieval time: {kg_total_time:.2f} seconds")
    # print(f"Total Answer generation time: {answer_total_time:.2f} seconds")
    # print(f"Total LLM input tokens: {total_input_tolens}")
    # print(f"Average LLM input tokens per question: {total_input_tolens}/{num_questions} = {total_input_tolens/num_questions:.2f}")
    # print(f"Total LLM output tokens: {total_output_tokens}")
    # print(f"Average LLM output tokens per question: {total_output_tokens}/{num_questions} = {total_output_tokens/num_questions:.2f}")
    # print(f"Total LLM tokens: {total_input_tolens + total_output_tokens}")
    # print(f"Average LLM tokens per question: {total_input_tolens + total_output_tokens}/{num_questions} = {(total_input_tolens + total_output_tokens)/num_questions:.2f}")
    # print(f"Total number of sufficiency checks: {total_num_sufficiency_checks}")
    # print(f"Average number of sufficiency checks per question: {total_num_sufficiency_checks}/{num_questions} = {total_num_sufficiency_checks/num_questions:.2f}")
    # print(f"Overall Evidence Recall: {recall_hit}/{num_facts} = {recall_hit/num_facts*100:.2f}%")
    # print(f"Overall Evidence Precision: {recall_hit}/{num_episodes_returned} = {recall_hit/num_episodes_returned*100:.2f}%")
    # print(f"Average episodes returned per question: {num_episodes_returned}/{num_questions} = {num_episodes_returned/num_questions:.2f}")
    # print(f"Number of questions used KG: {num_used_kg}/{num_questions} = {num_used_kg/num_questions*100:.2f}%")
    # print(f"Number of questions using long-term memory only: {num_used_longterm_only}/{num_questions} = {num_used_longterm_only/num_questions*100:.2f}%")
    # print(f"Number of questions using both long-term and KG memory: {num_used_both}/{num_questions} = {num_used_both/num_questions*100:.2f}%")
    with open(target_path, "a") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())

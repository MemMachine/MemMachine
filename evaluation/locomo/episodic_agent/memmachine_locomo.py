# This is based on Mem0 (https://github.com/mem0ai/mem0/tree/main/evaluation/src/memzero)

import asyncio
import json
import os
import subprocess
import time
from collections import defaultdict

from dotenv import load_dotenv
from locomo_agent import locomo_response
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

load_dotenv()
root_dir = os.path.dirname(os.path.abspath(__file__))

# Global Variables
# commit_id will remain the same for runs as long as git commit id remains the same
commit_id = (
    subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"])
    .decode("utf-8")
    .strip()
)


class MemMachineSearch:
    def __init__(self, output_path=f"results_IM_{commit_id}.json"):
        self._semaphore = asyncio.Semaphore(10)
        self._output_path = output_path
        if not os.path.exists(output_path):
            self.results = defaultdict(list)
        else:
            with open(output_path, "r") as f:
                self.results = defaultdict(list, json.load(f))

    async def answer_question_with_mcp(self, question, idx, users):
        t1 = time.time()
        response_parsed = await locomo_response(idx, question, users, "gpt-4o-mini")
        t2 = time.time()
        return (
            response_parsed["response"],
            response_parsed["trace"],
            t2 - t1,
        )

    async def process_question(self, val, idx, users, base_url):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            trace,
            response_time,
        ) = await self.answer_question_with_mcp(question, idx, users)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "agent_trace": trace,
            "response_time": response_time,
        }

        return result

    async def process_data_file(
        self, file_path, exclude_category={5}, base_url="http://localhost:8080"
    ):
        async def process_item(idx, item):
            if str(idx) in self.results.keys():
                print(f"Conversation {idx} has already been processed")
                return
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            qa_filtered = [
                i for i in qa if i.get("category", -1) not in exclude_category
            ]

            print(
                f"Filter category: {exclude_category}, {len(qa)} -> {len(qa_filtered)}"
            )

            results_single_convo = await self.process_questions_parallel(
                qa_filtered,
                idx,
                users=[speaker_a, speaker_b],
                base_url=base_url,
            )
            self.results[idx] = results_single_convo
            with open(self._output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        with open(file_path, "r") as f:
            data = json.load(f)
            for idx, convo in tqdm(zip(range(len(data)), data)):
                await process_item(idx, convo)
        # Final save at the end
        with open(self._output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    async def process_questions_parallel(
        self, qa_list, idx, users, base_url="http://localhost:8080"
    ):
        async def process_single_question(val):
            async with self._semaphore:
                result = await self.process_question(val, idx, users, base_url)
                return result

        out = await atqdm.gather(
            *[process_single_question(val) for val in qa_list],
            desc="processing questions",
        )
        return list(out)

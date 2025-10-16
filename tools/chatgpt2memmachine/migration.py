import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai_summary import OpenAISummary
from process_chat_history import (
    load_locomo,
    load_openai,
    locomo_count_conversations,
    openai_count_conversations,
)
from restcli import MemMachineRestClient
from tqdm import tqdm


class MigrationHack:
    def __init__(
        self,
        base_url="http://localhost:8080",
        user_session_file="user_session.json",
        chat_history_file="data/locomo10.json",
        chat_type="locomo",
        start_time=0,
        max_messages=0,
        extract_dir="extracted",
        model=None,
        dry_run=False,
    ):
        self.user_session_file = user_session_file
        with open(self.user_session_file, "r") as f:
            self.user_session = json.load(f)
        self.base_url = base_url
        self.client = MemMachineRestClient(
            base_url=self.base_url, session=self.user_session, verbose=False
        )
        self.chat_history_file = chat_history_file
        self.chat_type = chat_type
        self.start_time = start_time
        self.max_messages = max_messages
        # Extract the base filename from the locomo file path
        self.chat_base_name = os.path.splitext(
            os.path.basename(self.chat_history_file)
        )[0]
        self.extract_dir = extract_dir
        # list of messages in conversations loaded from file
        self.num_conversations = 0
        self.messages = {}  # key: conversation id, value: list of messages
        self.model = model
        self.dry_run = dry_run
        self.api_key = os.getenv("OPENAI_API_KEY", None)
        self.summaries = {}  # key: conversation id, value: list of summaries

    def load(self):
        total_messages = 0
        if self.chat_history_file is not None:
            print(f"-> loading chat history file {self.chat_history_file}")
            print("-> counting conversations...")
            if self.chat_type == "locomo":
                conv_count = locomo_count_conversations(
                    self.chat_history_file, verbose=False
                )
            elif self.chat_type == "openai":
                conv_count = openai_count_conversations(
                    self.chat_history_file, verbose=False
                )
            else:
                raise Exception(f"Error: Invalid chat type: {self.chat_type}")
            print(
                f"-> loaded {conv_count} conversations from {self.chat_type} file {self.chat_history_file}"
            )
            self.num_conversations = conv_count
        # write into extract_dir
        os.makedirs(self.extract_dir, exist_ok=True)
        # Create the extract file name with timestamp
        extract_file_prefix = f"{self.chat_base_name}_extracted"
        for conv_id in range(1, self.num_conversations + 1):
            extract_file = f"{extract_file_prefix}_conv_{conv_id}.txt"
            extract_file = os.path.join(self.extract_dir, extract_file)
            if os.path.exists(extract_file):
                print(f"== Extract file {extract_file} already cached, load from file")
                messages = []
                with open(extract_file, "r") as f:
                    for line in f:
                        messages.append(line.strip())
                self.messages[conv_id] = messages
            else:
                print(f"---> loading messages from conversation {conv_id}...")
                if self.chat_type == "locomo":
                    messages = load_locomo(
                        self.chat_history_file,
                        start_time=self.start_time,
                        conv_num=conv_id,
                        max_messages=self.max_messages,
                        verbose=False,
                    )
                elif self.chat_type == "openai":
                    messages = load_openai(
                        self.chat_history_file,
                        start_time=self.start_time,
                        conv_num=conv_id,
                        max_messages=self.max_messages,
                        verbose=False,
                    )
                else:
                    raise Exception(f"Error: Invalid chat type: {self.chat_type}")
                print(
                    f"---> loaded {len(messages)} messages from conversation {conv_id}"
                )
                total_messages += len(messages)
                self.messages[conv_id] = messages
                with open(extract_file, "w") as f:
                    # Write each message line by line to the extract file
                    for message in self.messages[conv_id]:
                        f.write(message + "\n")

    def summarize_messages(self, summarize_every=20):
        print("== Summarizing messages starts")
        if not self.api_key:
            raise Exception(
                "Error: API key not found, please set environment variable OPENAI_API_KEY"
            )
        openai_summary = OpenAISummary(api_key=self.api_key, model=self.model)

        summarized_file_prefix = f"{self.chat_base_name}_summarized"
        batch_num = 1
        for conv_id in self.messages:
            messages = self.messages[conv_id]
            summarized_file = f"{summarized_file_prefix}_conv_{conv_id}.txt"
            summarized_file = os.path.join(self.extract_dir, summarized_file)
            if os.path.exists(summarized_file):
                print(
                    f"== Summarized file {summarized_file} already cached, load from file"
                )
                if conv_id not in self.summaries or self.summaries[conv_id] is None:
                    self.summaries[conv_id] = []
                with open(summarized_file, "r") as f:
                    for line in f:
                        summary = line.strip()
                        if summary:
                            self.summaries[conv_id].append(summary)
            else:
                self.summaries[conv_id] = []
                for i in range(0, len(messages), summarize_every):
                    batch = messages[i : i + summarize_every]
                    batch_text = "\n".join(batch)
                    summary = ""
                    try:
                        # Get summary from OpenAI
                        response = openai_summary.summarize(batch_text)
                        if "choices" in response and len(response["choices"]) > 0:
                            summary = response["choices"][0]["message"]["content"]
                        else:
                            print(f"Error: No summary generated for batch {batch_num}")
                            print(f"Response: {response}")
                    except Exception as e:
                        print(f"Error processing batch {batch_num}: {e}")
                    if summary:
                        self.summaries[conv_id].append(summary)
                        with open(summarized_file, "a") as f:
                            text = summary.replace("\n", "")
                            f.write(text + "\n")
                    batch_num += 1
        print("== Summarizing messages done")

    def _process_conversation(self, conv_id, messages):
        """Process a single conversation with its own progress bar"""
        # Create a progress bar for this conversation
        pos = conv_id - 1
        msg_pbar = tqdm(
            messages, desc=f"Conv {conv_id}", unit="msg", position=pos, leave=True
        )
        for message in msg_pbar:
            # TODO: insert messages into episodic memory
            if not self.dry_run:
                self.client.post_episodic_memory(
                    message, session_id=f"conversation_{conv_id}"
                )

        msg_pbar.close()
        return conv_id, len(messages)

    def insert_memories(self, summary=False):
        print(f"--- Inserting memories starts, summary={summary}")

        # Process conversations concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(
            max_workers=min(self.num_conversations, 10)
        ) as executor:
            if summary:
                contents = self.summaries
            else:
                contents = self.messages
            # Submit all conversation processing tasks
            future_to_conv = {
                executor.submit(self._process_conversation, conv_id, messages): conv_id
                for conv_id, messages in contents.items()
            }

            # Create a progress bar for completed conversations
            completed_pbar = tqdm(
                total=len(contents), desc="Completed conversations", unit="conv"
            )

            # Process completed tasks
            for future in as_completed(future_to_conv):
                conv_id, msg_count = future.result()
                completed_pbar.set_description(
                    f"Completed conv {conv_id} ({msg_count} msgs)"
                )
                completed_pbar.update(1)

            completed_pbar.close()

        print("--- Inserting memories done")

    def migrate(self, summarize=False, summarize_every=20):
        print("== Loading starts")
        self.load()
        print("== Loading done")
        if summarize:
            print("== Summarizing starts")
            self.summarize_messages(summarize_every)
            print("== Summarizing done")
        print("== Migration starts")
        self.insert_memories(summarize)
        print("== Migration done")


def get_args():
    parser = argparse.ArgumentParser(
        description="Hackathon - Migration from ChatGPT to MemMachine"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8080",
        help="Base URL of the MemMachine API",
    )
    parser.add_argument(
        "--chat_history",
        type=str,
        default="data/locomo10.json",
        help="Chat history file",
    )
    parser.add_argument(
        "--chat_type", type=str, default="locomo", help="Chat type: locomo or openai"
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="0",
        help="only read messages after this time either YYYY-MM-DDTHH:MM:SS or secs since epoch",
    )
    parser.add_argument(
        "--max_messages", type=int, default=0, help="only read this many messages"
    )
    parser.add_argument(
        "--summarize", default=False, action="store_true", help="Summarize messages"
    )
    parser.add_argument(
        "--summarize_every", type=int, default=20, help="Summarize every n messages"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    base_url = args.base_url
    chat_history = args.chat_history
    chat_type = args.chat_type
    start_time = args.start_time
    max_messages = args.max_messages
    # use summarized messages when summarize is true
    summarize = args.summarize
    summarize_every = args.summarize_every
    migration_hack = MigrationHack(
        base_url=base_url,
        user_session_file="user_session.json",
        chat_history_file=chat_history,
        chat_type=chat_type,
        start_time=start_time,
        max_messages=max_messages,
    )

    migration_hack.migrate(summarize=summarize, summarize_every=summarize_every)
    print("== All completed successfully")

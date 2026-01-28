import argparse
import datetime
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import timezone

from openai_summary import OpenAISummary
from parsers import get_parser
from restcli import MemMachineRestClient
from tqdm import tqdm


def parse_time(time_str: str) -> Optional[float]:
    """
    Parse time string to timestamp.
    
    Supports:
    - Unix timestamp (integer or float)
    - ISO format: YYYY-MM-DDTHH:MM:SS
    - Date only: YYYY-MM-DD (assumes 00:00:00)
    
    Args:
        time_str: Time string to parse
        
    Returns:
        Timestamp as float, or None if parsing fails
    """
    if not time_str or time_str == "0":
        return None
    
    # Try as integer timestamp
    try:
        ts = int(time_str)
        if ts > 0:
            return float(ts)
    except ValueError:
        pass
    
    # Try as float timestamp
    try:
        ts = float(time_str)
        if ts > 0:
            return ts
    except ValueError:
        pass
    
    # Try ISO format with time
    try:
        time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
        return time_obj.timestamp()
    except ValueError:
        pass
    
    # Try ISO format with date only
    try:
        time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d")
        return time_obj.timestamp()
    except ValueError:
        pass
    
    return None


class MigrationHack:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        org_id: str = "",
        project_id: str = "",
        input: str = "data/conversations-chatgpt-sample.json",
        source: str = "openai",
        filters: dict | None = None,
        dry_run: bool = False,
        verbose: bool = False,
        max_workers: int = 1
    ):
        self.base_url = base_url
        self.verbose = verbose
        self.client = MemMachineRestClient(
            base_url=self.base_url,
            verbose=verbose,
        )
        self.input_file = input
        self.source = source
        if filters is None:
            self.filters = {}
        else:
            self.filters = filters
        if "user_only" in self.filters:
            self.user_only = self.filters["user_only"]
        else:
            self.user_only = True
        self.parser = get_parser(self.source, self.verbose)
        self.extract_dir = "extracted"
        os.makedirs(self.extract_dir, exist_ok=True)
        self.org_id = org_id
        self.project_id = project_id
        # Extract the base filename from the chat history file path
        self.input_file_base_name = os.path.splitext(
            os.path.basename(self.input_file),
        )[0]
        self.dry_run = dry_run
        self.max_workers = max_workers
        self.conversations = {}

    def load_and_extract(self):
        if self.input_file is None:
            raise Exception("ERROR: Input file not set")
        self.num_conversations = self.parser.count_conversations(self.input_file)
        if not self.dry_run:
            print(f"Found {self.num_conversations} conversation(s)")

        for conv_id in range(1, self.num_conversations + 1):
            extracted_file_name = f"{self.input_file_base_name}_{conv_id}_extracted.json"
            extracted_file = os.path.join(self.extract_dir, extracted_file_name)
            if os.path.exists(extracted_file):
                # load from file directly
                with open(extracted_file, "r") as f:
                    self.conversations[conv_id] = json.load(f)
                continue
            if not self.dry_run:
                print(f"Loading conversation {conv_id}...")
            # Build filters for this conversation (merge global filters with conv index)
            filters = dict(self.filters or {})
            filters["index"] = conv_id
            messages = self.parser.load(self.input_file, filters=filters)
            if not self.dry_run:
                print(f"Loaded {len(messages)} message(s) from conversation {conv_id}")
            self.conversations[conv_id] = messages
            # Dump extracted messages for this conversation
            self.parser.dump_data(messages, output_format="json", outfile=extracted_file)
        
        

    def summarize_messages(self, summarize_every=20):
        print("Summarizing messages...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception(
                "ERROR: API key not found, please set environment variable OPENAI_API_KEY",
            )
        openai_summary = OpenAISummary(api_key=api_key)

        summarized_file_prefix = f"{self.input_file_base_name}_summarized"
        self.summaries = {}
        batch_num = 1
        for conv_id in self.conversations:
            messages = self.conversations[conv_id]
            summarized_file = f"{summarized_file_prefix}_conv_{conv_id}.txt"
            summarized_file = os.path.join(self.extract_dir, summarized_file)
            if os.path.exists(summarized_file):
                if self.verbose:
                    print(f"Using cached summary file: {summarized_file}")
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
                            print(f"ERROR: No summary generated for batch {batch_num}")
                            if self.verbose:
                                print(f"Response: {response}")
                    except Exception as e:
                        print(f"ERROR: Failed to process batch {batch_num}: {e}")
                    if summary:
                        self.summaries[conv_id].append(summary)
                        with open(summarized_file, "a") as f:
                            text = summary.replace("\n", "")
                            f.write(text + "\n")
                    batch_num += 1

    def _format_message(self, message):
        """Format message for MemMachine API"""
        formatted = {
            "content": message.get("content", ""),
        }
        
        metadata = {}
        
        # Iterate through all keys in the message
        for key, value in message.items():
            if key == "content":
                # Content is already set above
                continue
            elif key == "role":
                # Set both role and producer from role field
                formatted["role"] = value
                formatted["producer"] = value
            elif key == "timestamp":
                # Convert timestamp to ISO 8601 format (UTC)
                if isinstance(value, (int, float)):
                    dt = datetime.datetime.fromtimestamp(value, tz=timezone.utc)
                    formatted["timestamp"] = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                else:
                    formatted["timestamp"] = value
            elif key in ["message_id", "chat_id", "chat_title"]:
                # Move these fields to metadata
                metadata[key] = value
            else:
                # Other fields can be omitted or added to metadata
                # For now, we'll omit them unless they're useful
                pass
        
        if metadata:
            formatted["metadata"] = metadata
        
        return formatted

    def _process_conversation(self, conv_id, messages):
        """Process a single conversation with its own progress bar"""
        # Filter out assistant messages if user_only is enabled
        if self.user_only:
            messages = [msg for msg in messages if msg.get("role", "").lower() != "assistant"]
        
        # Create a progress bar for this conversation
        pos = conv_id - 1
        msg_pbar = tqdm(
            messages,
            desc=f"Conv {conv_id}",
            unit="msg",
            position=pos,
            leave=True,
        )
        for message in msg_pbar:
            formatted_message = self._format_message(message)
            self.client.add_memory(
                org_id=self.org_id if self.org_id else "",
                project_id=self.project_id if self.project_id else "",
                messages=[formatted_message],
            )

        msg_pbar.close()
        return conv_id, len(messages)

    def _dry_run(self, summary=False):
        """Print summary of what would be migrated in dry-run mode"""
        if summary:
            contents = getattr(self, 'summaries', {})
            content_type = "summaries"
        else:
            contents = self.conversations
            content_type = "messages"
        
        total_conversations = len(contents)
        
        # Count total items, filtering assistant messages if user_only is enabled
        if self.user_only and not summary:
            total_items = sum(
                len([msg for msg in msgs if msg.get("role", "").lower() != "assistant"])
                for msgs in contents.values()
            )
        else:
            total_items = sum(len(msgs) for msgs in contents.values())
        
        org_display = self.org_id if self.org_id else "universal"
        project_display = self.project_id if self.project_id else "universal"
        
        print(f"\nDry Run Summary:")
        print(f"  Target: {org_display}/{project_display}")
        print(f"  Conversations: {total_conversations}")
        print(f"  Total {content_type}: {total_items}")
        if self.user_only and not summary:
            print(f"  Filter: User messages only (assistant messages excluded)")
        
        # Show sample payload
        if contents and not summary:
            # Get first user message from first conversation (or first message if user_only is disabled)
            first_conv_id = sorted(contents.keys())[0]
            first_messages = contents[first_conv_id]
            if first_messages and isinstance(first_messages, list) and len(first_messages) > 0:
                # Find first user message if user_only is enabled
                if self.user_only:
                    sample_message = next(
                        (msg for msg in first_messages if msg.get("role", "").lower() != "assistant"),
                        None
                    )
                else:
                    sample_message = first_messages[0]
                
                if sample_message and isinstance(sample_message, dict):
                    formatted_sample = self._format_message(sample_message)
                    sample_payload = {
                        "messages": [formatted_sample],
                    }
                    if self.org_id:
                        sample_payload["org_id"] = self.org_id
                    if self.project_id:
                        sample_payload["project_id"] = self.project_id
                    print(f"\n  Sample Payload:")
                    print(f"  {json.dumps(sample_payload, indent=2, ensure_ascii=False)}")

    def add_memories(self, summary=False):
        if self.dry_run:
            # In dry-run mode, just print a summary without actual processing
            self._dry_run(summary)
            return

        print(f"Adding memories to MemMachine...")
        contents = self.conversations
        # Process conversations using ThreadPoolExecutor
        # Default max_workers=1 for sequential processing
        workers = min(self.max_workers, len(contents))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_conversation, conv_id, messages): conv_id
                for conv_id, messages in contents.items()
            }
            
            completed_pbar = tqdm(total=len(contents), desc="Completed conversations", unit="conv")
            for future in as_completed(futures):
                conv_id, msg_count = future.result()
                completed_pbar.set_description(f"Completed conv {conv_id} ({msg_count} msgs)")
                completed_pbar.update(1)
            completed_pbar.close()

        print("Migration complete")

    def migrate(self, summarize=False, summarize_every=20):
        self.load_and_extract()
        
        if summarize:
            self.summarize_messages(summarize_every)
        
        self.add_memories(summarize)


def get_args():
    parser = argparse.ArgumentParser(
        description="Migrate chat history data to MemMachine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate OpenAI chat history to MemMachine
  %(prog)s -i chat.json --org-id my-org --project-id my-project --source openai
  
  # Migrate with filters (only messages after date, limit to 100)
  %(prog)s -i chat.json --org-id my-org --project-id my-project --since 2024-01-01 -l 100
  
  # Dry run to preview what would be migrated
  %(prog)s -i chat.json --org-id my-org --project-id my-project --dry-run
  
  # Migrate specific conversation (index 1) with verbose output
  %(prog)s -i chat.json --org-id my-org --project-id my-project --index 1 -v
        """.strip()
    )
    
    # Core arguments
    core_group = parser.add_argument_group('Core Arguments')
    core_group.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        metavar="FILE",
        help="Input chat history file (required)",
    )
    core_group.add_argument(
        "-s",
        "--source",
        type=str,
        choices=["openai", "locomo"],
        default="openai",
        help="Source format: 'openai' or 'locomo' (default: %(default)s)",
    )
    core_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    
    # MemMachine configuration
    memmachine_group = parser.add_argument_group('MemMachine Configuration')
    memmachine_group.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8080",
        help="Base URL of the MemMachine API (default: %(default)s)",
    )
    memmachine_group.add_argument(
        "--org-id",
        type=str,
        default="",
        help="Organization ID in MemMachine (optional, leave empty to use default)",
    )
    memmachine_group.add_argument(
        "--project-id",
        type=str,
        default="",
        help="Project ID in MemMachine (optional, leave empty to use default)",
    )
    
    # Filtering arguments
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument(
        "--since",
        metavar="TIME",
        help="Only process messages after this time. Supports: Unix timestamp, ISO format (YYYY-MM-DDTHH:MM:SS), or date (YYYY-MM-DD)",
    )
    filter_group.add_argument(
        "-l",
        "--limit",
        type=int,
        metavar="N",
        default=0,
        help="Maximum number of messages to process per conversation (0 = no limit, default: %(default)s)",
    )
    filter_group.add_argument(
        "--index",
        type=int,
        metavar="N",
        default=0,
        help="Process only the conversation/chat at index N (1-based, 0 = all). Works for both OpenAI and Locomo sources.",
    )
    filter_group.add_argument(
        "--chat-title",
        metavar="TITLE",
        help="[OpenAI only] Process only chats matching this title (case-insensitive)",
    )
    filter_group.add_argument(
        "--user-only",
        action="store_true",
        default=True,
        help="Only add user messages to MemMachine (exclude assistant messages)",
    )
    
    # Operation modes
    mode_group = parser.add_argument_group('Operation Modes')
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: loads and extracts data to 'extracted' directory but does not add memories to MemMachine",
    )
    mode_group.add_argument(
        "--workers",
        type=int,
        metavar="N",
        default=1,
        help="Number of worker threads for parallel processing (default: 1, sequential). Set to >1 for parallel processing.",
    )
    
    args = parser.parse_args()
    
    # Parse time string
    if args.since:
        parsed_time = parse_time(args.since)
        if parsed_time is None:
            parser.error(
                f"Invalid time format: '{args.since}'. Use Unix timestamp, ISO format (YYYY-MM-DDTHH:MM:SS), or date (YYYY-MM-DD)"
            )
        args.since = parsed_time
    else:
        args.since = None
    
    # Validate source-specific arguments
    if args.chat_title and args.source != "openai":
        parser.error("--chat-title is only supported for 'openai' source")
    
    return args


if __name__ == "__main__":
    args = get_args()
    args.source = args.source.lower()

    # Build filters dict for parser
    filters: dict = {}
    if args.since:
        filters["since"] = args.since
    if args.limit and args.limit > 0:
        filters["limit"] = args.limit
    if args.index and args.index > 0:
        filters["index"] = args.index
    if args.user_only:
        filters["user_only"] = args.user_only
    if args.chat_title:
        filters["chat_title"] = args.chat_title

    migration_hack = MigrationHack(
        base_url=args.base_url,
        org_id=args.org_id,
        project_id=args.project_id,
        input=args.input,
        source=args.source,
        filters=filters or None,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_workers=args.workers
    )

    # Currently we always migrate full conversations without summarization
    migration_hack.migrate()

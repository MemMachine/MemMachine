import argparse
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from parsers import get_parser
from restcli import MemMachineRestClient
from tqdm import tqdm
from utils import format_timestamp_iso8601, load_run_id_file, parse_time


def generate_run_id() -> str:
    """Generate a unique run ID based on current timestamp."""
    # Format: YYYYMMDDTHHMMSSffffff (filesystem-safe ISO 8601)
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y%m%dT%H%M%S%f")


class MigrationHack:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        org_id: str = "memory_migration",
        project_id: str = "memory_migration",
        memory_types: str = "episodic",
        input: str = "data/conversations-chatgpt-sample.json",
        source: str = "openai",
        filters: dict | None = None,
        dry_run: bool = False,
        verbose: bool = False,
        max_workers: int = 1,
        run_id: str | None = None,
        retry_failed: bool = False,
        resume: bool = False,
        disable_stm_summary: bool = False,
    ):
        self.base_url = base_url
        self.verbose = verbose
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
        # Use provided run_id or generate unique run ID for this execution
        self.run_id = run_id if run_id else generate_run_id()
        self.retry_failed = retry_failed
        self.resume = resume
        self.extract_dir = "extracted"
        os.makedirs(self.extract_dir, exist_ok=True)
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load message IDs from previous run if retry_failed or resume is enabled
        # Store as (conv_id, message_id) tuples for efficient filtering
        self.success_message_ids: set[tuple[int, str]] = set()
        self.error_message_ids: set[tuple[int, str]] = set()
        if self.retry_failed or self.resume:
            if not run_id:
                raise ValueError(
                    "--run-id is required when using --retry-failed or --resume"
                )
            self._load_previous_run_ids()
        if self.verbose:
            self.rest_client = MemMachineRestClient(
                base_url=self.base_url,
                verbose=True,
                run_id=self.run_id,
            )
        else:
            self.rest_client = MemMachineRestClient(
                base_url=self.base_url,
            )
        self.org_id = org_id
        self.project_id = project_id
        self.memory_types = [mt.strip().lower() for mt in memory_types.split(",")]
        # Extract the base filename from the chat history file path
        self.input_file_base_name = os.path.splitext(
            os.path.basename(self.input_file),
        )[0]
        self.dry_run = dry_run
        self.max_workers = max_workers
        self.conversations = {}
        self.total_messages = 0
        self.batch_size = 5  # Number of messages to send per API call
        # Statistics tracking
        self.stats = {
            "run_id": self.run_id,
            "start_time": datetime.datetime.now(datetime.UTC).isoformat(),
            "input_file": self.input_file,
            "source": self.source,
            "org_id": self.org_id or "universal",
            "project_id": self.project_id or "universal",
            "total_conversations": 0,
            "total_messages": 0,
            "successful_messages": 0,
            "failed_messages": 0,
            "user_only": self.user_only,
        }
        if self.retry_failed:
            self.stats["retry_failed"] = True
            self.stats["retry_run_id"] = run_id
        if self.resume:
            self.stats["resume"] = True
            self.stats["resume_run_id"] = run_id
        self.disable_stm_summary = disable_stm_summary

    def _load_previous_run_ids(self):
        """Load message IDs from success and error files of a previous run.

        Files are expected to be in format: conv_id:message_id
        For backward compatibility, also supports message_id-only format.
        """
        success_file = os.path.join(self.output_dir, f"success_{self.run_id}.txt")
        errors_file = os.path.join(self.output_dir, f"errors_{self.run_id}.txt")

        if self.resume:
            self.success_message_ids = load_run_id_file(success_file)
            if self.verbose:
                print(
                    f"Loaded {len(self.success_message_ids)} successful message records from previous run"
                )
            if len(self.success_message_ids) == 0:
                print(f"WARNING: No successful message IDs found in {success_file}")

        if self.retry_failed:
            self.error_message_ids = load_run_id_file(errors_file)
            if self.verbose:
                print(
                    f"Loaded {len(self.error_message_ids)} failed message records from previous run"
                )
            if len(self.error_message_ids) == 0:
                print(f"WARNING: No failed message IDs found in {errors_file}")

    def _extract_single_conversation(
        self, conv_id: int, all_data: list
    ) -> tuple[int, list, int]:
        """Extract and save a single conversation from the full dataset.

        Args:
            conv_id: Conversation ID (1-indexed)
            all_data: Full dataset loaded from JSON

        Returns:
            Tuple of (conv_id, messages, message_count)
        """
        extracted_file_name = f"{self.input_file_base_name}_{conv_id}_extracted.json"
        extracted_file = os.path.join(self.extract_dir, extracted_file_name)

        if os.path.exists(extracted_file):
            # Load from existing extracted file
            with open(extracted_file, "r") as f:
                messages = json.load(f)
            filtered = self._filter_messages(messages, self.filters)
        else:
            # Extract from full dataset (directly from all_data, no file I/O)
            # Get the conversation at index conv_id (1-indexed)
            if conv_id <= len(all_data):
                chat = all_data[conv_id - 1]  # Convert to 0-indexed

                # Extract messages from this chat using parser's logic
                chat_title_actual = chat.get("title", "")
                messages = self.parser._extract_chat_messages(chat, chat_title_actual)
                messages.sort(key=lambda x: x.get("timestamp", 0))

                # Dump extracted messages for this conversation
                self.parser.dump_data(
                    messages, output_format="json", outfile=extracted_file
                )
                filtered = messages
            else:
                filtered = []

        # Count messages
        msg_count = 0
        if isinstance(filtered, list):
            msg_count = len(filtered)
        elif isinstance(filtered, str):
            msg_count = 1

        return conv_id, filtered, msg_count

    def load_and_extract(self):
        if self.input_file is None:
            raise Exception("ERROR: Input file not set")
        self.num_conversations = self.parser.count_conversations(self.input_file)
        self.stats["total_conversations"] = self.num_conversations
        if not self.dry_run:
            print(f"Found {self.num_conversations} conversation(s)")
            print(f"Run ID: {self.run_id}")

        # Check if user specified a specific conversation index
        user_index = self.filters.get("index", 0) if self.filters else 0

        # Determine which conversations to process
        if user_index and user_index > 0:
            # User specified a specific conversation, only process that one
            if self.verbose:
                print(f"DEBUG: User specified conversation index: {user_index}")
            if user_index > self.num_conversations:
                raise Exception(
                    f"ERROR: Conversation index {user_index} is out of range. "
                    f"File contains {self.num_conversations} conversation(s)."
                )
            conv_ids_to_process = [user_index]
        else:
            # Process all conversations
            conv_ids_to_process = list(range(1, self.num_conversations + 1))

        # Different behavior for retry_failed vs resume:
        # - retry_failed: Only load conversations that have failed messages (optimize at conversation level)
        # - resume: Load all conversations, filter at message level (skip successful ones)
        if self.retry_failed and conv_ids_to_process:
            # Only process conversations that have failed messages
            relevant_conv_ids = {conv_id for conv_id, _ in self.error_message_ids}
            conv_ids_to_process = [
                cid for cid in conv_ids_to_process if cid in relevant_conv_ids
            ]
            if self.verbose and conv_ids_to_process:
                print(
                    f"Optimized: Processing {len(conv_ids_to_process)} conversation(s) with failed messages"
                )
            elif self.verbose:
                print("WARNING: No conversations found with failed messages")
        elif self.resume:
            # Resume: Load all conversations normally, filtering will happen at message level
            if self.verbose:
                print(
                    "Resume mode: Loading all conversations, will skip messages from success file"
                )

        # Load full dataset once (only if we need to extract new conversations)
        needs_extraction = any(
            not os.path.exists(
                os.path.join(
                    self.extract_dir,
                    f"{self.input_file_base_name}_{conv_id}_extracted.json",
                )
            )
            for conv_id in conv_ids_to_process
        )

        all_data = None
        if needs_extraction:
            if self.verbose:
                print("Loading full dataset for extraction...")
            all_data = self.parser.load_json(self.input_file)
            if self.verbose:
                print(f"Loaded {len(all_data)} conversations from file")

        # Parallel extraction using ThreadPoolExecutor
        workers = min(self.max_workers, len(conv_ids_to_process))

        if self.verbose:
            print(f"Extracting conversations with {workers} worker(s)...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._extract_single_conversation, conv_id, all_data
                ): conv_id
                for conv_id in conv_ids_to_process
            }

            # Use progress bar if not verbose
            if not self.verbose:
                pbar = tqdm(
                    total=len(conv_ids_to_process),
                    desc="Extracting conversations",
                    unit="conv",
                )

            for future in as_completed(futures):
                conv_id, messages, msg_count = future.result()
                self.conversations[conv_id] = messages
                self.total_messages += msg_count

                if not self.verbose:
                    pbar.update(1)
                elif self.verbose:
                    print(f"Extracted conversation {conv_id} ({msg_count} messages)")

            if not self.verbose:
                pbar.close()

        # Update stats with total messages count
        self.stats["total_messages"] = self.total_messages

    def _format_message(self, message):
        """Format message for MemMachine API"""
        # Handle string messages (text-only)
        if isinstance(message, str):
            return {
                "content": message,
            }

        # Handle dictionary messages
        formatted = {
            "content": message.get("content", ""),
        }

        metadata = {}

        # Iterate through all keys in the message
        for key, value in message.items():
            if key == "content":
                # Content is already set above
                continue
            if key == "role":
                # Set both role and producer from role field
                formatted["role"] = value
                formatted["producer"] = value
            elif key == "speaker":
                formatted["producer"] = value
            elif key == "timestamp":
                # Convert timestamp to ISO 8601 format (UTC)
                if isinstance(value, (int, float)):
                    formatted["timestamp"] = format_timestamp_iso8601(value)
                else:
                    formatted["timestamp"] = value
            elif key in ["message_id", "chat_id", "chat_title"]:
                # Move these fields to metadata
                metadata[key] = value

        if metadata:
            formatted["metadata"] = metadata

        return formatted

    def _log_message_ids(self, conv_id, message_ids, success=True, error_msg=None):
        """Log message IDs to success or error file.

        Args:
            conv_id: Conversation ID
            message_ids: List of message IDs to log
            success: If True, log to success file; if False, log to error file
            error_msg: Optional error message to include in error file
        """
        filename = (
            f"success_{self.run_id}.txt" if success else f"errors_{self.run_id}.txt"
        )
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "a") as f:
            f.writelines(f"{conv_id}:{msg_id}\n" for msg_id in message_ids)
            if not success and error_msg and self.verbose:
                f.write(f"  Error: {error_msg}\n")

    def _filter_messages(self, messages, filters: dict | None = None):
        """Filter out assistant messages if user_only is enabled."""
        if filters is None:
            filters = {}
        since = filters.get("since", 0) or 0
        limit = filters.get("limit", 0) or 0
        chat_title = filters.get("chat_title", "")
        user_only = filters.get("user_only", False)

        print(
            f"Filtering messages: since={since}, limit={limit}, chat_title={chat_title}, user_only={user_only}"
        )
        filtered = []
        for msg in messages:
            # String messages are always included (text-only, no role)
            if isinstance(msg, str):
                filtered.append(msg)
                continue
            # not a dictionary, skip
            if not isinstance(msg, dict):
                continue
            # check if the message is after the since timestamp
            if since and msg.get("timestamp", 0) < since:
                continue
            # check if the message is before the limit
            if limit and len(filtered) >= limit:
                break
            # check if user_only is enabled, only keep user messages
            if user_only and msg.get("role", "") != "user":
                continue
            # check if the message is a chat title message and chat_title is not empty
            if chat_title and msg.get("chat_title", "") != chat_title:
                continue
            filtered.append(msg)
        print(f"Total messages: {len(messages)}, filtered messages: {len(filtered)}")
        return filtered

    def _process_conversation(self, conv_id, messages):
        """Process a single conversation with batching support"""
        # Format all messages first
        formatted_messages = [self._format_message(msg) for msg in messages]

        # Filter messages based on retry_failed or resume flags
        # Behavior differs:
        # - retry_failed: Only process messages that are in errors_*.txt (only retry failed ones)
        # - resume: Process all messages except those in success_*.txt (continue from where we left off)
        original_count = len(formatted_messages)
        if self.retry_failed or self.resume:
            filtered_formatted = []
            skipped_count = 0
            for msg in formatted_messages:
                msg_id = msg.get("metadata", {}).get("message_id", "")
                # Skip messages without message_id when using retry/resume (can't track them)
                if not msg_id:
                    if self.verbose:
                        skipped_count += 1
                    continue

                # Match by exact (conv_id, message_id) tuple
                key = (conv_id, msg_id)

                if self.retry_failed:
                    # retry_failed: Only process messages that are in the error file
                    should_include = key in self.error_message_ids
                elif self.resume:
                    # resume: Skip messages that succeeded (are in success file), process everything else
                    should_include = key not in self.success_message_ids
                else:
                    should_include = True

                if should_include:
                    filtered_formatted.append(msg)
                else:
                    skipped_count += 1
            formatted_messages = filtered_formatted
            if self.verbose and skipped_count > 0:
                print(
                    f"Filtered {skipped_count} message(s) based on previous run results"
                )
            if len(formatted_messages) == 0:
                if self.verbose:
                    print(
                        f"No messages to process after filtering (original: {original_count} messages)"
                    )
                return conv_id, 0

        # Create progress bar only if not verbose (to avoid messing up console output)
        msg_pbar = None
        if not self.verbose:
            pos = conv_id - 1
            msg_pbar = tqdm(
                total=len(formatted_messages),
                desc=f"Conv {conv_id}",
                unit="msg",
                position=pos,
                leave=True,
            )

        # Process messages in batches
        for i in range(0, len(formatted_messages), self.batch_size):
            batch = formatted_messages[i : i + self.batch_size]
            if self.source == "openai":
                message_ids = [
                    msg.get("metadata", {}).get("message_id", "") for msg in batch
                ]
            try:
                self.rest_client.add_memory(
                    org_id=self.org_id if self.org_id else "",
                    project_id=self.project_id if self.project_id else "",
                    memory_types=self.memory_types if self.memory_types else [],
                    messages=batch,
                )
                # Log the success request into run-specific success file
                if self.source == "openai":
                    self._log_message_ids(conv_id, message_ids, success=True)
                self.stats["successful_messages"] += len(batch)
            except Exception as e:
                error_msg = str(e)
                print(
                    f"Error adding memory with message ids: [{','.join(message_ids)}]"
                )
                if self.verbose:
                    print(f"  Error details: {error_msg}")
                # Log the error request into run-specific errors file
                if self.source == "openai":
                    self._log_message_ids(
                        conv_id, message_ids, success=False, error_msg=error_msg
                    )
                self.stats["failed_messages"] += len(batch)
            # Update progress bar if it exists
            if msg_pbar:
                msg_pbar.update(len(batch))

        # Close progress bar if it was created
        if msg_pbar:
            msg_pbar.close()
        return conv_id, len(formatted_messages)

    def _dry_run(self):
        """Print summary of what would be migrated in dry-run mode"""
        contents = self.conversations
        total_conversations = len(contents)

        org_display = self.org_id if self.org_id else "universal"
        project_display = self.project_id if self.project_id else "universal"

        print("\nDry Run Summary:")
        print(f"  Run ID: {self.run_id}")
        print(f"  Target: {org_display}/{project_display}")
        print(f"  Conversations: {total_conversations}")
        print(f"  Total messages: {self.total_messages}")

        # Show sample payload
        if contents:
            # Get first user message from first conversation (or first message if user_only is disabled)
            first_conv_id = sorted(contents.keys())[0]
            first_messages = contents[first_conv_id]
            if (
                first_messages
                and isinstance(first_messages, list)
                and len(first_messages) > 0
            ):
                sample_message = None
                for msg in first_messages:
                    if isinstance(msg, dict):
                        sample_message = msg
                        break

                if sample_message and isinstance(sample_message, dict):
                    formatted_sample = self._format_message(sample_message)
                    sample_payload = {
                        "messages": [formatted_sample],
                    }
                    if self.org_id:
                        sample_payload["org_id"] = self.org_id
                    if self.project_id:
                        sample_payload["project_id"] = self.project_id
                    print("\n  Sample Payload:")
                    print(
                        f"  {json.dumps(sample_payload, indent=2, ensure_ascii=False)}"
                    )

    def add_memories(self):
        if len(self.conversations) == 0:
            print("No conversations to process. Exiting.")
            return
        if self.dry_run:
            # In dry-run mode, just print a summary without actual processing
            self._dry_run()
            # Update final statistics for dry-run
            self._update_final_stats()
            # Write statistics even in dry-run mode
            self._write_statistics()
            print(
                f"\nStatistics saved to: {os.path.join(self.output_dir, f'migration_stats_{self.run_id}.json')}"
            )
            return

        print("Adding memories to MemMachine...")
        # Process conversations using ThreadPoolExecutor
        # Default max_workers=1 for sequential processing
        workers = min(self.max_workers, len(self.conversations))

        # Warn user if using sequential processing with many messages
        if workers <= 1 and self.total_messages > 1000:
            print(
                f"WARNING: Processing {self.total_messages} messages with {workers} worker(s) (sequential). "
                f"Consider setting --workers to larger value to speed up migration."
            )
            response = input("Do you want to continue? (yes/no): ").strip().lower()
            if response not in ["yes", "y"]:
                print("Exiting. Please adjust --workers and try again.")
                return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_conversation, conv_id, messages): conv_id
                for conv_id, messages in self.conversations.items()
            }

            if self.verbose:
                # In verbose mode, skip progress bar to avoid messing up console output
                for future in as_completed(futures):
                    conv_id, msg_count = future.result()
                    print(f"Completed conversation {conv_id} ({msg_count} messages)")
            else:
                # Use progress bar when not verbose
                completed_pbar = tqdm(
                    total=len(self.conversations),
                    desc="Completed conversations",
                    unit="conv",
                )
                for future in as_completed(futures):
                    conv_id, msg_count = future.result()
                    completed_pbar.set_description(
                        f"Completed conv {conv_id} ({msg_count} msgs)"
                    )
                    completed_pbar.update(1)
                completed_pbar.close()

        # Update final statistics
        self._update_final_stats()

        # Write statistics to file
        self._write_statistics()

        print("Migration complete")
        print(f"Run ID: {self.run_id}")
        print(f"Duration: {self.stats['duration_seconds']:.2f} seconds")
        print(
            f"Statistics saved to: {os.path.join(self.output_dir, f'migration_stats_{self.run_id}.json')}"
        )

    def _update_final_stats(self):
        """Update final statistics with end time and duration."""
        self.stats["end_time"] = datetime.datetime.now(datetime.UTC).isoformat()
        start = datetime.datetime.fromisoformat(self.stats["start_time"])
        end = datetime.datetime.fromisoformat(self.stats["end_time"])
        self.stats["duration_seconds"] = (end - start).total_seconds()

    def _write_statistics(self):
        """Write migration statistics to a JSON file."""
        stats_file = os.path.join(
            self.output_dir, f"migration_stats_{self.run_id}.json"
        )
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    def migrate(self):
        """Load conversations and add them to MemMachine"""
        # Disable short-term memory summarization if requested
        if self.disable_stm_summary:
            org = self.org_id or "universal"
            proj = self.project_id or "universal"
            self.rest_client.ensure_project(org, proj)
            print("Disabling short-term memory summarization...")
            self.rest_client.configure_short_term_memory(org, proj, enabled=False)
            print("Short-term memory summarization disabled.")

        self.load_and_extract()
        self.add_memories()

        # Re-enable short-term memory summarization after import
        if self.disable_stm_summary:
            print("Re-enabling short-term memory summarization...")
            org = self.org_id or "universal"
            proj = self.project_id or "universal"
            self.rest_client.configure_short_term_memory(org, proj, enabled=True)
            print("Short-term memory summarization re-enabled.")


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
  
  # Retry failed messages from a previous run
  %(prog)s -i chat.json --org-id my-org --project-id my-project --run-id 20260202T235246858124 --retry-failed
  
  # Resume migration, skipping already successful messages
  %(prog)s -i chat.json --org-id my-org --project-id my-project --run-id 20260202T235246858124 --resume
        """.strip(),
    )

    # Core arguments
    core_group = parser.add_argument_group("Core Arguments")
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
    memmachine_group = parser.add_argument_group("MemMachine Configuration")
    memmachine_group.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8080",
        help="Base URL of the MemMachine API (default: %(default)s)",
    )
    memmachine_group.add_argument(
        "--org-id",
        type=str,
        default="memory_migration",
        help="Organization ID in MemMachine (default: %(default)s)",
    )
    memmachine_group.add_argument(
        "--project-id",
        type=str,
        default="memory_migration",
        help="Project ID in MemMachine (default: %(default)s)",
    )
    memmachine_group.add_argument(
        "--memory-types",
        type=str,
        default="episodic",
        help="Memory types to use (default: %(default)s). Can be 'episodic', 'semantic', or 'episodic,semantic'",
    )

    # Filtering arguments
    filter_group = parser.add_argument_group("Filtering Options")
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
    mode_group = parser.add_argument_group("Operation Modes")
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
    mode_group.add_argument(
        "--run-id",
        type=str,
        metavar="ID",
        help="Specify the run ID from a previous execution. Required when using --retry-failed or --resume to identify which previous run to use.",
    )
    mode_group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only messages that failed in a previous run. Requires --run-id to specify the previous run.",
    )
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Skip messages that succeeded in a previous run. Requires --run-id to specify the previous run.",
    )
    mode_group.add_argument(
        "--disable-stm-summary",
        action="store_true",
        dest="disable_stm_summary",
        default=True,
        help="Disable short-term memory summarization during import to reduce LLM costs (default: enabled).",
    )
    mode_group.add_argument(
        "--enable-stm-summary",
        action="store_false",
        dest="disable_stm_summary",
        help="Keep short-term memory summarization enabled during import.",
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

    # Validate source-specific arguments
    if args.chat_title and args.source != "openai":
        parser.error("--chat-title is only supported for 'openai' source")

    # Validate retry/resume arguments
    if (args.retry_failed or args.resume) and args.source != "openai":
        parser.error(
            "--retry-failed and --resume are only supported for 'openai' source. "
            "These features require message_id tracking which is only available in OpenAI format."
        )
    if (args.retry_failed or args.resume) and not args.run_id:
        parser.error("--run-id is required when using --retry-failed or --resume")
    if args.retry_failed and args.resume:
        parser.error("--retry-failed and --resume cannot be used together")
    if args.run_id and not (args.retry_failed or args.resume):
        parser.error(
            "--run-id can only be used with --retry-failed or --resume. "
            "For a new migration run, omit --run-id to generate a new unique run ID."
        )

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
        memory_types=args.memory_types,
        input=args.input,
        source=args.source,
        filters=filters or None,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_workers=args.workers,
        run_id=args.run_id,
        retry_failed=args.retry_failed,
        resume=args.resume,
        disable_stm_summary=args.disable_stm_summary,
    )

    # Currently we always migrate full conversations without summarization
    migration_hack.migrate()

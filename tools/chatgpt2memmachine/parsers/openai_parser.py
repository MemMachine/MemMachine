"""
OpenAI chat history parser.

This module provides the parser for OpenAI chat history export format.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseParser


class OpenAIParser(BaseParser):
    """Parser for OpenAI chat history export format."""

    def count_conversations(self, infile: str) -> int:
        """Count the number of conversations in the OpenAI export file."""
        if self.verbose:
            self.logger.debug(f"Loading OpenAI input file {infile}")
        data = self.load_json(infile)
        return len(data)

    def validate(self, infile: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate OpenAI chat history file structure without processing messages.

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Check if file exists and is readable
        try:
            data = self.load_json(infile)
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
            return False, errors, warnings

        # Check if data is a list
        if not isinstance(data, list):
            errors.append("Root element must be a list/array of chats")
            return False, errors, warnings

        if len(data) == 0:
            warnings.append("Input file contains no chats")
            return True, errors, warnings

        self.logger.debug(f"Validating {len(data)} chat(s)")

        # Track totals across all chats
        total_conversation_count = 0
        total_message_count = 0
        total_valid_message_count = 0

        # Validate each chat
        for chat_idx, chat in enumerate(data, 1):
            chat_prefix = f"Chat {chat_idx}"

            # Required chat-level fields
            if not isinstance(chat, dict):
                errors.append(f"{chat_prefix}: Chat must be a dictionary/object")
                continue

            if "title" not in chat:
                errors.append(f"{chat_prefix}: Missing required field 'title'")
            elif not isinstance(chat.get("title"), str):
                warnings.append(f"{chat_prefix}: Field 'title' is not a string")

            if "create_time" not in chat:
                warnings.append(
                    f"{chat_prefix}: Missing optional field 'create_time' (chat timestamp)"
                )
            elif not isinstance(chat.get("create_time"), (int, float)):
                warnings.append(f"{chat_prefix}: Field 'create_time' is not a number")

            if "mapping" not in chat:
                errors.append(f"{chat_prefix}: Missing required field 'mapping'")
                continue

            mapping = chat.get("mapping")
            if not isinstance(mapping, dict):
                errors.append(
                    f"{chat_prefix}: Field 'mapping' must be a dictionary/object"
                )
                continue

            if len(mapping) == 0:
                warnings.append(f"{chat_prefix}: Mapping is empty (no messages)")
                continue

            # Count this conversation
            total_conversation_count += 1

            # Validate messages in mapping
            message_count, valid_message_count = self._validate_chat_messages(
                mapping, chat_prefix, errors, warnings
            )

            # Update totals
            total_message_count += message_count
            total_valid_message_count += valid_message_count

            if message_count > 0 and valid_message_count == 0:
                warnings.append(
                    f"{chat_prefix}: No messages with valid content structure (all will be skipped)"
                )
            elif message_count > 0:
                self.logger.debug(
                    f"{chat_prefix}: {valid_message_count}/{message_count} messages have valid content structure "
                    f"(Running totals: {total_conversation_count} conversations, "
                    f"{total_valid_message_count}/{total_message_count} valid messages)"
                )

        # Final summary debug logging
        self.logger.debug(
            f"Validation summary: {total_conversation_count} conversations, "
            f"{total_valid_message_count}/{total_message_count} valid messages"
        )

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def _validate_chat_messages(
        self,
        mapping: Dict[str, Any],
        chat_prefix: str,
        errors: List[str],
        warnings: List[str],
    ) -> Tuple[int, int]:
        """Validate messages in a chat mapping and return counts."""
        message_count = 0
        valid_message_count = 0

        for msg_id, chat_map in mapping.items():
            if not isinstance(chat_map, dict):
                warnings.append(
                    f"{chat_prefix}: Mapping entry '{msg_id}' is not a dictionary"
                )
                continue

            if "message" not in chat_map:
                warnings.append(
                    f"{chat_prefix}: Mapping entry '{msg_id}' missing 'message' field"
                )
                continue

            message = chat_map.get("message")
            if message is None:
                continue

            message_count += 1

            if not isinstance(message, dict):
                errors.append(f"{chat_prefix}: Message '{msg_id}' is not a dictionary")
                continue

            content = message.get("content")
            if not isinstance(content, dict):
                errors.append(
                    f"{chat_prefix}: Message '{msg_id}' field 'content' is not a dictionary"
                )
                continue

            content_type = content.get("content_type")
            if not content_type:
                errors.append(
                    f"{chat_prefix}: Message '{msg_id}' missing required field 'content.content_type'"
                )
                continue

            if content_type != "text":
                continue

            parts = content.get("parts", [])
            if not parts:
                errors.append(
                    f"{chat_prefix}: Message '{msg_id}' missing required field 'content.parts'"
                )
            elif not isinstance(parts, list):
                errors.append(
                    f"{chat_prefix}: Message '{msg_id}' field 'content.parts' is not a list"
                )
            elif len(parts) == 0:
                warnings.append(
                    f"{chat_prefix}: Message '{msg_id}' has empty 'content.parts'"
                )
            elif all(not str(part).strip() for part in parts):
                warnings.append(
                    f"{chat_prefix}: Message '{msg_id}' has 'content.parts' with only empty strings"
                )
            else:
                valid_message_count += 1

        return message_count, valid_message_count

    def _extract_chat_messages(
        self, chat: Dict[str, Any], chat_title: str
    ) -> List[Dict[str, Any]]:
        """
        Extract messages from a chat and add chat metadata.

        Args:
            chat: Chat dictionary containing mapping and metadata
            chat_title: Title of the chat

        Returns:
            List of message dictionaries with role, content, timestamp, and metadata
        """
        chat_messages = []
        chat_id = chat.get("id")
        chat_create_time = chat.get("create_time")

        for chat_map in chat.get("mapping", {}).values():
            message = chat_map.get("message")
            if not message:
                continue

            author = message.get("author")
            content = message.get("content")
            if not author or not content:
                continue

            role = author.get("role")
            if not role:
                continue

            content_type = content.get("content_type")
            if content_type != "text":
                continue

            parts = content.get("parts", [])
            if not parts:
                continue

            # Join text parts
            try:
                text_content = "".join(str(part) for part in parts if part)
            except Exception as e:
                self.logger.warning(f"Failed to join message parts: {e}")
                continue

            if not text_content.strip():
                continue

            timestamp = message.get("create_time")
            if not timestamp:
                self.logger.warning(
                    f"Message missing timestamp: {text_content[:50]}..."
                )
                continue

            # Build message data structure
            message_data = {
                "role": role,
                "content": text_content,
                "timestamp": timestamp,
                "content_type": content_type,
                "chat_id": chat_id,
                "chat_title": chat_title,
                "chat_create_time": chat_create_time,
            }

            # Add optional metadata
            optional_fields = {"id": "message_id"}
            for field, key in optional_fields.items():
                if field in message:
                    message_data[key] = message[field]

            chat_messages.append(message_data)

        return chat_messages

    def load(
        self,
        infile: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load messages from OpenAI chat history export.

        Args:
            infile: Path to OpenAI chat history JSON file
            filters: Optional dictionary with filter parameters:
                - since: Only load messages after this timestamp (float)
                - index: Load only this conversation number (1-indexed, int)
                - limit: Maximum number of messages to load (int)
                - chat_title: Load only chats matching this title (case-insensitive, str)

        Returns:
            List of message dictionaries with role, content, timestamp, and metadata
        """
        # Extract filter values from filters dict
        if filters is None:
            filters = {}

        since = filters.get("since", 0) or 0
        index = filters.get("index", 0) or 0
        limit = filters.get("limit", 0) or 0
        user_only = filters.get("user_only", False)
        chat_title = filters.get("chat_title")

        self.logger.debug(
            f"Loading OpenAI chat history: since={since}, limit={limit}, index={index}, user_only={user_only}, chat_title={chat_title}"
        )

        data = self.load_json(infile)
        messages = []
        chat_count = 0
        processed_chat_count = 0
        msg_count = 0

        for chat in data:
            chat_count += 1

            # Apply filters
            if not self._should_process_chat(
                chat, chat_count, index, chat_title, since
            ):
                continue

            processed_chat_count += 1
            chat_title_actual = chat.get("title", "")
            self.logger.info(f"Processing chat {chat_count}: {chat_title_actual}")

            # Extract and process messages
            chat_messages = self._extract_chat_messages(chat, chat_title_actual)
            chat_messages.sort(key=lambda x: x.get("timestamp", 0))

            # Add messages up to limit
            for msg in chat_messages:
                if user_only and msg.get("role") == "assistant":
                    continue
                messages.append(msg)
                msg_count += 1
                if limit and msg_count >= limit:
                    self.logger.info(f"Reached max messages limit: {msg_count}")
                    return messages

            self.logger.debug(
                f"Finished processing chat {chat_count}: {len(chat_messages)} messages"
            )

        self.logger.info(
            f"Total messages loaded: {len(messages)} from {processed_chat_count} chat(s)"
        )
        return messages

    def _should_process_chat(
        self,
        chat: Dict[str, Any],
        chat_count: int,
        index: int,
        chat_title: Optional[str],
        since: float,
    ) -> bool:
        """Check if a chat should be processed based on filters."""
        if index and chat_count != index:
            return False

        if chat_title:
            chat_title_actual = chat.get("title", "")
            if chat_title.lower() != chat_title_actual.lower():
                self.logger.debug(
                    f"Skipping chat (title mismatch): {chat_title_actual}"
                )
                return False

        chat_time = chat.get("create_time")
        if chat_time and since and self.timestamp_compare(since, chat_time) > 0:
            self.logger.debug(f"Skipping old chat {chat_count} (time={chat_time})")
            return False

        return True

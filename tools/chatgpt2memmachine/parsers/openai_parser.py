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
        # Count chats in the list
        chat_count = 0
        for _ in data:
            chat_count += 1
        return chat_count

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
            message_count = 0
            valid_message_count = 0

            for msg_id, chat_map in mapping.items():
                if not isinstance(chat_map, dict):
                    warnings.append(
                        f"{chat_prefix}: Mapping entry '{msg_id}' is not a dictionary"
                    )
                    continue

                # Check if message exists (can be null)
                if "message" not in chat_map:
                    warnings.append(
                        f"{chat_prefix}: Mapping entry '{msg_id}' missing 'message' field"
                    )
                    continue

                message = chat_map.get("message")
                if message is None:
                    # Null messages are valid (root nodes)
                    continue

                message_count += 1

                if not isinstance(message, dict):
                    errors.append(
                        f"{chat_prefix}: Message '{msg_id}' is not a dictionary"
                    )
                    continue

                content = message.get("content")
                if not isinstance(content, dict):
                    errors.append(
                        f"{chat_prefix}: Message '{msg_id}' field 'content' is not a dictionary"
                    )
                    continue

                if "content_type" not in content:
                    errors.append(
                        f"{chat_prefix}: Message '{msg_id}' missing required field 'content.content_type'"
                    )
                elif content.get("content_type") != "text":
                    # Non-text content is skipped, but not an error
                    continue

                if "parts" not in content:
                    errors.append(
                        f"{chat_prefix}: Message '{msg_id}' missing required field 'content.parts'"
                    )
                elif not isinstance(content.get("parts"), list):
                    errors.append(
                        f"{chat_prefix}: Message '{msg_id}' field 'content.parts' is not a list"
                    )
                elif len(content.get("parts", [])) == 0:
                    warnings.append(
                        f"{chat_prefix}: Message '{msg_id}' has empty 'content.parts'"
                    )
                else:
                    # Check if all parts are empty strings
                    parts = content.get("parts", [])
                    if all(not str(part).strip() for part in parts):
                        warnings.append(
                            f"{chat_prefix}: Message '{msg_id}' has 'content.parts' with only empty strings"
                        )
                    else:
                        # Message has valid content structure (text type, non-empty parts)
                        valid_message_count += 1

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

    def _extract_message_from_chat_map(
        self, chat_map: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract message data from a chat mapping entry.

        Args:
            chat_map: A single entry from chat["mapping"]

        Returns:
            Dictionary with message data (role, content, timestamp, etc.) or None if invalid
        """
        # Validate required fields
        if not chat_map.get("message"):
            return None

        message = chat_map["message"]
        author = message.get("author")
        content = message.get("content")

        if not author or not content:
            return None

        role = author.get("role")
        if not role:
            return None

        content_type = content.get("content_type")
        if not content_type:
            return None

        # Only extract text content (skip images, code, etc.)
        if content_type != "text":
            return None

        # Extract message parts
        parts = content.get("parts", [])
        if not parts:
            return None

        # Join text parts
        try:
            text_content = "".join(str(part) for part in parts if part)
        except Exception as e:
            self.logger.warning(f"Failed to join message parts: {e}")
            return None

        if not text_content.strip():
            return None

        # Extract timestamp
        timestamp = message.get("create_time")
        if not timestamp:
            self.logger.warning(f"Message missing timestamp: {text_content[:50]}...")
            return None

        # Build message data structure
        message_data = {
            "role": role,
            "content": text_content,
            "timestamp": timestamp,
            "content_type": content_type,
        }

        # Add optional metadata
        if "id" in message:
            message_data["message_id"] = message["id"]
        if "model_slug" in message:
            message_data["model"] = message["model_slug"]
        if "status" in message:
            message_data["status"] = message["status"]

        return message_data

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
        chat_title = filters.get("chat_title")

        self.logger.debug(
            f"Loading OpenAI chat history: since={since}, limit={limit}, index={index}, chat_title={chat_title}"
        )

        # Load JSON data
        data = self.load_json(infile)

        messages = []
        chat_count = 0
        msg_count = 0

        # Process each chat
        for chat in data:
            chat_count += 1

            # Filter by conversation number
            if index and chat_count != index:
                continue

            # Filter by title
            chat_title_actual = chat.get("title", "")
            if chat_title and chat_title.lower() != chat_title_actual.lower():
                self.logger.debug(f"Skipping chat (title mismatch): {chat_title_actual}")
                continue

            # Filter by time
            chat_time = chat.get("create_time")
            if chat_time and since and self.timestamp_compare(since, chat_time) > 0:
                self.logger.debug(f"Skipping old chat {chat_count} (time={chat_time})")
                continue

            self.logger.info(f"Processing chat {chat_count}: {chat_title_actual}")

            # Extract messages from this chat
            chat_messages = []
            for chat_map in chat.get("mapping", {}).values():
                message_data = self._extract_message_from_chat_map(chat_map)
                if message_data:
                    # Add chat metadata
                    message_data["chat_id"] = chat.get("id")
                    message_data["chat_title"] = chat_title_actual
                    message_data["chat_create_time"] = chat_time
                    chat_messages.append(message_data)

            # Sort messages by timestamp
            chat_messages.sort(key=lambda x: x.get("timestamp", 0))

            # Add to results
            for msg in chat_messages:
                messages.append(msg)
                msg_count += 1

                if limit and msg_count >= limit:
                    self.logger.info(f"Reached max messages limit: {msg_count}")
                    return messages

            self.logger.debug(
                f"Finished processing chat {chat_count}: {len(chat_messages)} messages"
            )

        self.logger.info(f"Total messages loaded: {len(messages)} from {chat_count} chats")
        return messages


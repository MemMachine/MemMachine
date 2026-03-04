"""
Locomo chat history parser.

This module provides the parser for Locomo chat history format.
"""

import datetime
from typing import Any

from .base import BaseParser


class LocomoParser(BaseParser):
    """Parser for Locomo chat history format."""

    def count_conversations(self, infile: str) -> int:
        """Count the number of conversations in the Locomo export file."""
        if self.verbose:
            self.logger.debug(f"Loading Locomo input file {infile}")
        data = self.load_json(infile)
        # Loop to count every session
        conv_count = 0
        section_count = 0
        done = False
        while not done:
            for section in data:
                section_count += 1
                if self.verbose:
                    self.logger.debug(
                        f"Looking for next conversation section {section_count}"
                    )
                if "conversation" in section:
                    conv_count += 1
                    if self.verbose:
                        self.logger.debug(f"Found conversation {conv_count}")
                if done:
                    break
            if self.verbose:
                self.logger.debug(f"Counted all sections={section_count}")
            done = True
        return conv_count

    def load(
        self,
        infile: str,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load messages from Locomo chat history.

        Args:
            infile: Path to Locomo chat history JSON file
            filters: Optional dictionary with filter parameters:
                - since: Only load messages after this timestamp (float)
                - index: Load only this conversation number (1-indexed, int)
                - limit: Maximum number of messages to load (int)

        Returns:
            List of message dictionaries, each containing:
            - speaker: Speaker name/identifier
            - timestamp: Message timestamp (Unix timestamp)
            - content: Same as text (for compatibility)
        """
        # Extract filter values from filters dict
        if filters is None:
            filters = {}

        since = filters.get("since", 0) or 0
        index = filters.get("index", 0) or 0
        limit = filters.get("limit", 0) or 0

        if self.verbose:
            self.logger.debug(f"since={since} index={index} limit={limit}")

        if self.verbose:
            self.logger.debug(f"Loading Locomo input file {infile}")

        data = self.load_json(infile)

        results = []
        conv_count = 0
        msg_count = 0
        section_count = 0
        done = False

        # Loop to load every session
        while not done:
            for section in data:
                section_count += 1
                if self.verbose:
                    self.logger.debug(
                        f"Looking for next conversation section {section_count}"
                    )

                if "conversation" in section:
                    conversation = section["conversation"]
                    conv_count += 1

                    if index and conv_count != index:
                        # User asked to do one specific conversation
                        continue

                    if self.verbose:
                        self.logger.debug(f"Loading conversation {conv_count}")

                    for num in range(1, 9999):
                        session_name = f"session_{num}"
                        session_date_name = f"session_{num}_date_time"

                        if session_name not in conversation:
                            # Processed all of the sessions in this conversation
                            if self.verbose:
                                self.logger.debug(
                                    f"Finished conversation {conv_count} (1)"
                                )
                            break

                        if self.verbose:
                            self.logger.debug(
                                f"Loading conversation {conv_count} session {num}"
                            )

                        messages = conversation[session_name]
                        session_date_str = ""
                        session_date_obj = None

                        if not session_date_obj:
                            try:
                                session_date_str = conversation[session_date_name]
                                session_date_obj = datetime.datetime.strptime(
                                    session_date_str, "%I:%M %p on %d %b, %Y"
                                )
                            except Exception:
                                pass

                        if not session_date_obj:
                            try:
                                session_date_str = conversation[session_date_name]
                                session_date_obj = datetime.datetime.strptime(
                                    session_date_str, "%I:%M %p on %d %B, %Y"
                                )
                            except Exception:
                                pass

                        try:
                            session_time = session_date_obj.timestamp()
                            if since:
                                if self.timestamp_compare(since, session_time) > 0:
                                    if self.verbose:
                                        self.logger.debug(
                                            f"Skipping old conversation {conv_count} session {num} time={session_time}"
                                        )
                                    break
                        except Exception:
                            if self.verbose:
                                self.logger.warning(
                                    f"ERROR: cannot read timestamp of conversation {conv_count} session {num} date={session_date_str}"
                                )

                        for message in messages:
                            results.append(
                                {
                                    "content": message.get("text", ""),
                                    "speaker": message.get("speaker", ""),
                                    "timestamp": session_date_obj.timestamp()
                                    if session_date_obj
                                    else 0,
                                }
                            )
                            msg_count += 1
                            if limit and msg_count >= limit:
                                done = True
                                break

                        if done:
                            break

                    if self.verbose:
                        self.logger.debug(
                            f"Finished conversation {conv_count} sessions={num}"
                        )

                if done:
                    break

            if self.verbose:
                self.logger.debug(f"Loaded all sections={section_count}")
            done = True

        return results

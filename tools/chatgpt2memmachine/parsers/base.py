"""
Base parser class for chat history parsers.

This module provides the base class and common utilities for parsing
different chat history formats.
"""

import datetime
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseParser(ABC):
    """
    Base class for chat history parsers.

    All parsers should inherit from this class and implement the abstract methods.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the parser.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this parser instance."""
        logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        logger.setLevel(logging.DEBUG if self.verbose else logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(handler)

        return logger

    @staticmethod
    def timestamp_compare(ts1: Any, ts2: Any) -> int:
        """
        Compare two timestamps.

        Args:
            ts1: First timestamp
            ts2: Second timestamp

        Returns:
            -1 if ts1 < ts2, 1 if ts1 > ts2, 0 if equal
        """
        ts1 = BaseParser.timestamp_ms_to_sec(ts1)
        ts2 = BaseParser.timestamp_ms_to_sec(ts2)
        if ts1 < ts2:
            return -1
        if ts1 > ts2:
            return 1
        return 0

    @staticmethod
    def timestamp_ms_to_sec(ts: Any) -> int:
        """
        Convert timestamp from milliseconds to seconds if needed.

        Args:
            ts: Timestamp (can be in seconds or milliseconds)

        Returns:
            Timestamp in seconds
        """
        if isinstance(ts, float):
            ts = int(ts)
        if ts > 9999999999:
            ts = int(ts / 1000)
        return ts

    @staticmethod
    def timestamp_to_obj(ts: Any) -> datetime.datetime:
        """
        Convert timestamp to datetime object.

        Args:
            ts: Timestamp (seconds or milliseconds)

        Returns:
            Datetime object
        """
        ts = BaseParser.timestamp_ms_to_sec(ts)
        return datetime.datetime.fromtimestamp(ts)

    def load_json(self, infile: str) -> Any:
        """
        Load JSON data from file.

        Args:
            infile: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        try:
            with open(infile, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except FileNotFoundError:
            self.logger.error(f"File not found: {infile}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in {infile}: {e}")
            raise

    @abstractmethod
    def count_conversations(self, infile: str) -> int:
        """
        Count the number of conversations in the input file.

        Args:
            infile: Path to input file

        Returns:
            Number of conversations
        """
        pass

    @abstractmethod
    def load(
        self,
        infile: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Load messages from the input file.

        Args:
            infile: Path to input file
            filters: Optional dictionary with filter parameters:
                - since: Only load messages after this timestamp (float)
                - index: Load only this conversation number (1-indexed, int)
                - limit: Maximum number of messages to load (int)
                - Additional parser-specific filter keys may be supported

        Returns:
            List of messages (format depends on parser implementation)
        """
        pass

    def validate(self, infile: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate file structure without processing.

        Args:
            infile: Path to input file

        Returns:
            Tuple of (is_valid, errors, warnings)
            - is_valid: True if file can be parsed without critical errors
            - errors: List of error messages
            - warnings: List of warning messages

        Note:
            Default implementation returns True with no errors/warnings.
            Subclasses should override this if validation is supported.
        """
        return True, [], []

    @staticmethod
    def dump_data(
        data: Any,
        output_format: str = "json",
        outfile: Optional[str] = None,
    ) -> None:
        """
        Output data in the specified format.

        Args:
            data: Data to output (list of strings or list of dicts)
            output_format: Output format ('json')
            outfile: Optional output file path (defaults to stdout)
        """
        if outfile:
            fp = open(outfile, "w", encoding="utf-8")
        else:
            fp = sys.stdout

        try:
            if output_format == "json":
                # JSON output
                json.dump(data, fp, indent=2, ensure_ascii=False)
                fp.write("\n")
            else:
                # Text output (default)
                for msg in data:
                    content = msg.get("content", "")
                    # Clean up content
                    content = content.strip()
                    content = re.sub(r"\\n", " ", content)
                    content = re.sub(r"\n", " ", content)
                    print(content, file=fp)
        finally:
            if outfile:
                fp.close()

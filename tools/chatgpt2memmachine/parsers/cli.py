"""
Main entry point for processing chat history files.

This script provides a command-line interface for parsing chat history
from various sources (OpenAI, Locomo, etc.).
"""

import argparse
import json
import os
import sys
from typing import Any, Optional

# Handle imports for both script execution and module import
try:
    from . import get_parser
    from ..utils import parse_time
except ImportError:
    # When run as a script, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from parsers import get_parser
    from utils import parse_time


def get_args():
    """
    Parse command-line arguments.
    
    Key features:
    - Standard argument names (--input, --output)
    - Unified filtering arguments (--index for both sources)
    - Outputs to JSON format only
    """
    parser = argparse.ArgumentParser(
        description="Parse chat history from various sources (OpenAI, Locomo, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse OpenAI chat history to JSON
  %(prog)s -i chat.json -o output.json --source openai
  
  # Parse with filters (only messages after date, limit to 100)
  %(prog)s -i chat.json -o output.json --since 2024-01-01 --limit 100
  
  # Count conversations
  %(prog)s -i chat.json --count
  
  # Validate file structure
  %(prog)s -i chat.json --validate --source openai
  
  # Parse specific conversation (index 1)
  %(prog)s -i chat.json -o output.json --index 1
        """.strip()
    )
    
    # Core arguments
    core_group = parser.add_argument_group('Core Arguments')
    core_group.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="FILE",
        help="Input chat history file (required)",
    )
    core_group.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Output JSON file (default: stdout)",
    )
    core_group.add_argument(
        "-s",
        "--source",
        choices=["openai", "locomo"],
        default="locomo",
        help="Source format: 'openai' or 'locomo' (default: %(default)s)",
    )
    core_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
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
        help="Maximum number of messages to process (0 = no limit, default: %(default)s)",
    )
    filter_group.add_argument(
        "--index",
        type=int,
        metavar="N",
        default=0,
        help="Process only the conversation/chat at index N (1-based, 0 = all). Works for both OpenAI and Locomo sources.",
    )
    # For OpenAI-specific: chat title filter
    filter_group.add_argument(
        "--chat-title",
        metavar="TITLE",
        help="[OpenAI only] Process only chats matching this title (case-insensitive)",
    )
    
    # Operation modes
    mode_group = parser.add_argument_group('Operation Modes')
    mode_group.add_argument(
        "--count",
        action="store_true",
        help="Count and display the number of conversations in the input file, then exit",
    )
    mode_group.add_argument(
        "--validate",
        action="store_true",
        help="Validate file structure without processing (OpenAI source only), then exit",
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
    
    if args.validate and args.source != "openai":
        parser.error("--validate is only supported for 'openai' source")
    
    return args


if __name__ == "__main__":
    args = get_args()
    args.source = args.source.lower()

    # Get parser instance
    try:
        parser = get_parser(args.source, args.verbose)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle count requests to return the number of conversations in the input file
    if args.count:
        count = parser.count_conversations(args.input)
        print(f"--> Number of conversations found: {count}")
        sys.exit(0)

    # Handle validation
    if args.validate:
        print("Running validation (dry-run mode)...", file=sys.stderr)
        is_valid, errors, warnings = parser.validate(args.input)

        # Print errors
        if errors:
            print("\nERRORS (required fields missing or invalid):", file=sys.stderr)
            for error in errors:
                print(f"  ERROR: {error}", file=sys.stderr)

        # Print warnings
        if warnings:
            print("\nWARNINGS (potential issues):", file=sys.stderr)
            for warning in warnings:
                print(f"  WARNING: {warning}", file=sys.stderr)

        # Summary
        print(f"\nValidation Summary:", file=sys.stderr)
        print(f"  Errors: {len(errors)}", file=sys.stderr)
        print(f"  Warnings: {len(warnings)}", file=sys.stderr)

        if is_valid:
            print("  Status: VALID - File structure is correct", file=sys.stderr)
            sys.exit(0)
        else:
            print("  Status: INVALID - File has critical errors", file=sys.stderr)
            sys.exit(1)

    # Load data
    filters = {
        "since": args.since,
        "index": args.index,
        "limit": args.limit,
    }

    # Add source-specific arguments
    if args.source == "openai" and args.chat_title:
        filters["chat_title"] = args.chat_title

    try:
        data = parser.load(args.input, filters=filters)
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}", file=sys.stderr)
        sys.exit(1)

    # Output data as JSON
    parser.dump_data(data, output_format="json", outfile=args.output)


# chatgpt2memmachine

Import chat history from external sources into MemMachine.

## Supported Sources

- **OpenAI**: ChatGPT conversation exports
- **Locomo**: Locomo chat history format

## Quick Start

### Basic Migration

Migrate OpenAI chat history to MemMachine:

```bash
python migration.py -i conversations.json --org-id my-org --project-id my-project --source openai
```

Migrate Locomo chat history:

```bash
python migration.py -i locomo_history.json --org-id my-org --project-id my-project --source locomo
```

## Command-Line Options

### Core Arguments

- `-i, --input FILE` (required): Input chat history file
- `-s, --source {openai,locomo}`: Source format (default: `openai`)
- `-v, --verbose`: Enable verbose/debug logging

### MemMachine Configuration

- `--base-url URL`: Base URL of the MemMachine API (default: `http://localhost:8080`)
- `--org-id ID`: Organization ID in MemMachine (optional, leave empty for default)
- `--project-id ID`: Project ID in MemMachine (optional, leave empty for default)

### Filtering Options

- `--since TIME`: Only process messages after this time. Supports:
  - Unix timestamp
  - ISO format: `YYYY-MM-DDTHH:MM:SS`
  - Date only: `YYYY-MM-DD`
- `-l, --limit N`: Maximum number of messages to process per conversation (0 = no limit, default: 0)
- `--index N`: Process only the conversation/chat at index N (1-based, 0 = all)
- `--chat-title TITLE`: [OpenAI only] Process only chats matching this title (case-insensitive)
- `--user-only`: Only add user messages to MemMachine (exclude assistant messages, default: enabled)

### Operation Modes

- `--dry-run`: Preview what would be migrated without actually adding memories
- `--workers N`: Number of worker threads for parallel processing (default: 1, sequential)
- `--run-id ID`: Specify the run ID from a previous execution (required with `--retry-failed` or `--resume`)
- `--retry-failed`: Retry only messages that failed in a previous run (requires `--run-id`)
- `--resume`: Skip messages that succeeded in a previous run (requires `--run-id`)

## Examples

### Basic Migration

```bash
# Migrate OpenAI chat history
python migration.py -i chat.json --org-id my-org --project-id my-project --source openai

# Migrate Locomo chat history
python migration.py -i locomo.json --org-id my-org --project-id my-project --source locomo
```

### Filtered Migration

```bash
# Only process messages after a specific date, limit to 100 messages per conversation
python migration.py -i chat.json --org-id my-org --project-id my-project --since 2024-01-01 -l 100

# Process only a specific conversation (index 1) with verbose output
python migration.py -i chat.json --org-id my-org --project-id my-project --index 1 -v

# Process only chats matching a specific title (OpenAI only)
python migration.py -i chat.json --org-id my-org --project-id my-project --chat-title "My Project"
```

### Preview and Testing

```bash
# Dry run to preview what would be migrated
python migration.py -i chat.json --org-id my-org --project-id my-project --dry-run
```

### Retry and Resume

```bash
# Retry failed messages from a previous run
python migration.py -i chat.json --org-id my-org --project-id my-project \
  --run-id 20260202T235246858124 --retry-failed

# Resume migration, skipping already successful messages
python migration.py -i chat.json --org-id my-org --project-id my-project \
  --run-id 20260202T235246858124 --resume
```

### Performance Optimization

```bash
# Use parallel processing with 4 workers for faster migration
python migration.py -i chat.json --org-id my-org --project-id my-project --workers 4
```

## Output Files

Each migration run generates a unique run ID and creates the following files in the `output/` directory:

- `success_{run_id}.txt`: List of successfully migrated message IDs (format: `conv_id:message_id`)
- `errors_{run_id}.txt`: List of failed message IDs with error details
- `migration_stats_{run_id}.json`: Migration statistics (conversations, messages, success/failure counts, duration)
- `api_requests_{run_id}.csv`: API request logs (when verbose mode is enabled)
- `trace_{run_id}.txt`: Detailed API trace logs (when verbose mode is enabled)

Extracted conversations are saved in the `extracted/` directory for caching.

## Getting Help

For detailed help and all available options:

```bash
python migration.py --help
```

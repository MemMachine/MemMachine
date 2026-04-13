# Retrieval-Agent Benchmark Configuration

All retrieval-agent benchmarks are driven by a single **`configuration.yml`** file
placed in this directory (`evaluation/retrieval_agent/configuration.yml`).

The file controls every component used during a run:

| Concern | Config section |
|---|---|
| Language model for the retrieval agent (retrieval/planning) | `retrieval_agent.llm_model` |
| Language model for answer generation (answer generation) | `retrieval_agent.answer_llm_model` (falls back to `llm_model`) |
| Language model for the LLM judge (evaluation) | `retrieval_agent.judge_llm_model` (falls back to `llm_model`) |

| Embedder for long-term memory | `episodic_memory.long_term_memory.embedder` |
| Reranker | `retrieval_agent.reranker` or `episodic_memory.long_term_memory.reranker` |
| Vector graph store (Neo4j) | `episodic_memory.long_term_memory.vector_graph_store` |
| All resource definitions | `resources.embedders`, `resources.language_models`, `resources.rerankers`, `resources.databases` |

`run_test.sh` checks for the file at startup and exits with an error if it is
missing.

The MemMachine configuration schema also requires a `semantic_memory` section.
The samples below keep semantic memory disabled because these retrieval-agent
benchmarks do not use it directly, but `semantic_memory.config_database` must
still reference a valid SQL database ID.

---

## Quick Start

1. Copy one of the sample configurations below into
   `evaluation/retrieval_agent/configuration.yml`.
2. Fill in your API keys / connection details.
3. Install the benchmark dependencies:

```sh
cd evaluation/retrieval_agent
python -m pip install -r requirements.txt
```

`requirements.txt` installs the local MemMachine packages used by these
scripts plus `pandas` for `generate_scores.py`. `run_test.sh` checks for the
required modules before starting any search run.

Existing concurrency controls are exposed as named flags in `run_test.sh`:
`--ingest-concurrency`, `--search-concurrency`, and `--judge-concurrency`.

4. Run a benchmark:

```sh
cd evaluation/retrieval_agent
./run_test.sh wikimultihop exp1 ingest retrieval_agent 100
./run_test.sh wikimultihop exp1 search retrieval_agent 100
```

---

## Configuration Samples

### Sample 1 — OpenAI models + AWS Bedrock reranker (default setup)

```yaml
episode_store:
  database: profile_storage
  with_count_cache: true

episodic_memory:
  enabled: true
  long_term_memory:
    embedder: openai_embedder
    reranker: aws_reranker_id
    vector_graph_store: my_storage_id
  long_term_memory_enabled: true
  short_term_memory:
    llm_model: openai_model
    message_capacity: 500
    summary_prompt_system: "You are an AI agent that can make summary for a list of episodes."
    summary_prompt_user: "Summarize: {summary}\n{episodes}\nYour summary (under {max_length} words):"
  short_term_memory_enabled: true

logging:
  level: INFO

retrieval_agent:
  llm_model: openai_model
  reranker: aws_reranker_id

semantic_memory:
  enabled: false
  config_database: profile_storage

resources:
  databases:
    my_storage_id:
      provider: neo4j
      config:
        uri: bolt://localhost:7687
        user: neo4j
        password: neo4j_password

    profile_storage:
      provider: postgres
      config:
        dialect: postgresql
        driver: asyncpg
        host: localhost
        port: 5432
        user: memmachine
        password: memmachine_password
        db_name: memmachine

  embedders:
    openai_embedder:
      provider: openai
      config:
        api_key: sk-...
        base_url: https://api.openai.com/v1
        model: text-embedding-3-small
        dimensions: 1536

  language_models:
    openai_model:
      provider: openai-responses
      config:
        api_key: sk-...
        base_url: https://api.openai.com/v1
        model: gpt-4o-mini

  rerankers:
    aws_reranker_id:
      provider: amazon-bedrock
      config:
        region: us-west-2
        aws_access_key_id: AKIA...
        aws_secret_access_key: ...
        model_id: amazon.rerank-v1:0

session_manager:
  database: profile_storage
```

---

### Sample 2 — Ollama (local) models + BM25 reranker

Use this when you run models locally via [Ollama](https://ollama.com/).

```yaml
episode_store:
  database: sqlite_db
  with_count_cache: true

episodic_memory:
  enabled: true
  long_term_memory:
    embedder: ollama_embedder
    reranker: bm25_reranker
    vector_graph_store: my_storage_id
  long_term_memory_enabled: true
  short_term_memory:
    llm_model: ollama_model
    message_capacity: 500
    summary_prompt_system: "You are an AI agent that summarizes episodes."
    summary_prompt_user: "Summarize: {summary}\n{episodes}\nYour summary (under {max_length} words):"
  short_term_memory_enabled: true

logging:
  level: INFO

retrieval_agent:
  llm_model: ollama_model
  reranker: bm25_reranker

semantic_memory:
  enabled: false
  config_database: sqlite_db

resources:
  databases:
    my_storage_id:
      provider: neo4j
      config:
        uri: bolt://localhost:7687
        user: neo4j
        password: neo4j_password

    sqlite_db:
      provider: sqlite
      config:
        dialect: sqlite
        driver: aiosqlite
        path: evaluation_session.db

  embedders:
    ollama_embedder:
      provider: openai          # Ollama exposes an OpenAI-compatible endpoint
      config:
        api_key: EMPTY
        base_url: http://localhost:11434/v1
        model: nomic-embed-text
        dimensions: 768

  language_models:
    ollama_model:
      provider: openai-chat-completions
      config:
        api_key: EMPTY
        base_url: http://localhost:11434/v1
        model: llama3.2

  rerankers:
    bm25_reranker:
      provider: bm25
      config:
        k1: 1.5
        b: 0.75
        epsilon: 0.25
        language: english
        tokenizer: default

session_manager:
  database: sqlite_db
```

---

### Sample 3 — AWS Bedrock (end-to-end)

```yaml
episode_store:
  database: profile_storage
  with_count_cache: true

episodic_memory:
  enabled: true
  long_term_memory:
    embedder: aws_embedder
    reranker: aws_reranker_id
    vector_graph_store: my_storage_id
  long_term_memory_enabled: true
  short_term_memory:
    llm_model: aws_model
    message_capacity: 500
    summary_prompt_system: "You are an AI agent that summarizes episodes."
    summary_prompt_user: "Summarize: {summary}\n{episodes}\nYour summary (under {max_length} words):"
  short_term_memory_enabled: true

logging:
  level: INFO

retrieval_agent:
  llm_model: aws_model
  reranker: aws_reranker_id

semantic_memory:
  enabled: false
  config_database: profile_storage

resources:
  databases:
    my_storage_id:
      provider: neo4j
      config:
        uri: bolt://localhost:7687
        user: neo4j
        password: neo4j_password

    profile_storage:
      provider: postgres
      config:
        dialect: postgresql
        driver: asyncpg
        host: localhost
        port: 5432
        user: memmachine
        password: memmachine_password
        db_name: memmachine

  embedders:
    aws_embedder:
      provider: amazon-bedrock
      config:
        region: us-west-2
        aws_access_key_id: AKIA...
        aws_secret_access_key: ...
        model_id: amazon.titan-embed-text-v2:0

  language_models:
    aws_model:
      provider: amazon-bedrock
      config:
        region: us-west-2
        aws_access_key_id: AKIA...
        aws_secret_access_key: ...
        model_id: anthropic.claude-3-5-sonnet-20241022-v2:0

  rerankers:
    aws_reranker_id:
      provider: amazon-bedrock
      config:
        region: us-west-2
        aws_access_key_id: AKIA...
        aws_secret_access_key: ...
        model_id: amazon.rerank-v1:0

session_manager:
  database: profile_storage
```

---

### Sample 4 — OpenAI-compatible endpoint (vLLM / any provider)

Works with any server that speaks the OpenAI Chat Completions protocol.

```yaml
episode_store:
  database: sqlite_db
  with_count_cache: true

episodic_memory:
  enabled: true
  long_term_memory:
    embedder: custom_embedder
    reranker: bm25_reranker
    vector_graph_store: my_storage_id
  long_term_memory_enabled: true
  short_term_memory:
    llm_model: custom_model
    message_capacity: 500
    summary_prompt_system: "You are an AI agent that summarizes episodes."
    summary_prompt_user: "Summarize: {summary}\n{episodes}\nYour summary (under {max_length} words):"
  short_term_memory_enabled: true

logging:
  level: INFO

retrieval_agent:
  llm_model: custom_model
  reranker: bm25_reranker

semantic_memory:
  enabled: false
  config_database: sqlite_db

resources:
  databases:
    my_storage_id:
      provider: neo4j
      config:
        uri: bolt://localhost:7687
        user: neo4j
        password: neo4j_password

    sqlite_db:
      provider: sqlite
      config:
        dialect: sqlite
        driver: aiosqlite
        path: evaluation_session.db

  embedders:
    custom_embedder:
      provider: openai
      config:
        api_key: your-api-key
        base_url: http://your-vllm-host:8000/v1
        model: your-embedding-model
        dimensions: 1536

  language_models:
    custom_model:
      provider: openai-chat-completions
      config:
        api_key: your-api-key
        base_url: http://your-vllm-host:8000/v1
        model: your-chat-model

  rerankers:
    bm25_reranker:
      provider: bm25
      config:
        k1: 1.5
        b: 0.75
        epsilon: 0.25
        language: english
        tokenizer: default

session_manager:
  database: sqlite_db
```

---

### Sample 5 — Support role-specific LLM configuration in retrieval-agent benchmarks

Works with any server that speaks the OpenAI Chat Completions protocol.

```yaml
episode_store:
  database: sqlite_db
  with_count_cache: true

episodic_memory:
  enabled: true
  long_term_memory:
    embedder: custom_embedder
    reranker: bm25_reranker
    vector_graph_store: my_storage_id
  long_term_memory_enabled: true
  short_term_memory:
    llm_model: custom_model
    message_capacity: 500
    summary_prompt_system: "You are an AI agent that summarizes episodes."
    summary_prompt_user: "Summarize: {summary}\n{episodes}\nYour summary (under {max_length} words):"
  short_term_memory_enabled: true

logging:
  level: INFO

retrieval_agent:
  llm_model: agent_model         # Used for retrieval/planning agent
  answer_llm_model: answer_model  # Optional: used for answer generation (falls back to llm_model)
  judge_llm_model: judge_model   # Optional: used for LLM judge (falls back to llm_model)
  reranker: bm25_reranker

semantic_memory:
  llm_model: openai_compatible_model
  embedding_model: openai_compatible_embedder
  database: profile_storage
  config_database: profile_storage

resources:
  databases:
    my_storage_id:
      provider: neo4j
      config:
        uri: bolt://localhost:7687
        user: neo4j
        password: neo4j_password

    sqlite_db:
      provider: sqlite
      config:
        dialect: sqlite
        driver: aiosqlite
        path: evaluation_session.db

  embedders:
    custom_embedder:
      provider: openai
      config:
        api_key: your-api-key
        base_url: http://your-vllm-host:8000/v1
        model: your-embedding-model
        dimensions: 1536

  language_models:
    openai_compatible_model:
      provider: openai-chat-completions
      config:
        api_key: your-api-key
        base_url: http://your-vllm-host:8000/v1
        model: your-llm-model

    answer_model:
      provider: openai-chat-completions
      config:
        api_key: your-api-key
        base_url: http://your-vllm-host:8000/v1
        model: your-llm-answer-model

    judge_model:
      provider: openai-chat-completions
      config:
        api_key: your-api-key
        base_url: http://your-vllm-host:8000/v1
        model: your-llm-judge-model

  rerankers:
    bm25_reranker:
      provider: bm25
      config:
        k1: 1.5
        b: 0.75
        epsilon: 0.25
        language: english
        tokenizer: default

session_manager:
  database: sqlite_db
```

---

## Key Fields Reference

### `retrieval_agent`

| Field | Description |
|---|---|
| `llm_model` | ID of the language model used by the retrieval agent for planning and tool selection. Also used as the default for answer generation and LLM judge if the specific fields below are not set. Must match a key under `resources.language_models`. |
| `answer_llm_model` | Optional. ID of the language model used for answer generation. If not set, falls back to `llm_model`. |
| `judge_llm_model` | Optional. ID of the language model used by the LLM judge during evaluation. If not set, falls back to `llm_model`. |
| `reranker` | ID of the reranker used by the retrieval agent. Overrides `episodic_memory.long_term_memory.reranker` when set. |

### `episodic_memory.long_term_memory`

| Field | Description |
|---|---|
| `embedder` | ID of the embedder used to index and search episodes. |
| `reranker` | Fallback reranker if `retrieval_agent.reranker` is not set. |
| `vector_graph_store` | ID of the Neo4j database used as the vector store. |

### `resources.language_models` — provider options

| Provider | Notes |
|---|---|
| `openai-responses` | OpenAI Responses API (gpt-4o, gpt-4o-mini, etc.) |
| `openai-chat-completions` | OpenAI Chat Completions API; also works with Ollama, vLLM, and any OpenAI-compatible endpoint |
| `amazon-bedrock` | AWS Bedrock Converse API |

### `resources.embedders` — provider options

| Provider | Notes |
|---|---|
| `openai` | OpenAI embeddings; also compatible with Ollama (`nomic-embed-text`, etc.) and other OpenAI-compatible endpoints |
| `amazon-bedrock` | AWS Bedrock embeddings |

### `resources.rerankers` — provider options

| Provider | Notes |
|---|---|
| `amazon-bedrock` | AWS Bedrock reranker |
| `bm25` | Local BM25 reranker, no external service needed |
| `cohere` | Cohere reranker (requires `cohere_key`) |
| `rrf-hybrid` | Reciprocal Rank Fusion combining multiple rerankers |

---

## Running Benchmarks

From `evaluation/retrieval_agent/`:

```sh
# WikiMultiHop — ingest then search 500 questions
./run_test.sh wikimultihop exp1 ingest retrieval_agent 500
./run_test.sh wikimultihop exp1 search retrieval_agent 500
./run_test.sh wikimultihop exp1 search retrieval_agent 500 --search-concurrency 2 --judge-concurrency 4

# HotpotQA validation set — 200 questions
./run_test.sh hotpotqa exp1 ingest validation retrieval_agent 200
./run_test.sh hotpotqa exp1 search validation retrieval_agent 200

# LoCoMo
./run_test.sh locomo exp1 ingest retrieval_agent
./run_test.sh locomo exp1 ingest retrieval_agent --ingest-concurrency 2
./run_test.sh locomo exp1 search retrieval_agent
./run_test.sh locomo exp1 search retrieval_agent --search-concurrency 1 --judge-concurrency 4

# BEAM — requires explicit data paths
./run_test.sh beam exp1 ingest retrieval_agent /path/to/chat.json /path/to/probing_questions.json
./run_test.sh beam exp1 search retrieval_agent /path/to/chat.json /path/to/probing_questions.json
./run_test.sh beam exp1 search retrieval_agent /path/to/chat.json /path/to/probing_questions.json --search-concurrency 10 --judge-concurrency 30
```

### Downloading BEAM Dataset

Due to the large size of the BEAM dataset, it is hosted externally on Hugging Face for download and provided with an option for selective download based on your needs.
Download and convert it using:

```bash
cd evaluation/data

# Download specific sizes (100K, 500K, or 1M)
python beam_download.py --size 100K --output ./beam
python beam_download.py --size 500K --output ./beam
python beam_download.py --size 1M --output ./beam

# Download multiple sizes at once
python beam_download.py --size 100K 500K 1M --output ./beam
```

**Note**: 10M dataset is not supported in this script.


For the full argument reference run:

```sh
./run_test.sh --help
./run_test.sh wikimultihop --help
./run_test.sh beam --help
```

---

## Full Search Test with Three Separate LLM Models

The **search** phase uses three distinct LLM roles:

| Role | Config Field | Purpose |
|------|--------------|---------|
| **Agent/Planner** | `retrieval_agent.llm_model` | Runs the retrieval agent for planning and tool selection |
| **Answer Generator** | `retrieval_agent.answer_llm_model` | Generates answers to questions (falls back to `llm_model` if not set) |
| **Judge** | `retrieval_agent.judge_llm_model` | Evaluates generated answers against gold answers (falls back to `llm_model` if not set) |

### Example: Testing with Three Different Models

Configure your `configuration.yml` with three separate models:

```yaml
retrieval_agent:
  llm_model: agent_model       # For agent/planning
  answer_llm_model: answer_model  # For answer generation
  judge_llm_model: judge_model    # For evaluation

resources:
  language_models:
    agent_model:
      provider: openai-chat-completions
      config:
        model: "gpt-4o-mini"
        api_key: your-api-key
        base_url: https://api.openai.com/v1

    answer_model:
      provider: openai-chat-completions
      config:
        model: "gpt-4o"
        api_key: your-api-key
        base_url: https://api.openai.com/v1

    judge_model:
      provider: openai-chat-completions
      config:
        model: "gpt-4.1"
        api_key: your-api-key
        base_url: https://api.openai.com/v1
```

Then run the full search test:

```sh
# Ingest data first
./run_test.sh wikimultihop multi_model ingest retrieval_agent 100

# Search with all three models (agent + answer + judge)
./run_test.sh wikimultihop multi_model search retrieval_agent 100 \
    --search-concurrency 1 --judge-concurrency 2
```

The evaluation pipeline:
1. **Agent** (`llm_model`) processes the question and retrieves memories
2. **Answer model** (`answer_llm_model`) generates the final answer
3. **Judge model** (`judge_llm_model`) evaluates if the answer is CORRECT or WRONG

### Output Files

After a search run, you'll get:

```
result/
├── wikimultihop_retrieval_agent_output_multi_model.json      # Generated answers
└── wikimultihop_retrieval_agent_evaluation_metrics_multi_model.json  # LLM judge scores
```

The evaluation metrics file contains per-category accuracy scores for each question and category.

---

## Deleting Ingested Data

`run_test.sh delete` removes all episodes a prior `ingest` wrote to the configured
long-term memory store for each benchmark's session key(s). Use it between runs to
reuse the same `RESULT_POSTFIX` without double-counting earlier data.

```sh
# From evaluation/retrieval_agent/
./run_test.sh locomo       exp1 delete retrieval_agent
./run_test.sh wikimultihop exp1 delete retrieval_agent
./run_test.sh hotpotqa     exp1 delete retrieval_agent
./run_test.sh longmemeval  exp1 delete retrieval_agent
```

Notes:

- `delete` ignores `SPLIT_NAME` and `LENGTH` — only `RESULT_POSTFIX` and
  `TEST_TARGET` are needed. `TEST_TARGET` is accepted for argument symmetry with
  `ingest`/`search` but is unused by the delete path.
- For LoCoMo the delete iterates all 10 conversation groups (`group_0` … `group_9`),
  since ingestion uses one session per conversation.
- For WikiMultiHop / HotpotQA the delete clears the single fixed session_id used
  by ingestion (`group1` and `hotpotqa_group` respectively).
- For LongMemEval the session_id is derived from `RESULT_POSTFIX`
  (`longmemeval_<RESULT_POSTFIX>`), matching the ingest-time session.
- Concurrency flags (`--ingest-concurrency`, `--search-concurrency`,
  `--judge-concurrency`) are rejected with `delete`.

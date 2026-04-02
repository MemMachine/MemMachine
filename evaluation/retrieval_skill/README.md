# Retrieval Skill Benchmarks

Evaluation harness for MemMachine retrieval skill across four datasets:
HotpotQA, WikiMultiHop, LongMemEval, and LoCoMo.

## Configurable Models

The **answer model** (generates answers from retrieved memories) and the
**evaluation model** (LLM judge that scores correctness) can be configured
via `benchmark_config.yml`.

### Config file format

The format mirrors the MemMachine server `configuration.yml` language model
section. Each entry requires a `provider` and a `config` block:

```yaml
answer_model:
  provider: <provider>
  config:
    # provider-specific fields

evaluation_model:
  provider: <provider>
  config:
    # provider-specific fields
```

Environment variables can be referenced with `${VAR}` or `$VAR` syntax.

### Supported providers

| Provider | Description | Key config fields |
|---|---|---|
| `openai-responses` | OpenAI Responses API | `api_key`, `base_url`, `model` |
| `openai-chat-completions` | OpenAI Chat Completions API (also Ollama) | `api_key`, `base_url`, `model` |
| `amazon-bedrock` | AWS Bedrock Converse API | `model_id`, `region`, `aws_access_key_id`, `aws_secret_access_key` |

### Examples

**OpenAI (default)**

```yaml
answer_model:
  provider: openai-responses
  config:
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
    model: gpt-5-mini

evaluation_model:
  provider: openai-responses
  config:
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
    model: gpt-5-mini
```

**Ollama (via `openai-chat-completions`)**

```yaml
answer_model:
  provider: openai-chat-completions
  config:
    api_key: EMPTY
    base_url: http://localhost:11434/v1
    model: qwen3.5:9b

evaluation_model:
  provider: openai-chat-completions
  config:
    api_key: EMPTY
    base_url: http://localhost:11434/v1
    model: qwen3.5:9b
```

**Amazon Bedrock**

```yaml
answer_model:
  provider: amazon-bedrock
  config:
    model_id: openai.gpt-oss-20b-1:0
    region: us-west-2
    aws_access_key_id: ${AWS_ACCESS_KEY_ID}
    aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY}

evaluation_model:
  provider: amazon-bedrock
  config:
    model_id: openai.gpt-oss-20b-1:0
    region: us-west-2
    aws_access_key_id: ${AWS_ACCESS_KEY_ID}
    aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY}
```

**Mixed providers** -- use different providers for answer and evaluation:

```yaml
answer_model:
  provider: openai-chat-completions
  config:
    api_key: EMPTY
    base_url: http://localhost:11434/v1
    model: qwen3.5:9b

evaluation_model:
  provider: openai-responses
  config:
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
    model: gpt-5-mini
```

## Usage

### Via `run_test.sh`

Pass `--config path/to/benchmark_config.yml` as an optional flag:

```bash
# With config
./run_test.sh hotpotqa run1 search validation retrieval_skill 30 \
    --config benchmark_config.yml

# Without config (uses OPENAI_API_KEY env var, gpt-5-mini, OpenAI Responses API)
./run_test.sh hotpotqa run1 search validation retrieval_skill 30
```

The `--config` flag is passed to both the search script and the evaluation
(LLM judge) script.

### Via individual Python scripts

Each test script accepts `--config`:

```bash
uv run python hotpotQA_test.py \
    --run-type search \
    --test-target llm \
    --length 30 \
    --split-name validation \
    --config benchmark_config.yml
```

The standalone evaluator and LLM judge also accept `--config`:

```bash
uv run python evaluate.py --data-path results.json --target-path eval.json --config benchmark_config.yml
uv run python llm_judge.py --input_file results.json --config benchmark_config.yml
```

### Backward compatibility

When `--config` is not provided, all scripts fall back to the previous
behavior: OpenAI Responses API with the `OPENAI_API_KEY` environment variable
and `gpt-5-mini` model.

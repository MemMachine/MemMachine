# LoCoMo

## Tool-Specific Prerequisites

- Please ensure your `cfg.yml` file has been copied into your `episodic_memory` directory (`/memmachine/evaluation/episodic_memory/`) and renamed to `locomo_config.yaml`.


## Running the Benchmark

Ready to go? Follow these simple steps:

**A.** All commands should be run from their respective tool directory (default `evaluation/episodic_memory/`).

**B.** The path to your data file, `locomo10.json`, should be updated to match its location. By default, you can find it in `/memmachine/evaluation/data/`.

**C.** Once you have performed step 1 below, you can repeat the benchmark run by performing steps 2-4.  Once are you finished performing the benchmark, run step 5.

**Note:** For the recommended retrieval-agent benchmark workflow and
cross-benchmark command references, see `evaluation/README.md`.

### Step 1: Ingest a Conversation

First, let's add conversation data to MemMachine. This only needs to be done once per test run.
```sh
python locomo_ingest.py --data-path path/to/locomo10.json
```

### Step 2: Search the Conversation

Let's search through the data you just added.
```sh
python locomo_search.py --data-path path/to/locomo10.json --target-path results.json
```

### Step 3: Evaluate the Responses

Next, run a LoCoMo evaluation against the search results.
```sh
python locomo_evaluate.py --data-path results.json --target-path evaluation_metrics.json
```

### Step 4: Generate Your Final Score

Once the evaluation is complete, you can generate the final scores.
```sh
python generate_scores.py
```

The output will be a table in your shell showing the mean scores for each category and an overall score, like the example below:
```sh
Mean Scores Per Category:
          llm_score  count         type
category
1            0.8050    282    multi_hop
2            0.7259    321     temporal
3            0.6458     96  open_domain
4            0.9334    841   single_hop

Overall Mean Scores:
llm_score    0.8487
dtype: float64
```

### Step 5: Clean Up Your Data

When you're finished, you may want to delete the test data.
```sh
python locomo_delete.py --data-path path/to/locomo10.json
```

# LongMemEval

## Compatibility Note

The scripts use lower-level MemMachine server episodic memory directly without the same metadata and data structuring as the full MemMachine server.
As such, the data ingested will not be compatible with the full MemMachine server, and no config file is needed.

## Running the Benchmark

We evaluate on `longmemeval_s_cleaned.json`.

Get the LongMemEval dataset:
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/main

No config file is needed.

Set up Neo4j by any method.

> [! WARNING]
> We don't provide a way to clean up just the data ingested for LongMemEval, so we highly recommend against ingesting the data into a database with existing data.


> [! WARNING]
> As configured in the scripts, ~50GB of free space on disk is required. Ingestion will take ~1.5 hours. Required disk and ingestion time can be reduced by a factor of ~5x by setting `message_sentence_chunking=False` in the ingestion script, potentially with slightly lower scores (~1-2% lower).

Enable Cohere 3.5 reranker for your AWS account.

The evaluation scripts use Cohere 3.5 Reranker from AWS Bedrock.
The reranker can be switched in the script by importing and setting a different reranker.

Set the following environment variables:

- `AWS_ACCESS_KEY_ID`: for reranker

- `AWS_SECRET_ACCESS_KEY`: for reranker

- `NEO4J_URI`: URI for the Neo4j database

- `NEO4J_USERNAME`: as configured for the Neo4j database

- `NEO4J_PASSWORD`: as configured for the Neo4j database

- `OPENAI_API_KEY`: for embeddings, answer generation, and LLM-as-a-judge scoring

### Step 1: Ingest LongMemEval

```sh
python longmemeval_ingest.py --data-path path/to/longmemeval_s_cleaned.json
```

### Step 2: Query the Memory and Generate Responses to Questions

```sh
python longmemeval_search.py --data-path path/to/longmemeval_s_cleaned.json --target-path search.json
```

It may be useful to direct the stdout output to a file e.g.

```sh
python longmemeval_search.py --data-path path/to/longmemeval_s_cleaned.json --target-path search.json > search.out
```

### Step 3: Evaluate the Responses

```sh
python longmemeval_evaluate.py --data-path search.json --target-path eval.json
```

### Step 4: Print Scores

```sh
python lme_generate.py --data-path eval.json
```

```sh
overall: 0.9580
multi-session: 0.9323
temporal-reasoning: 0.9624
single-session-user: 0.9857
single-session-assistant: 1.0000
single-session-preference: 0.9667
```

## Tool-Specific Prerequisites

- Please ensure your `cfg.yml` file has been copied into your `episodic_memory` directory (`/memmachine/evaluation/episodic_memory/`) and renamed to `locomo_config.yaml`.


## Running the Benchmark

Ready to go? Follow these simple steps:

**A.** All commands should be run from their respective tool directory (default `evaluation/episodic_memory/`).

**B.** The path to your data file, `locomo10.json`, should be updated to match its location. By default, you can find it in `/memmachine/evaluation/data/`.

**C.** Once you have performed step 1 below, you can repeat the benchmark run by performing steps 2-4.  Once are you finished performing the benchmark, run step 5.

**Note:** Please refer to the [Episodic Memory Tool Set](./locomo/episodic_memory/README.md) for exact commands.

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

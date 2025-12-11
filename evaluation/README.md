# Benchmark Evaluations: A Guide to Testing Your MemMachine

Welcome to the MemMachine evaluation toolsets! We've created a simple tool to help you measure the performance, response quality of your MemMachine instance, and generate a LoCoMo score for your system.

**Episodic Memory Tool Set:** This tool measures how fast and accurately MemMachine performs core episodic memory tasks. For a list of specific commands, check out the [Episodic Memory Tool Set](./locomo/episodic_memory/README.md).


## Getting Started

Before you run any benchmarks, you'll need to set up your environment.

**General Prerequisites:**

- **MemMachine Backend:** Both tools require that your MemMachine backend be installed and configured. If you need help with this, you can check out our [QuickStart Guide](http://docs.memmachine.ai/getting_started/quickstart).

- **Start the Backend:** Once everything is set up, start MemMachine with this command:

  ```sh
  memmachine-server
  ```

**Tool-Specific Prerequisites:**

- Please ensure your `cfg.yml` file has been copied into your `locomo` directory (`/memmachine/evaluation/locomo/`) and renamed to `locomo_config.yaml`.


## Running the Benchmark

Ready to go? Follow these simple steps:

**A.** All commands should be run from their respective tool directory (default `locomo/episodic_memory/`).

**B.** The path to your data file, `locomo10.json`, should be updated to match its location. By default, you can find it in `/memmachine/evaluation/locomo/`.

**C.** Once you have performed step 1 below, you can repeat the benchmark run by performing steps 2-4.  Once are you finished performing the benchmark, run step 5.

**Note:** Please refer to the [Episodic Memory Tool Set](./locomo/episodic_memory/README.md) for exact commands.

### Step 1: Ingest a Conversation

First, let's add conversation data to MemMachine. This only needs to be done once per test run.

### Step 2: Search the Conversation

Let's search through the data you just added.

### Step 3: Evaluate the Responses

Next, run a LoCoMo evaluation against the search results.

### Step 4: Generate Your Final Score

Once the evaluation is complete, you can generate the final scores.

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

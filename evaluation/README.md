# Benchmark Evaluation Guide

This folder contains evaluation scripts for measuring MemMachine retrieval and
memory quality on benchmark datasets.

## Benchmark Suites

- `retrieval_skill` (recommended): Current evaluation pipeline for retrieval
  behavior and answer quality. Uses the shared provider-native `SkillRunner`
  against MemMachine search.
- `episodic_memory` (legacy): Earlier LoCoMo dataset episodic memory benchmark workflow. Uses
  both MemMachine REST API and Python SDK.

## Retrieval-Agent Modes

The retrieval-skill benchmarks currently support two test targets:

1. `retrieval_skill`: MemMachine retrieval with top-level retrieval-agent
   orchestration.
2. `llm`: Pure LLM baseline without MemMachine retrieval
   (full session content provided by dataset context).

`run_test.sh` no longer exposes the older non-agent `memmachine` target.

## Prerequisites

- MemMachine backend is installed and configured.
- Start MemMachine before running benchmarks. Run from `memmachine/` root dir:

```sh
./memmachine-compose.sh start
```

- If you use the legacy episodic workflow, copy your `cfg.yml` into
  `evaluation/episodic_memory/` and rename it to `locomo_config.yaml`.

## Run Retrieval-Agent Benchmarks (Recommended)

Run from `evaluation/retrieval_skill/`:

```sh
./run_test.sh <test> <test_specific_args> ...
```

For full argument details, run:

```sh
./run_test.sh --help
./run_test.sh locomo --help
./run_test.sh wikimultihop --help
./run_test.sh hotpotqa --help
```

For answer-model and judge-model configuration, see
`evaluation/retrieval_skill/README.md`.

Examples:

- LoCoMo ingest:

```sh
./run_test.sh locomo exp1 ingest retrieval_skill
```

- LoCoMo search + scoring:

```sh
./run_test.sh locomo exp1 search retrieval_skill
```

- WikiMultiHop search (500 examples):

```sh
./run_test.sh wikimultihop exp1 search retrieval_skill 500
```

- HotpotQA validation set search (200 examples):

```sh
./run_test.sh hotpotqa exp1 search validation retrieval_skill 200
```

Sample output:
```sh
Mean Scores Per Category:
            llm_score  count
category
bridge         0.9307    404
comparison     0.9375     96

Mean Scores Per Level:
       llm_score  count
level
hard       0.932    500
Overall Mean Scores:
llm_score    0.932
dtype: float64
--------------------------------
Tools Overall Accuracy:
Tool: ChainOfQueryAgent
  Accuracy: 167/181 = 92.27%
Tool: MemMachineSearch
  Accuracy: 299/319 = 93.73%
--------------------------------
HotpotQA Info Matrix:
hotpotqa Recall: 1116/1209 = 92.31%
hotpotqa Precision: 1116/4997 = 22.33%
hotpotqa Average Episodes Retrieved per Question: 9.99
Tool: ChainOfQueryAgent
    Recall: 427/448 = 95.31%
    Precision: 427/1810 = 23.59%
    Avg Episodes Retrieved per Question: 10.00
    Avg Input Tokens per Question: 2874.03
    Avg Output Tokens per Question: 1613.96
Tool: MemMachineSearch
    Recall: 443/496 = 89.31%
    Precision: 443/2007 = 22.07%
    Avg Episodes Retrieved per Question: 9.99
    Avg Input Tokens per Question: 0.00
    Avg Output Tokens per Question: 0.00
```

In current retrieval traces, the direct path is reported as
`selected_agent_name=MemMachineSearch`, while the multi-hop COQ branch is
reported as `selected_agent_name=ChainOfQueryAgent`.

## Legacy Episodic Benchmark

For the legacy episodic-memory benchmark flow, see:

- `evaluation/episodic_memory/README.md`

## Dataset Paths

By default, benchmark scripts expect files under `evaluation/data/`, for
example:

- `evaluation/data/locomo10.json`
- `evaluation/data/wikimultihop.json`

## Wikimultihop Benchmark Note

In the WikiMultiHop dataset, each question has relatively short context
(about 25 context entries per question). To simulate a more realistic retrieval
scenario, the benchmark ingests all contexts into a single session and fully
randomizes their order.

Note that the WikiMultiHop dataset itself has some phrasing/chunking issues.
In some cases, one meaningful sentence is split across two entries, which can
cause key information to be missing. We may correct and update the dataset in
the future.

For pure LLM mode, all contexts are fed directly to the LLM as input.

## References

```bibtex
@misc{luo2025agentlightningtrainai,
  title={Agent Lightning: Train ANY AI Agents with Reinforcement Learning},
  author={Xufang Luo and Yuge Zhang and Zhiyuan He and Zilong Wang and Siyun Zhao and Dongsheng Li and Luna K. Qiu and Yuqing Yang},
  year={2025},
  eprint={2508.03680},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2508.03680},
}
```

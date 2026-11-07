# Benchmark Evaluation Guide

This folder contains evaluation scripts for measuring MemMachine retrieval and
memory quality on benchmark datasets.

## Benchmark Suites

- `retrieval_agent` (recommended): Current evaluation pipeline for retrieval
  behavior and answer quality. Uses MemMachine Python SDK.
- `episodic_memory` (legacy): Earlier LoCoMo dataset episodic memory benchmark workflow. Uses
  both MemMachine REST API and Python SDK.

## Retrieval-Agent Modes

The retrieval-agent benchmarks support three test targets:

1. `memmachine`: MemMachine retrieval without retrieval-agent orchestration.
2. `retrieval_agent`: MemMachine retrieval with retrieval-agent orchestration.
3. `llm`: Pure LLM baseline without MemMachine retrieval
   (full session content provided by dataset context).

## Prerequisites

- MemMachine backend is installed and configured.
- Start MemMachine before running benchmarks. Run from `memmachine/` root dir:

```sh
./memmachine-compose.sh start
```

- If you use the legacy episodic workflow, copy your `cfg.yml` into
  `evaluation/episodic_memory/` and rename it to `locomo_config.yaml`.

## Run Retrieval-Agent Benchmarks (Recommended)

> **Configuration**: All retrieval-agent benchmarks require a
> `configuration.yml` file placed in `evaluation/retrieval_agent/`. This file
> controls the language model, embedder, reranker, and database for every run —
> enabling non-OpenAI and local models. See
> [evaluation/retrieval_agent/README.md](retrieval_agent/README.md) for full
> details and ready-to-use configuration samples.

Run from `evaluation/retrieval_agent/`:

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

Examples:

- LoCoMo ingest:

```sh
./run_test.sh locomo exp1 ingest retrieval_agent
```

- LoCoMo search + scoring:

```sh
./run_test.sh locomo exp1 search retrieval_agent
```

- WikiMultiHop search (500 examples):

```sh
./run_test.sh wikimultihop exp1 search retrieval_agent 500
```

- HotpotQA validation set search (200 examples):

```sh
./run_test.sh hotpotqa exp1 search validation retrieval_agent 200
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
Tool: SplitQueryAgent
  Accuracy: 111/118 = 94.07%
Tool: MemMachineAgent
  Accuracy: 188/201 = 93.53%
Tool: ChainOfQueryAgent
  Accuracy: 167/181 = 92.27%
--------------------------------
HotpotQA Info Matrix:
hotpotqa Recall: 1116/1209 = 92.31%
hotpotqa Precision: 1116/4997 = 22.33%
hotpotqa Average Episodes Retrieved per Question: 9.99
Tool: SplitQueryAgent
    Recall: 246/265 = 92.83%
    Precision: 246/1180 = 20.85%
    Avg Episodes Retrieved per Question: 10.00
    Avg Input Tokens per Question: 1228.59
    Avg Output Tokens per Question: 434.92
Tool: ChainOfQueryAgent
    Recall: 427/448 = 95.31%
    Precision: 427/1810 = 23.59%
    Avg Episodes Retrieved per Question: 10.00
    Avg Input Tokens per Question: 2874.03
    Avg Output Tokens per Question: 1613.96
Tool: MemMachineAgent
    Recall: 443/496 = 89.31%
    Precision: 443/2007 = 22.07%
    Avg Episodes Retrieved per Question: 9.99
    Avg Input Tokens per Question: 0.00
    Avg Output Tokens per Question: 0.00
ToolSelectAgent Avg Input Tokens per Question: 1049.25
ToolSelectAgent Avg Output Tokens per Question: 195.44
```

## Legacy Episodic Benchmark

For the legacy episodic-memory benchmark flow, see:

- `evaluation/episodic_memory/README.md`

## Datasets

`run_test.sh` reads the LoCoMo and WikiMultiHop datasets from
`evaluation/data/`. Download them from their authors before running these
benchmarks. Run the commands below from the repository root.

### LoCoMo

`locomo10.json` from the
[LoCoMo repository](https://github.com/snap-research/locomo):

```sh
mkdir -p evaluation/data
curl -fL -o evaluation/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc/data/locomo10.json
```

Optionally, check the download:

```sh
echo "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4  evaluation/data/locomo10.json" | shasum -a 256 -c
```

### WikiMultiHop

The dev split of the original `data.zip` release from the
[2WikiMultihopQA repository](https://github.com/Alab-NII/2wikimultihop):

```sh
mkdir -p evaluation/data
curl -fL -o 2wikimultihop.zip \
  'https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1'
```

Optionally, check the download:

```sh
echo "e8e57c0aafc4a26d41131e320ebb5afb6f2aca86b8a6e6611b08f52033cb7d04  2wikimultihop.zip" | shasum -a 256 -c
```

The benchmark scripts read one JSON object per line, with `context`,
`supporting_facts`, and `evidences` encoded as JSON strings, so convert the
downloaded `dev.json`:

```sh
unzip -p 2wikimultihop.zip data/dev.json | python3 -c '
import json, sys
for row in json.load(sys.stdin.buffer):
    for key in ("context", "supporting_facts", "evidences"):
        row[key] = json.dumps(row[key])
    print(json.dumps(row))
' > evaluation/data/wikimultihop.json
rm 2wikimultihop.zip
```

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

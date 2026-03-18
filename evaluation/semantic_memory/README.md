## Semantic Memory LoCoMo Evaluation

### Prereqs
- Docker (for testcontainers Postgres), or a local Postgres+pgvector.
- OpenAI API key for the answer model and LLM judge.
- Copy and edit `semantic_config.yaml` with real credentials.

### Run
```sh
python runner.py \
  --data-path ../data/locomo10.json \
  --config semantic_config.yaml \
  --run-id my_run
```

### Outputs
`runs/<run_id>/<variant>/` contains `search_results.json`,
`evaluation_metrics.json`, and score summaries.

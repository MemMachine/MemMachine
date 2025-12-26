## Dependencies for evaluation scripts
```
openai-agents
pandas
```

## Environment variables
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY

NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD

OPENAI_API_KEY
```

## Add conversations to memory
```sh
python locomo_ingest.py --data-path path/to/locomo10.json
```

## Search memory and answer questions
```sh
python locomo_search.py --data-path path/to/locomo10.json --target-path results.json
```

## Search memory and answer questions using a simple agent
```sh
python locomo_search_agent.py --data-path path/to/locomo10.json --target-path results.json
```

## Evaluate responses
```sh
python locomo_evaluate.py --data-path results.json --target-path evaluation_metrics.json
```

## Generate scores
```sh
python generate_scores.py --data-path evaluation_metrics.json
```

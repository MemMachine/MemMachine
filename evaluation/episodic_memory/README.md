## Add conversations to memory
```sh
python locomo_ingest.py --data-path path/to/locomo10.json
```

## Search memory and answer questions
```sh
python locomo_search.py --data-path path/to/locomo10.json --target-path results.json
```

## Evaluate responses
```sh
python locomo_evaluate.py --data-path results.json --target-path evaluation_metrics.json
```

## Generate scores
```sh
python generate_scores.py
```

## Delete data
```sh
python locomo_delete.py --data-path path/to/locomo10.json
```

## Add conversations to memory
```sh
python locomo_ingest.py --data-path path/to/locomo10.json
```

## Search memory and answer questions
```sh
python run_experiments.py --method search --dataset path/to/locomo10.json
```

## Evaluate responses
```sh
commit_id=$(git rev-parse --short=7 HEAD)
python evals.py --input_file results_IM_$commit_id.json --output_file evaluation.json
```

## Generate scores
```sh
python generate_scores.py --input_path evaluation.json
```

## Delete data
```sh
python locomo_delete.py --data-path path/to/locomo10.json
```

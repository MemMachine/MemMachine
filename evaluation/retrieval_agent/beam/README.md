# BEAM Benchmark

BEAM evaluates long-form conversation memory through rubric-based assessment.

## Dependencies

BEAM evaluation requires additional Python packages:

```bash
pip install scipy datasets
```

- `scipy`: Used for Kendall tau-b correlation in event ordering evaluation
- `datasets`: Used for downloading BEAM dataset from HuggingFace

## Dataset Download

Download and convert the BEAM dataset from HuggingFace:

```bash
cd evaluation/retrieval_agent/beam
python beam_download.py --size 100K --output ./beam_data
python beam_download.py --size 500K --output ./beam_data
python beam_download.py --size 1M --output ./beam_data

# Download multiple sizes at once
python beam_download.py --size 100K 500K 1M --output ./beam_data

# Download 10M dataset 
python beam_download.py --size 10M --output ./beam_data
```

## Usage

Run BEAM benchmark from the `evaluation/retrieval_agent/` directory:

```bash
cd evaluation/retrieval_agent

# Ingest chat data
./run_test.sh beam exp1 ingest retrieval_agent /path/to/chat.json /path/to/probing_questions.json

# Search and evaluate
./run_test.sh beam exp1 search retrieval_agent /path/to/chat.json /path/to/probing_questions.json

# With custom concurrency
./run_test.sh beam exp1 search retrieval_agent /path/to/chat.json /path/to/probing_questions.json --search-concurrency 10 --judge-concurrency 30
```

For more details, see the official BEAM repository: https://github.com/mohammadtavakoli78/BEAM

## Note on Scoring

The official BEAM evaluation code casts rubric scores to `int()` before summing, which causes 0.5 scores to be silently dropped (converted to 0). This implementation preserves float scores (0.0, 0.5, 1.0) to properly reflect partial compliance. As a result, scores may differ by a few percentage points compared to running the official BEAM code directly.

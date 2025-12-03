# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/generate_scores.py).
# It has been modified to print category names and only report LLM judge scores.

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def flatten_items(data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Flatten a nested dict of questions into a dataframe."""
    all_items: list[dict[str, Any]] = []
    for values in data.values():
        all_items.extend(values)
    df = pd.DataFrame(all_items)
    df["category"] = pd.to_numeric(df["category"])
    return df


def log_results(df: pd.DataFrame) -> None:
    """Log aggregated metrics by category and overall means."""
    categories = ["multi_hop", "temporal", "open_domain", "single_hop"]
    result = df.groupby("category").agg({"llm_score": "mean"}).round(4)
    result["count"] = df.groupby("category").size()
    result["type"] = result.index.map(lambda x: categories[int(x) - 1])

    logger.info("Mean Scores Per Category:\n%s", result)
    overall_means = df.agg({"llm_score": "mean"}).round(4)
    logger.info("Overall Mean Scores:\n%s", overall_means)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()

    with args.input_path.open("r") as input_file:
        data: dict[str, list[dict[str, Any]]] = json.load(input_file)

    df = flatten_items(data)
    log_results(df)


if __name__ == "__main__":
    main()

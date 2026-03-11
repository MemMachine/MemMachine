import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="evaluation.json")
args = parser.parse_args()

categories = [
    "overall",
    "multi-session",
    "temporal-reasoning",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]

# Load the evaluation metrics data
with open(args.data_path, "r") as f:
    data = json.load(f)

for category in categories:
    llm_score = data[category]["llm_score"]
    print(f"{category}: {llm_score:.4f}")

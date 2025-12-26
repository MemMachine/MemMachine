# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/generate_scores.py).
# It has been modified to print category names and only report LLM judge scores.

import argparse
import json

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="evaluation.json")
args = parser.parse_args()

categories = ["multi_hop", "temporal", "open_domain", "single_hop"]

# Load the evaluation metrics data
with open(args.data_path, "r") as f:
    data = json.load(f)

# Flatten the data into a list of question items
all_items = []
for key in data:
    all_items.extend(data[key])

# Convert to DataFrame
df = pd.DataFrame(all_items)

# Convert category to numeric type
df["category"] = pd.to_numeric(df["category"])

# Calculate mean scores by category
result = df.groupby("category").agg({"llm_score": "mean"}).round(4)

# Add count of questions per category
result["count"] = df.groupby("category").size()

result["type"] = result.index.map(lambda x: categories[x - 1])

# Print the results
print("Mean Scores Per Category:")
print(result)

# Calculate overall means
overall_means = df.agg({"llm_score": "mean"}).round(4)

print("\nOverall Mean Scores:")
print(overall_means)

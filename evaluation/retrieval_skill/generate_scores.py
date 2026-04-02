# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/generate_scores.py).
# It has been modified to print category names and only report LLM judge scores.

import argparse
import json

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="evaluation.json")
args = parser.parse_args()

categories = [
    "multi_hop",
    "temporal",
    "open_domain",
    "single_hop",
    "LOCOMO-IGNORED",  # Locomo category 5 is ignored
    "compositional",
    "comparison",
    "inference",
    "bridge_comparison",
]

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
# df["category"] = pd.to_numeric(df["category"])

# Calculate mean scores by category
result = df.groupby("category").agg({"llm_score": "mean"}).round(4)

# Add count of questions per category
result["count"] = df.groupby("category").size()

if isinstance(result.index, int):
    result["type"] = result.index.map(lambda x: categories[x - 1])

# Print the results
print("Mean Scores Per Category:")
print(result)

if "level" in df.columns:
    level_means = df.groupby("level").agg({"llm_score": "mean"}).round(4)
    level_means["count"] = df.groupby("level").size()
    print("\nMean Scores Per Level:")
    print(level_means)

# Calculate overall means
overall_means = df.agg({"llm_score": "mean"}).round(4)

print("\nOverall Mean Scores:")
print(overall_means)

wiki_matrix = None
locomo_matrix = None
hotpotqa_matrix = None
longmemeval_matrix = None
general_matrix = None
skills_called: dict[str, int] = {}
skills_correct: dict[str, int] = {}
for item in all_items:
    if wiki_matrix is None:
        wiki_matrix = item.get("wiki_final_matrix", None)
    if locomo_matrix is None:
        locomo_matrix = item.get("locomo_final_matrix", None)
    if hotpotqa_matrix is None:
        hotpotqa_matrix = item.get("hotpotqa_final_matrix", None)
    if longmemeval_matrix is None:
        longmemeval_matrix = item.get("longmemeval_final_matrix", None)
    if general_matrix is None:
        general_matrix = item.get("general_final_matrix", None)

    skill = (
        item.get("selected_skill_name")
        or item.get("selected_skill")
        or item.get("skill")
        or "Unknown"
    )
    skills_called[skill] = skills_called.get(skill, 0) + 1
    if item.get("llm_score", 1):
        skills_correct[skill] = skills_correct.get(skill, 0) + 1

skills_overall = "Skills Overall Accuracy:\n"
for skill, called in skills_called.items():
    correct = skills_correct.get(skill, 0)
    accuracy = correct / called * 100 if called > 0 else 0.0
    skills_overall += (
        f"Skill: {skill}\n  Accuracy: {correct}/{called} = {accuracy:.2f}%\n"
    )

print("\n--------------------------------")
print(skills_overall)
print("--------------------------------")

if wiki_matrix:
    print(f"\nWiki Info Matrix:\n{wiki_matrix}")
if locomo_matrix:
    print(f"\nLocomo Info Matrix:\n{locomo_matrix}")
if hotpotqa_matrix:
    print(f"\nHotpotQA Info Matrix:\n{hotpotqa_matrix}")
if longmemeval_matrix:
    print(f"\nLongMemEval Info Matrix:\n{longmemeval_matrix}")
if general_matrix:
    print(f"\nGeneral Info Matrix:\n{general_matrix}")

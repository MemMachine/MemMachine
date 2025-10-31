# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/generate_scores.py).
# It has been modified to only report LLM judge scores.

import json, sys

import pandas as pd

path = sys.argv[1]

# Load the evaluation metrics data
with open(path, "r") as f:
    data = json.load(f)

# Flatten the data into a list of question items
all_items = []
for key in data:
    all_items.extend(data[key])

final_matrix = ""
num_use_kg = 0
num_use_em = 0
num_positive_use_em = 0
num_positive_use_kg = 0
num_correct = 0
num_incorrect = 0
num_total = 0
for item in all_items:
    if "final_matrix" in item:
        final_matrix = item["final_matrix"]
    
    if item.get("used_kg", True):
        num_use_kg += 1
        if item["llm_score"] == 1:
            num_positive_use_kg += 1
    elif item.get("used_em", False):
        num_use_em += 1
        if item["llm_score"] == 1:
            num_positive_use_em += 1
    
    if item["llm_score"] == 1:
        num_correct += 1
    else:
        num_incorrect += 1
    
    num_total += 1

if num_use_kg != 0:
    final_matrix += f"Positive cases using KG(EM search not sufficient): {num_positive_use_kg}/{num_use_kg} = {num_positive_use_kg/num_use_kg*100:.2f}%\n"
else:
    final_matrix += "Using 0 KG searches.\n"

if num_use_em != 0:
    final_matrix += f"Positive cases using EM only: {num_positive_use_em}/{num_use_em} = {num_positive_use_em/num_use_em*100:.2f}%\n"
else:
    final_matrix += "Using 0 EM searches.\n"

# Convert to DataFrame
df = pd.DataFrame(all_items)

# Convert category to numeric type
# df["category"] = pd.to_numeric(df["category"])

# Calculate mean scores by category
result = df.groupby("category").agg({"llm_score": "mean"}).round(4)

# Add count of questions per category
result["count"] = df.groupby("category").size()

# Print the results
print("Mean Scores Per Category:")
print(result)

# Calculate overall means
overall_means = df.agg({"llm_score": "mean"}).round(4)

print("\nOverall Mean Scores:")
print(overall_means)

# print(f"\nNumber of positive cases using long-term memory: {num_positive_use_em}")
# print(f"Number of negative cases using long-term memory: {num_negative_use_em}")
# print(f"Total correct answers: {num_correct}/{num_total}")
print(f"\nFinal Info Matrix:\n{final_matrix}")
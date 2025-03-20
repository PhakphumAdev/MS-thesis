import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the directory containing the sample files
directory = "typhoon2-t1-3b/scb10x__llama3.2-typhoon2-t1-3b-research-preview/"  # Update this path as needed

# Extract model name from directory name
model_name = os.path.basename(directory.rstrip("/"))

# Define the five tasks
tasks = ["physical", "psychological", "social", "spatial", "temporal"]

# Initialize dictionary to store correctness data
task_data = {}

# Process each task
for task in tasks:
    task_files = [f for f in os.listdir(directory) if f.startswith(f"samples_{task}_") and f.endswith(".jsonl")]

    # Find English and Thai versions of the task
    en_file = next((f for f in task_files if "_en" in f), None)
    th_file = next((f for f in task_files if "_th" in f), None)

    # Function to extract correctness data
    def extract_correctness(file_path):
        accuracies = []
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                accuracies.append(data.get("acc", 0))  # Default to 0 if missing
        return np.array(accuracies)

    # Load data for English and Thai
    if en_file and th_file:
        en_correct = extract_correctness(os.path.join(directory, en_file))
        th_correct = extract_correctness(os.path.join(directory, th_file))

        # Ensure all tasks have the same number of questions
        max_len = max(len(en_correct), len(th_correct))
        en_correct = np.pad(en_correct, (0, max_len - len(en_correct)), constant_values=0)
        th_correct = np.pad(th_correct, (0, max_len - len(th_correct)), constant_values=0)

        # Store in dictionary (both languages as separate rows)
        task_data[f"{task}-en"] = en_correct
        task_data[f"{task}-th"] = th_correct

# Convert to a NumPy matrix
heatmap_matrix = np.array(list(task_data.values()))

# Create the multi-task heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_matrix, cmap="RdYlGn", cbar=True, xticklabels=False, yticklabels=list(task_data.keys()))

# Add title and labels
plt.title(f"Model's Performance Across Tasks (EN & TH)\n({model_name})", fontsize=14)
plt.xlabel("Question ID", fontsize=12)
plt.ylabel("Task & Language", fontsize=12)

# Save the heatmap image
heatmap_filename = f"{model_name}-multitask-heatmap.png"
plt.savefig(heatmap_filename, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {heatmap_filename}")
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

# List of JSON files
file_paths = [
    "llama-3.2-3b-it/meta-llama__Llama-3.2-3B-Instruct/results_2025-03-02T18-42-11.210385.json",
    "pakphum/llma3.2-instruct-merged-typhoonsame/pakphum__llma3.2-instruct-merged-typhoonsame/results_2025-03-20T21-59-38.212514.json",
    "typhoon2-3b-it/scb10x__llama3.2-typhoon2-3b-instruct/results_2025-02-28T02-34-35.834452.json"
]

# Categories to plot
categories = ["physical", "psychological", "social", "spatial", "temporal"]
num_vars = len(categories)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Close the radar loop

# Setup subplots: one for English, one for Thai
fig, (ax_en, ax_th) = plt.subplots(1, 2, figsize=(22, 12), subplot_kw=dict(polar=True))

# Process each file
for file_path in file_paths:
    with open(file_path, "r") as file:
        data = json.load(file)

    # Use model name or sanitized name as label
    label = data.get("model_name", os.path.basename(file_path))

    # Extract accuracy data
    accuracies_en = [data["results"][f"{cat}_en"]["acc,none"] for cat in categories]
    accuracies_th = [data["results"][f"{cat}_th"]["acc,none"] for cat in categories]

    # Complete the loop
    accuracies_en += accuracies_en[:1]
    accuracies_th += accuracies_th[:1]

    # Plot English
    ax_en.plot(angles, accuracies_en, label=label, linestyle='-', marker='o')
    ax_en.fill(angles, accuracies_en, alpha=0.1)

    # Plot Thai
    ax_th.plot(angles, accuracies_th, label=label, linestyle='--', marker='x')
    ax_th.fill(angles, accuracies_th, alpha=0.1)

# Format English radar chart
ax_en.set_xticks(angles[:-1])
ax_en.set_xticklabels(categories, fontsize=14)
ax_en.set_title("English Accuracy Comparison", fontsize=20, pad=20)
ax_en.set_ylim(0, 1)
ax_en.set_yticks(np.linspace(0, 1, 11))
ax_en.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=14)

# Format Thai radar chart
ax_th.set_xticks(angles[:-1])
ax_th.set_xticklabels(categories, fontsize=14)
ax_th.set_title("Thai Accuracy Comparison", fontsize=20, pad=20)
ax_th.set_ylim(0, 1)
ax_th.set_yticks(np.linspace(0, 1, 11))
ax_th.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=14)

# Adjust layout
plt.subplots_adjust(wspace=0.5, bottom=0.2)

# Save the plot
output_path = "graph/multi_model_radar_chart_separated.png"
os.makedirs("graph", exist_ok=True)
plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Radar chart saved to {output_path}")
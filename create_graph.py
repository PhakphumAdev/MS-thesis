import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Load the JSON file
file_path = "typhoon2-t1-3b/scb10x__llama3.2-typhoon2-t1-3b-research-preview/results_2025-02-27T13-58-23.710122.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract model name from JSON
title = data.get("model_name", "Model Comparison")
model_name_sanitized = data.get("model_name_sanitized", "model_chart")

# Extract accuracy results for English and Thai
categories = ["physical", "psychological", "social", "spatial", "temporal"]
accuracies_en = [data["results"][f"{cat}_en"]["acc,none"] for cat in categories]
accuracies_th = [data["results"][f"{cat}_th"]["acc,none"] for cat in categories]

# Number of variables
num_vars = len(categories)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Complete the loop

# Append first value to close the radar chart
accuracies_en += accuracies_en[:1]
accuracies_th += accuracies_th[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot the data
ax.plot(angles, accuracies_en, label='English', linestyle='solid', marker='o')
ax.fill(angles, accuracies_en, alpha=0.2)
ax.plot(angles, accuracies_th, label='Thai', linestyle='solid', marker='o')
ax.fill(angles, accuracies_th, alpha=0.2)

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Set title and legend
ax.set_title(title, fontsize=14, pad=20)
ax.legend()

# Set scale to 0-1
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 11))

# Save the plot with model_name_sanitized
output_path = f"graph/{model_name_sanitized}_radar_chart.png"
plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print(f"Radar chart saved to {output_path}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load dataset
data = pd.read_csv("survey lung cancer.csv")

# 2. Define attributes to plot and target class variable
attributes = ["GENDER","SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "LUNG_CANCER"]
target_col = "LUNG_CANCER"

# Prepare target categories and matching colors
cancer_statuses = data[target_col].unique()
colors = ["red", "green"]  # Soft Red for YES, Soft Blue for NO

# Create 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=100)
axes = axes.flatten()

# 3. Loop through attributes to build stacked bar plots
for i, attr in enumerate(attributes):
    ax = axes[i]

    # Aggregate counts per attribute value and target outcome
    counts = (
        data.groupby([attr, target_col], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    # Plot stacked bars
    bottom = np.zeros(len(counts.index))
    
    # Map numerical survey codes (1/2) to clear readable text labels
    if attr == "GENDER":
        x_labels = [str(x) for x in counts.index]
    else:
        # standard survey encoding: 1 -> NO / Low, 2 -> YES / High
        label_map = {1: "NO (1)", 2: "YES (2)"}
        x_labels = [label_map.get(x, str(x)) for x in counts.index]

    for status, color in zip(counts.columns, colors):
        values = counts[status].values
        ax.bar(
            x_labels,
            values,
            bottom=bottom,
            label=f"Lung Cancer: {status}",
            color=color,
            edgecolor="white",
            width=0.4,
        )
        bottom += values

    # Subplot styling
    ax.set_title(f"Distribution of {attr.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{attr.replace('_', ' ').title()} Categories", fontsize=9)
    ax.set_ylabel("Patient Count", fontsize=9)
    ax.legend(title="Lung Cancer", fontsize=8, title_fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load dataset and clean column headers
data = pd.read_csv("crx.csv")
data.columns = data.columns.str.strip()

# Target column (defaulting to the last column)
target_col = data.columns[-1]
attributes = ["A2", "A5", "A6", "A11"]

# Target colors
unique_targets = data[target_col].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_targets)))

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
axes = axes.flatten()

# Loop through attributes
for i, attr in enumerate(attributes):
    ax = axes[i]

    if attr in ["A2", "A11"]:
        # Bin continuous features into 4 equal intervals
        plot_series = pd.cut(data[attr], bins=4).astype(str)
    else:
        # Keep discrete/categorical features as strings
        plot_series = data[attr].astype(str)

    # Aggregate counts
    counts = pd.crosstab(plot_series, data[target_col])

    # Plot side-by-side grouped bars
    x_indices = np.arange(len(counts.index))
    num_classes = len(counts.columns)
    bar_width = 0.8 / num_classes  # Dynamically calculate width based on target classes

    for idx, (status, color) in enumerate(zip(counts.columns, colors)):
        values = counts[status].values
        # Shift x position for each class to place bars side by side
        bar_positions = x_indices + (idx * bar_width) - (bar_width * (num_classes - 1) / 2)
        
        ax.bar(
            bar_positions,
            values,
            width=bar_width,
            label=f"{status}",
            color=color,
            edgecolor="white",
        )

    # Set x-ticks at center of grouped bars
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(x) for x in counts.index])

    # Styling
    ax.set_title(f"Distribution of Attribute {attr}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{attr} Categories", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Rotate x-axis tick labels for A6 or long category lists
    if attr == "A6" or len(counts.index) > 6:
        ax.tick_params(axis='x', rotation=45)

# Place unified legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title=f"Target ({target_col})", loc="upper right", bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.show()
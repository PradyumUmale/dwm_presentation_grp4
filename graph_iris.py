import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load dataset
data = pd.read_csv("iris.csv")

# 2. Define the four attributes to plot
attributes = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
axes = axes.flatten()  # Flatten to iterate easily (0 to 3)

species_list = ["setosa", "versicolor", "virginica"]
colors = ["red", "green", "blue"]

# 3. Loop through each attribute and create a stacked bar plot
for i, attr in enumerate(attributes):
    ax = axes[i]
    
    # Calculate class means and split points for current attribute
    class_1 = data[data["species"] == "setosa"][attr]
    class_2 = data[data["species"] == "versicolor"][attr]
    class_3 = data[data["species"] == "virginica"][attr]

    sp1 = (class_1.mean() + class_2.mean()) / 2
    sp2 = (class_2.mean() + class_3.mean()) / 2
    split_points = sorted([sp1, sp2])

    # Define bins and labels
    bins = [-np.inf, split_points[0], split_points[1], np.inf]
    labels = [
        f"Low (<{split_points[0]:.2f})",
        f"Mid ({split_points[0]:.2f}-{split_points[1]:.2f})",
        f"High (>{split_points[1]:.2f})"
    ]

    # Create temporary grouping
    group_col = f"{attr}_group"
    data[group_col] = pd.cut(data[attr], bins=bins, labels=labels)

    # Aggregate counts
    counts = (
        data.groupby([group_col, "species"], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    # Plot stacked bar chart on specific subplot axis
    bottom = np.zeros(len(counts.index))
    for species, color in zip(species_list, colors):
        if species in counts.columns:
            values = counts[species].values
            ax.bar(
                counts.index,
                values,
                bottom=bottom,
                label=species,
                color=color,
                edgecolor="white",
                width=0.45,
            )
            bottom += values

    # Subplot styling
    ax.set_title(f"Binned by {attr.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Interval Groups", fontsize=9)
    ax.set_ylabel("Flower Count", fontsize=9)
    ax.legend(title="Species", fontsize=8, title_fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

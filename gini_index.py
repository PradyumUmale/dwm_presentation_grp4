import numpy as np
import pandas as pd

data1 = pd.read_csv("iris.csv")
data2 = pd.read_csv("survey lung cancer.csv")
data3 = pd.read_csv("crx.csv")

attr1 = ["sepal_width", "sepal_length", "petal_width", "petal_length"]
attr2 = ["GENDER", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "LUNG_CANCER"]
attr3 = ["A2", "A5", "A6", "A11"]

target1 = "species"
target2 = "LUNG_CANCER"
target3 = "class"

# 3. Define Custom Splits / Categorical Subsets
# Continuous attributes use split points (floats)
custom_splits = {}

# Continuous Iris Splits
for attr in attr1:
    c1 = data1[data1["species"] == "setosa"][attr]
    c2 = data1[data1["species"] == "versicolor"][attr]
    c3 = data1[data1["species"] == "virginica"][attr]
    custom_splits[attr] = sorted(
        [(c1.mean() + c2.mean()) / 2, (c2.mean() + c3.mean()) / 2]
    )

# Continuous CRX Splits
for attr in ["A2", "A11"]:
    data3[attr] = pd.to_numeric(data3[attr], errors="coerce")
    c1 = data3[data3["class"] == "+"][attr]
    c2 = data3[data3["class"] == "-"][attr]
    custom_splits[attr] = [(c1.mean() + c2.mean()) / 2]

# Categorical custom binary grouping example for CRX A6 attribute
# Format: ([Group 1 categories], [Group 2 categories])
custom_splits["A5"] = (["g", "p"], ["gg"])
custom_splits["A6"] = (["aa", "c", "d", "ff", "i", "j", "k", "m"],
                       ["cc", "e", "q", "r", "w", "x"])


# 4. Helper Functions
def calc_gini(target_series):
    """Calculates Gini Index Gini(D) of a target series."""
    counts = target_series.value_counts()
    probs = counts / len(target_series)
    return 1.0 - np.sum(probs**2)


def discretize_attribute(series, split_rule=None):
    """
    Handles discretization and categorical subset splitting:
    - If split_rule is a list of numeric thresholds (continuous): Bins series into ranges.
    - If split_rule is a tuple of two lists/sets (categorical): Groups categories into 2 subsets.
    - Else: Keeps original categorical/discrete values.
    """
    if split_rule is None:
        return series

    # Case 1: Continuous numeric attribute with split points
    if isinstance(split_rule, list):
        bins = [-np.inf] + sorted(split_rule) + [np.inf]
        return pd.cut(series, bins=bins)

    # Case 2: Categorical attribute with two manually specified class groups
    elif isinstance(split_rule, tuple) and len(split_rule) == 2:
        group1, group2 = split_rule
        # Map values to Group 1, Group 2, or Other
        def map_category(val):
            if val in group1:
                return "Group_1"
            elif val in group2:
                return "Group_2"
            return "Other"

        return series.apply(map_category)

    return series


def calc_gini_A(df, attribute, target, split_rule=None):
    """Calculates Gini_A(D) for an attribute using custom splits or original categories."""
    gini_A = 0.0
    total_len = len(df)

    # Transform attribute series based on custom rule or default values
    series = discretize_attribute(df[attribute], split_rule)

    for val in series.unique():
        if pd.isna(val):
            continue
        subset = df[series == val]
        weight = len(subset) / total_len

        if weight > 0:
            gini_subset = calc_gini(subset[target])
            gini_A += weight * gini_subset

    return gini_A


def process_dataset_gini(name, df, attributes, target):
    """Prints aligned table of Gini(D), Gini_A(D), and Delta Gini (Impurity Reduction)."""
    gini_D = calc_gini(df[target])

    gini_A_dict = {}
    delta_gini_dict = {}

    for attr in attributes:
        split_rule = custom_splits.get(attr, None)
        gini_A = calc_gini_A(df, attr, target, split_rule=split_rule)
        delta_gini = gini_D - gini_A

        gini_A_dict[attr] = round(gini_A, 4)
        delta_gini_dict[attr] = round(delta_gini, 4)

    # DataFrame formatting ensures clean table alignment
    results_df = pd.DataFrame(
        [gini_A_dict, delta_gini_dict],
        index=["Gini_A(D)", "Reduction"],
    )
    print(f"{name} : Gini(D) = {gini_D:.4f}")
    if name == "\nCredit Approval (CRX)":
        print(f"Atribute division:\nA5: {custom_splits['A5']}\nA6: {custom_splits['A6']}")
    print(results_df.to_string())


# 5. Run Gini Calculations
process_dataset_gini("Iris", data1, attr1, target1)
process_dataset_gini("\nLung Cancer", data2, attr2, target2)
process_dataset_gini("\nCredit Approval (CRX)", data3, attr3, target3)
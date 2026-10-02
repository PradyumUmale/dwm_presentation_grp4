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

custom_splits_iris = {}
custom_splits_crx = {}

for attr in attr1:
    class_1 = data1[data1["species"] == "setosa"][attr]
    class_2 = data1[data1["species"] == "versicolor"][attr]
    class_3 = data1[data1["species"] == "virginica"][attr]

    sp1 = (class_1.mean() + class_2.mean()) / 2
    sp2 = (class_2.mean() + class_3.mean()) / 2
    custom_splits_iris[attr] = sorted([sp1, sp2])  

for attr in ["A2", "A11"]:
    class_1 = data3[data3["class"] == "+"][attr]
    class_2 = data3[data3["class"] == "-"][attr]

    sp1 = (class_1.mean() + class_2.mean()) / 2
    custom_splits_crx[attr] = [sp1]  

def calc_entropy(target_series):
    counts = target_series.value_counts()
    probs = counts / len(target_series)
    return -np.sum(probs * np.log2(probs))


def discretize_attribute(series, split_points=None):
    """Discretizes continuous numeric series based on custom split points or median split."""
    if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 10:
        return series  # Categorical or discrete integer series

    if split_points is not None:
        # Use user-defined custom split points
        bins = [-np.inf] + sorted(split_points) + [np.inf]
        return pd.cut(series, bins=bins)
    else:
        # Default fallback: Split at the median if no split points provided
        median_val = series.median()
        bins = [-np.inf, median_val, np.inf]
        return pd.cut(series, bins=bins)


def calc_info(df, attribute, target, split_points=None):
    """Calculates Info_A(D) and SplitInfo_A(D) using custom or default split boundaries."""
    info_A = 0.0
    total_len = len(df)

    # Discretize series if continuous
    series = discretize_attribute(df[attribute], split_points)

    for val in series.unique():
        if pd.isna(val):
            continue
        subset = df[series == val]
        weight = len(subset) / total_len

        if weight > 0:
            # Info_A(D) calculation
            entropy_subset = calc_entropy(subset[target])
            info_A += weight * entropy_subset

    return info_A


def process_dataset(name, df, attributes, target, split_dict=None):
    """Prints Info(D), Info_A(D), Gain(A), SplitInfo(A), and Gain Ratio(A)."""
    if split_dict is None:
        split_dict = {}

    info_D = calc_entropy(df[target])

    info_A_dict = {}
    gain_dict = {}

    for attr in attributes:
        # Get custom split points if defined for this attribute
        splits = split_dict.get(attr, None)

        info_A = calc_info(df, attr, target, split_points=splits)
        gain = info_D - info_A

        info_A_dict[attr] = round(info_A, 4)
        gain_dict[attr] = round(gain, 4)

    # Build DataFrame for visually aligned output
    results_df = pd.DataFrame(
        [info_A_dict, gain_dict],
        index=["Info(A,D)", "Info Gain(A)"],
    )

    print(f"{name} : Info(D) = {info_D:.4f}")
    print(results_df.to_string())


# Run calculations with custom split points
process_dataset("Iris", data1, attr1, target1, split_dict=custom_splits_iris)
process_dataset("\nLung Cancer", data2, attr2, target2)
process_dataset("\nCredit Approval", data3, attr3, target3, split_dict=custom_splits_crx)
import numpy as np
import pandas as pd

# 1. Load Datasets
data1 = pd.read_csv("iris.csv")
data2 = pd.read_csv("survey lung cancer.csv")
data3 = pd.read_csv("crx.csv")

# 2. Define Attributes and Target Labels
attr1 = ["sepal_width", "sepal_length", "petal_width", "petal_length"]
attr2 = ["GENDER", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "LUNG_CANCER"]
attr3 = ["A2", "A5", "A6", "A11"]

target1 = "species"
target2 = "LUNG_CANCER"
target3 = "class"

# 3. Compute Custom Split Points for Continuous Attributes
custom_splits = {}

# Iris Splits (3 classes -> 2 split points)
for attr in attr1:
    c1 = data1[data1["species"] == "setosa"][attr]
    c2 = data1[data1["species"] == "versicolor"][attr]
    c3 = data1[data1["species"] == "virginica"][attr]
    custom_splits[attr] = sorted(
        [(c1.mean() + c2.mean()) / 2, (c2.mean() + c3.mean()) / 2]
    )

# CRX Numeric Splits (2 classes -> 1 split point)
for attr in ["A2", "A11"]:
    c1 = data3[data3["class"] == "+"][attr]
    c2 = data3[data3["class"] == "-"][attr]
    custom_splits[attr] = [(c1.mean() + c2.mean()) / 2]


# 4. Helper Functions
def calc_entropy(target_series):
    """Calculates Entropy Info(D) of a target series."""
    counts = target_series.value_counts()
    probs = counts / len(target_series)
    return -np.sum(probs * np.log2(probs))


def calc_info_and_splitinfo(df, attribute, target):
    """Calculates Info_A(D) and SplitInfo_A(D) for an attribute."""
    info_A = 0.0
    split_info_A = 0.0
    total_len = len(df)

    series = df[attribute]

    # Apply custom split boundaries if defined
    if attribute in custom_splits:
        bins = [-np.inf] + custom_splits[attribute] + [np.inf]
        series = pd.cut(series, bins=bins)

    for val in series.unique():
        if pd.isna(val):
            continue
        subset = df[series == val]
        weight = len(subset) / total_len

        if weight > 0:
            entropy_subset = calc_entropy(subset[target])
            info_A += weight * entropy_subset
            split_info_A -= weight * np.log2(weight)

    return info_A, split_info_A


def process_dataset(name, df, attributes, target):
    """Prints aligned table of Info(D), Info(A,D), Gain(A), SplitInfo(A), and Gain Ratio(A)."""
    info_D = calc_entropy(df[target])

    info_A_dict, gain_dict, split_info_dict, gain_ratio_dict = {}, {}, {}, {}

    for attr in attributes:
        info_A, split_info = calc_info_and_splitinfo(df, attr, target)
        gain = info_D - info_A
        gain_ratio = gain / split_info if split_info > 0 else 0.0

        info_A_dict[attr] = round(info_A, 4)
        gain_dict[attr] = round(gain, 4)
        split_info_dict[attr] = round(split_info, 4)
        gain_ratio_dict[attr] = round(gain_ratio, 4)

    # DataFrame ensures header alignment regardless of column name lengths
    results_df = pd.DataFrame(
        [info_A_dict, gain_dict, split_info_dict, gain_ratio_dict],
        index=["Info(A,D)", "Info Gain(A)", "SplitInfo(A)", "Gain Ratio(A)"],
    )

    print(f"{name} : Info(D) = {info_D:.4f}")
    print(results_df.to_string())


# 5. Run Calculations
process_dataset("Iris", data1, attr1, target1)
process_dataset("\nLung Cancer", data2, attr2, target2)
process_dataset("\nCredit Approval (CRX)", data3, attr3, target3)
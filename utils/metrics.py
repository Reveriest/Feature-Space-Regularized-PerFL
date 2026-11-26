import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def parse_cm(cm_value):
    if isinstance(cm_value, str):
        cm_value = ast.literal_eval(cm_value)
    return np.asarray(cm_value, dtype=float)

def cm_accuracy(cm):
    cm = parse_cm(cm)
    total = cm.sum()
    if total <= 0:
        return np.nan
    return float(np.trace(cm) / total)

def cm_macro_f1(cm):
    cm = parse_cm(cm)
    if cm.size == 0:
        return np.nan
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )
    return float(np.mean(f1)) if f1.size else np.nan

def cm_macro_tpr(cm):
    cm = parse_cm(cm)
    if cm.size == 0:
        return np.nan
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    denominator = tp + fn
    tpr = np.divide(
        tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )
    return float(np.mean(tpr)) if tpr.size else np.nan

def cm_macro_fpr(cm):
    cm = parse_cm(cm)
    if cm.size == 0:
        return np.nan
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    denominator = fp + tn
    fpr = np.divide(
        fp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )
    return float(np.mean(fpr)) if fpr.size else np.nan

def cm_labels_micro_f1(cm, labels):
    cm = parse_cm(cm)
    if cm.size == 0:
        return np.nan
    num_labels = min(cm.shape)
    labels = [int(label) for label in labels if 0 <= int(label) < num_labels]
    if not labels:
        return np.nan
    cm = cm[:num_labels, :num_labels]
    tp = np.diag(cm)[labels].sum()
    predicted = cm[:, labels].sum()
    support = cm[labels, :].sum()
    fp = predicted - tp
    fn = support - tp
    denominator = 2 * tp + fp + fn
    return float((2 * tp) / denominator) if denominator > 0 else 0.0

METRIC_FUNCTIONS = {
    "accuracy": cm_accuracy,
    "macro_f1": cm_macro_f1,
    "macro_tpr": cm_macro_tpr,
    "macro_fpr": cm_macro_fpr,
}

def final_metrics(cm_results, minority_map=None):
    required_columns = {"Client", "CM", "aggregator", "round"}
    missing_columns = sorted(required_columns.difference(cm_results.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    data = cm_results.copy()
    identity_columns = [
        column
        for column in [
            "datasets",
            "experiment type",
            "imbalance",
            "heterogeneity",
            "samplesize_ratio",
            "Seed",
            "aggregator",
        ]
        if column in data.columns
    ]
    data["round"] = pd.to_numeric(data["round"], errors="raise").astype(int)
    final_rounds = data.groupby(identity_columns, dropna=False)["round"].transform("max")
    data = data.loc[data["round"].eq(final_rounds)].copy()
    group_columns = identity_columns + ["round"]

    global_rows = data.loc[data["Client"].eq(-1)].copy()
    if global_rows.empty:
        raise ValueError("No global confusion matrices with Client=-1 were found.")

    for metric_name, metric_function in METRIC_FUNCTIONS.items():
        global_rows[metric_name] = global_rows["CM"].map(metric_function)

    result = (
        global_rows.groupby(group_columns, dropna=False, sort=False)[list(METRIC_FUNCTIONS)]
        .mean()
        .reset_index()
    )

    client_rows = data.loc[~data["Client"].eq(-1)].copy()
    if client_rows.empty:
        result["WCacc"] = np.nan
        result["STDAcc"] = np.nan
        result["F1Minor"] = np.nan
    else:
        client_rows["client_accuracy"] = client_rows["CM"].map(cm_accuracy)
        wcacc = (
            client_rows.groupby(group_columns, dropna=False, sort=False)["client_accuracy"]
            .min()
            .reset_index(name="WCacc")
        )
        result = result.merge(wcacc, on=group_columns, how="left", validate="one_to_one")

        stdacc = (
            client_rows.groupby(group_columns, dropna=False, sort=False)["client_accuracy"]
            .std(ddof=0)
            .reset_index(name="STDAcc")
        )
        result = result.merge(stdacc, on=group_columns, how="left", validate="one_to_one")

        if minority_map is None:
            raise ValueError(
                "F1Minor requires the in-memory minority map produced by exp_main.py."
            )
        client_rows["minority_labels"] = client_rows["Client"].map(minority_map)
        if client_rows["minority_labels"].isna().any():
            missing_clients = sorted(
                client_rows.loc[client_rows["minority_labels"].isna(), "Client"]
                .astype(int)
                .unique()
                .tolist()
            )
            raise ValueError(f"Missing minority labels for clients: {missing_clients}")

        client_rows["F1Minor_client"] = client_rows.apply(
            lambda row: cm_labels_micro_f1(
                row["CM"],
                row["minority_labels"],
            ),
            axis=1,
        )
        f1minor = (
            client_rows.groupby(group_columns, dropna=False, sort=False)["F1Minor_client"]
            .mean()
            .reset_index(name="F1Minor")
        )
        result = result.merge(f1minor, on=group_columns, how="left", validate="one_to_one")

    result = result.sort_values(group_columns, kind="stable").reset_index(drop=True)
    return result.drop(columns="round")

def default_output_path(input_path):
    input_path = Path(input_path)
    stem = input_path.stem
    if stem.startswith("cm"):
        stem = f"metrics{stem[2:]}"
    else:
        stem = f"{stem}_metrics"
    return input_path.with_name(f"{stem}.csv")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()

def main():
    args = parse_args()
    output_path = args.output or default_output_path(args.input)
    cm_results = pd.read_pickle(args.input)
    metrics = final_metrics(cm_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(f"Saved metrics to: {output_path}")
    print("Metrics:")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()

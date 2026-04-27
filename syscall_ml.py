from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


TRAIN_REQUIRED_COLUMNS = ["pid", "syscall", "label"]
INFERENCE_REQUIRED_COLUMNS = ["pid", "syscall"]
DEFAULT_BASE_CALLS = ["open", "read", "write", "execve", "fork"]
STRACE_LINE_RE = re.compile(r"^\s*(?P<pid>\d+)\s+(?P<syscall>[a-zA-Z_][a-zA-Z0-9_]*)\(")
BASIC_BENIGN_COMMANDS = [
    ["ls", "-la"],
    ["cat", "/etc/hosts"],
    ["uname", "-a"],
    ["id"],
    ["whoami"],
    ["ps", "aux"],
]
GROUP_KEY_COLUMN = "process_uid"


@dataclass
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class DatasetSummary:
    rows: int
    unique_pids: int
    class_counts: dict[int, int]
    class_pid_counts: dict[int, int]


@dataclass
class CVSummary:
    model_name: str
    mean_f1: float
    mean_auc: float
    mean_precision: float
    mean_recall: float
    folds_used: int


@dataclass
class SyscallFeatureExtractor:
    base_calls: list[str] = field(default_factory=lambda: DEFAULT_BASE_CALLS.copy())
    top_bigrams: int = 50
    min_bigram_freq: int = 2
    bigram_vocab_: list[str] = field(default_factory=list)
    feature_columns_: list[str] = field(default_factory=list)

    def fit(self, windows: list[list[str]]) -> "SyscallFeatureExtractor":
        bigram_counts: Counter[str] = Counter()
        for window in windows:
            for i in range(len(window) - 1):
                bigram_counts[f"{window[i]}__{window[i + 1]}"] += 1

        candidates = [bg for bg, count in bigram_counts.most_common() if count >= self.min_bigram_freq]
        self.bigram_vocab_ = candidates[: self.top_bigrams]
        sample = self._transform_windows(windows[:1] if windows else [])
        self.feature_columns_ = sample.columns.tolist()
        return self

    def transform(self, windows: list[list[str]]) -> pd.DataFrame:
        if not self.feature_columns_:
            raise ValueError("Feature extractor must be fit before transform.")

        frame = self._transform_windows(windows)
        for col in self.feature_columns_:
            if col not in frame.columns:
                frame[col] = 0
        return frame[self.feature_columns_]

    def fit_transform(self, windows: list[list[str]]) -> pd.DataFrame:
        self.fit(windows)
        return self.transform(windows)

    def _transform_windows(self, windows: list[list[str]]) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for window in windows:
            counter = Counter(window)
            length = len(window)
            row: dict[str, float] = {
                "length": float(length),
                "unique": float(len(set(window))),
                "entropy": float(entropy(list(counter.values()))) if counter else 0.0,
            }

            for syscall in self.base_calls:
                count = counter.get(syscall, 0)
                row[f"{syscall}_count"] = float(count)
                row[f"{syscall}_ratio"] = float(count / length) if length else 0.0

            if self.bigram_vocab_:
                window_bigrams = Counter(
                    f"{window[i]}__{window[i + 1]}" for i in range(len(window) - 1)
                )
                for bigram in self.bigram_vocab_:
                    row[f"bigram::{bigram}"] = float(window_bigrams.get(bigram, 0))

            rows.append(row)

        return pd.DataFrame(rows).fillna(0)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def parse_int_grid(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def current_utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def make_session_id(prefix: str = "session") -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"


def ensure_process_uid(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if GROUP_KEY_COLUMN in frame.columns:
        frame[GROUP_KEY_COLUMN] = frame[GROUP_KEY_COLUMN].astype(str)
        return frame

    frame[GROUP_KEY_COLUMN] = frame["pid"].astype(int).astype(str)
    return frame


def get_group_column(df: pd.DataFrame) -> str:
    return GROUP_KEY_COLUMN if GROUP_KEY_COLUMN in df.columns else "pid"


def load_syscalls(csv_path: str, require_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = TRAIN_REQUIRED_COLUMNS if require_labels else INFERENCE_REQUIRED_COLUMNS
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_columns).copy()
    df["pid"] = df["pid"].astype(int)
    df["syscall"] = df["syscall"].astype(str)
    if require_labels:
        if "label" not in df.columns:
            raise ValueError("Training data must include the 'label' column.")
        df["label"] = df["label"].astype(int)
    elif "label" in df.columns:
        parsed = pd.to_numeric(df["label"], errors="coerce")
        if parsed.notna().any():
            df["label"] = parsed.astype("Int64")
        else:
            df = df.drop(columns=["label"])
    return ensure_process_uid(df)


def ensure_linux_strace() -> None:
    if platform.system().lower() != "linux":
        raise RuntimeError("The collect command is supported only on Linux because it relies on strace.")
    if shutil.which("strace") is None:
        raise RuntimeError("strace was not found in PATH. Install strace to use the collect command.")


def parse_strace_output(raw_trace_path: str, session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(raw_trace_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = STRACE_LINE_RE.match(line)
            if not match:
                continue
            rows.append(
                {
                    "pid": int(match.group("pid")),
                    "syscall": match.group("syscall"),
                    GROUP_KEY_COLUMN: f"{session_id}:{match.group('pid')}",
                }
            )
    return rows


def append_rows_to_csv(csv_path: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    frame = pd.DataFrame(rows)
    exists = os.path.exists(csv_path)
    frame.to_csv(csv_path, mode="a", header=not exists, index=False)
    return len(frame)


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    group_col = get_group_column(df)
    pid_labels = df.groupby(group_col)["label"].first()
    return DatasetSummary(
        rows=int(len(df)),
        unique_pids=int(df[group_col].nunique()),
        class_counts={int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        class_pid_counts={int(k): int(v) for k, v in pid_labels.value_counts().sort_index().items()},
    )


def validate_pid_labels(df: pd.DataFrame) -> None:
    group_col = get_group_column(df)
    mixed = df.groupby(group_col)["label"].nunique()
    bad_pids = mixed[mixed > 1].index.tolist()
    if bad_pids:
        raise ValueError(f"Each process group must map to exactly one label. Mixed labels found for groups: {bad_pids[:10]}")


def split_holdout_by_pid(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> SplitData:
    validate_pid_labels(df)
    group_col = get_group_column(df)
    pid_labels = df.groupby(group_col)["label"].first().reset_index()
    rng = np.random.default_rng(random_state)
    test_pids: list[str] = []

    for label, group in pid_labels.groupby("label"):
        pids = group[group_col].astype(str).tolist()
        rng.shuffle(pids)
        if len(pids) < 2:
            raise ValueError(f"Not enough distinct pid groups for label={label}. Need at least 2.")
        n_test = max(1, int(round(len(pids) * test_size)))
        n_test = min(n_test, len(pids) - 1)
        test_pids.extend(pids[:n_test])

    test_pid_set = set(test_pids)
    train_df = df[~df[group_col].astype(str).isin(test_pid_set)].copy()
    test_df = df[df[group_col].astype(str).isin(test_pid_set)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Holdout split failed: train or test dataframe is empty.")
    return SplitData(train_df=train_df, test_df=test_df)


def sliding_window(df: pd.DataFrame, window_size: int) -> tuple[list[list[str]], pd.Series, pd.Series]:
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    validate_pid_labels(df)
    group_col = get_group_column(df)
    windows: list[list[str]] = []
    labels: list[int] = []
    pids: list[str] = []

    for pid, group in df.groupby(group_col, sort=False):
        calls = group["syscall"].tolist()
        label = int(group["label"].iloc[0])
        if len(calls) < window_size:
            continue
        for i in range(len(calls) - window_size + 1):
            windows.append(calls[i : i + window_size])
            labels.append(label)
            pids.append(str(pid))

    return windows, pd.Series(labels), pd.Series(pids)


def compute_scale_pos_weight(y: pd.Series) -> float:
    counts = y.value_counts().to_dict()
    negative = counts.get(0, 0)
    positive = counts.get(1, 0)
    if positive == 0 or negative == 0:
        return 1.0
    return float(negative / positive)


def build_models(y: pd.Series | None = None) -> dict[str, Any]:
    scale_pos_weight = compute_scale_pos_weight(y) if y is not None else 1.0
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        ),
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    models: dict[str, Any],
    n_splits: int = 3,
) -> list[CVSummary]:
    unique_groups = groups.nunique()
    n_splits = min(n_splits, unique_groups)
    if n_splits < 2:
        raise ValueError("Need at least 2 distinct pid groups for GroupKFold.")

    gkf = GroupKFold(n_splits=n_splits)
    summaries: list[CVSummary] = []

    for name, model in models.items():
        f1_scores: list[float] = []
        auc_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, zero_division=0))

            if y_test.nunique() < 2:
                print(f"[!] {name} | Fold {fold_idx} skipped AUC: only one class in y_test")
            else:
                y_prob = model.predict_proba(X_test)[:, 1]
                auc_scores.append(roc_auc_score(y_test, y_prob))

        summaries.append(
            CVSummary(
                model_name=name,
                mean_f1=float(np.mean(f1_scores)) if f1_scores else float("nan"),
                mean_auc=float(np.mean(auc_scores)) if auc_scores else float("nan"),
                mean_precision=float(np.mean(precision_scores)) if precision_scores else float("nan"),
                mean_recall=float(np.mean(recall_scores)) if recall_scores else float("nan"),
                folds_used=len(f1_scores),
            )
        )

    return summaries


def evaluate_holdout(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics: dict[str, Any] = {
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "classification_report_dict": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if y_test.nunique() >= 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        metrics["probabilities"] = [float(v) for v in y_prob]
    else:
        metrics["roc_auc"] = None
        metrics["probabilities"] = [0.0 for _ in range(len(y_test))]

    metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
    metrics["predictions"] = [int(v) for v in y_pred]
    metrics["true_labels"] = [int(v) for v in y_test.tolist()]
    return metrics


def save_predictions(rows: list[dict[str, Any]], output_dir: str, filename: str) -> str:
    path = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_cv_results_table(results: list[dict[str, Any]], output_dir: str) -> tuple[str, str]:
    frame = pd.DataFrame(results).sort_values(
        by=["cv_f1", "cv_auc", "window_size", "top_bigrams"],
        ascending=[False, False, True, True],
    )
    csv_path = os.path.join(output_dir, "cv_results.csv")
    md_path = os.path.join(output_dir, "cv_results.md")
    frame.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as fh:
        columns = frame.columns.tolist()
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in frame.itertuples(index=False, name=None):
            fh.write("| " + " | ".join(str(value) for value in row) + " |\n")
    return csv_path, md_path


def save_experiment_plot(results: list[dict[str, Any]], output_dir: str) -> str:
    frame = pd.DataFrame(results).sort_values(by=["window_size", "top_bigrams", "model_name"])
    frame["config"] = frame.apply(lambda row: f"w={int(row['window_size'])}, bg={int(row['top_bigrams'])}", axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, group in frame.groupby("model_name"):
        ax.plot(group["config"], group["cv_f1"], marker="o", label=f"{model_name} F1")
    ax.set_title("Cross-validation F1 by configuration")
    ax.set_ylabel("CV F1")
    ax.set_xlabel("Configuration")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "cv_f1_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_confusion_matrix_plot(matrix: list[list[int]], output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title("Holdout confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_markdown_report(
    output_dir: str,
    dataset_summary: dict[str, Any],
    train_summary: dict[str, Any],
    best_config: dict[str, Any],
    holdout_metrics: dict[str, Any],
) -> str:
    path = os.path.join(output_dir, "experiment_report.md")
    lines = [
        "# Syscall ML Experiment Report",
        "",
        "## Dataset Summary",
        f"- Rows: {dataset_summary['rows']}",
        f"- Unique PIDs: {dataset_summary['unique_pids']}",
        f"- Class counts (rows): {dataset_summary['class_counts']}",
        f"- Class counts (PIDs): {dataset_summary['class_pid_counts']}",
        "",
        "## Training Configuration",
        f"- Candidate window sizes: {train_summary['window_sizes']}",
        f"- Candidate top bigrams: {train_summary['top_bigrams_grid']}",
        f"- CV splits: {train_summary['cv_splits']}",
        f"- Holdout ratio: {train_summary['test_size']}",
        "",
        "## Selected Configuration",
        f"- Model: {best_config['model_name']}",
        f"- Window size: {best_config['window_size']}",
        f"- Top bigrams: {best_config['top_bigrams']}",
        f"- Feature count: {best_config['feature_count']}",
        f"- CV F1: {best_config['cv_f1']:.4f}",
        f"- CV AUC: {best_config['cv_auc']:.4f}" if best_config["cv_auc"] == best_config["cv_auc"] else "- CV AUC: n/a",
        "",
        "## Holdout Metrics",
        f"- Accuracy: {holdout_metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {holdout_metrics['balanced_accuracy']:.4f}",
        f"- Precision: {holdout_metrics['precision']:.4f}",
        f"- Recall: {holdout_metrics['recall']:.4f}",
        f"- F1-score: {holdout_metrics['f1']:.4f}",
        f"- ROC-AUC: {holdout_metrics['roc_auc']:.4f}" if holdout_metrics["roc_auc"] is not None else "- ROC-AUC: n/a",
        "",
        "## Classification Report",
        "```text",
        holdout_metrics["classification_report"].rstrip(),
        "```",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def generate_shap_summary(model: Any, X_test: pd.DataFrame, output_dir: str) -> str:
    sample = X_test.sample(min(100, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)
    output_path = os.path.join(output_dir, "shap_summary.png")
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def choose_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in results if not pd.isna(item["cv_f1"])]
    if not valid:
        raise ValueError("No valid training results were produced.")
    return max(valid, key=lambda item: (item["cv_f1"], item["cv_auc"]))


def load_artifacts(artifacts_dir: str) -> tuple[dict[str, Any], Any, Any, int]:
    with open(os.path.join(artifacts_dir, "metrics.json"), "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    model = joblib.load(os.path.join(artifacts_dir, "model.pkl"))
    extractor = joblib.load(os.path.join(artifacts_dir, "feature_extractor.pkl"))
    window_size = int(metrics["selected_window_size"])
    return metrics, model, extractor, window_size


def normalize_optional_label(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def build_window_rows(
    df: pd.DataFrame,
    window_size: int,
    predictions: list[int],
    probabilities: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    group_col = get_group_column(df)
    for process_uid, group in df.groupby(group_col, sort=False):
        syscalls = group["syscall"].tolist()
        true_label = normalize_optional_label(group["label"].iloc[0]) if "label" in group.columns else None
        if len(syscalls) < window_size:
            continue
        for start in range(len(syscalls) - window_size + 1):
            rows.append(
                {
                    "pid": int(group["pid"].iloc[0]),
                    GROUP_KEY_COLUMN: str(process_uid),
                    "window_start": int(start),
                    "window_end": int(start + window_size - 1),
                    "window_size": int(window_size),
                    "syscall_sequence": " ".join(syscalls[start : start + window_size]),
                    "predicted_label": int(predictions[idx]),
                    "malicious_probability": float(probabilities[idx]),
                    "true_label": true_label,
                }
            )
            idx += 1
    return rows


def aggregate_pid_rows(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(window_rows)
    agg = frame.groupby(GROUP_KEY_COLUMN).agg(
        pid=("pid", "first"),
        window_count=("pid", "size"),
        malicious_probability_mean=("malicious_probability", "mean"),
        malicious_probability_max=("malicious_probability", "max"),
        predicted_label_majority=("predicted_label", lambda s: int(s.mode().iloc[0])),
    )
    if "true_label" in frame.columns and frame["true_label"].notna().any():
        agg["true_label"] = frame.groupby(GROUP_KEY_COLUMN)["true_label"].first()
    return agg.reset_index().to_dict(orient="records")


def find_suspicious_pids(pid_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    suspicious = [
        row for row in pid_rows
        if float(row["malicious_probability_max"]) >= threshold
        or float(row["malicious_probability_mean"]) >= threshold
        or int(row["predicted_label_majority"]) == 1
    ]
    suspicious = sorted(
        suspicious,
        key=lambda row: (
            float(row["malicious_probability_max"]),
            float(row["malicious_probability_mean"]),
            int(row["window_count"]),
        ),
        reverse=True,
    )
    return suspicious


def print_suspicious_summary(suspicious_pids: list[dict[str, Any]], limit: int = 10) -> None:
    print(f"[+] Suspicious PID count: {len(suspicious_pids)}")
    for row in suspicious_pids[:limit]:
        true_label = row.get("true_label", "n/a")
        print(
            "  "
            f"pid={row['pid']} "
            f"windows={row['window_count']} "
            f"mean={float(row['malicious_probability_mean']):.3f} "
            f"max={float(row['malicious_probability_max']):.3f} "
            f"pred={row['predicted_label_majority']} "
            f"true={true_label}"
        )


def predict_from_dataframe(df: pd.DataFrame, model: Any, extractor: Any, window_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inference_df = df.copy()
    has_true_labels = "label" in inference_df.columns and inference_df["label"].notna().any()
    if "label" not in inference_df.columns:
        inference_df["label"] = 0
    else:
        inference_df["label"] = inference_df["label"].fillna(0).astype(int)

    windows, _, _ = sliding_window(inference_df, window_size)
    if not windows:
        raise ValueError("No windows generated for inference. Check the dataset and window size.")
    X = extractor.transform(windows)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    window_rows = build_window_rows(
        df=df if has_true_labels else inference_df.drop(columns=["label"], errors="ignore"),
        window_size=window_size,
        predictions=[int(v) for v in predictions],
        probabilities=[float(v) for v in probabilities],
    )
    pid_rows = aggregate_pid_rows(window_rows)
    return window_rows, pid_rows


def collect_trace_rows(
    output_csv: str,
    raw_dir: str,
    label: int | None,
    commands_to_run: list[list[str]],
    session_prefix: str = "collect",
) -> dict[str, Any]:
    ensure_dir(os.path.dirname(output_csv) or ".")
    ensure_dir(raw_dir)

    session_id = make_session_id(session_prefix)
    total_written = 0
    command_summaries: list[dict[str, Any]] = []

    for idx, current_command in enumerate(commands_to_run, start=1):
        raw_trace_path = os.path.join(
            raw_dir,
            f"strace_{session_id}_{idx}.log",
        )
        strace_cmd = [
            "strace",
            "-f",
            "-qq",
            "-o",
            raw_trace_path,
            *current_command,
        ]

        print("[*] Running collector command:")
        print(" ".join(strace_cmd))
        completed = subprocess.run(strace_cmd, check=False)
        print(f"[*] Target command exit code: {completed.returncode}")

        rows = parse_strace_output(raw_trace_path, session_id=session_id)
        if label is not None:
            for row in rows:
                row["label"] = int(label)
        for row in rows:
            row["collected_at"] = current_utc_iso()
            row["source_command"] = " ".join(current_command)
            row["session_id"] = session_id

        written = append_rows_to_csv(output_csv, rows)
        total_written += written
        command_summaries.append(
            {
                "raw_trace": raw_trace_path,
                "rows_written": written,
                "label": label,
                "session_id": session_id,
                "source_command": current_command,
                "target_exit_code": completed.returncode,
                "collected_at": current_utc_iso(),
            }
        )

    summary = {
        "output_csv": output_csv,
        "total_rows_written": total_written,
        "label": label,
        "session_id": session_id,
        "commands": command_summaries,
        "collected_at": current_utc_iso(),
    }
    save_json(os.path.join(raw_dir, "last_collect_summary.json"), summary)
    return summary


def print_demo_instructions() -> None:
    print("=== DIAMORPHINE DEMO ===")
    print("1. On a clean Linux system, collect benign traces:")
    print("   python syscall_ml.py demo benign")
    print("2. Install/load Diamorphine yourself.")
    print("3. On the infected system, collect suspicious traces from the same built-in commands:")
    print("   python syscall_ml.py demo infected")
    print("4. Train the model:")
    print("   python syscall_ml.py demo train")
    print("5. Show live detection using the current system state:")
    print("   python syscall_ml.py demo detect")
    print("")
    print("The built-in demo commands reuse the same simple Linux commands before and after infection.")
    print("That gives you an easy before/after comparison for a diploma demo.")


def command_collect(args: argparse.Namespace) -> None:
    ensure_linux_strace()

    if args.benign and args.suspicious:
        raise ValueError("Use only one of --benign or --suspicious.")
    if args.benign:
        args.label = 0
    elif args.suspicious:
        args.label = 1

    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]

    if command:
        commands_to_run = [command]
    else:
        commands_to_run = [cmd[:] for cmd in BASIC_BENIGN_COMMANDS]

    summary = collect_trace_rows(
        output_csv=args.output_csv,
        raw_dir=args.raw_dir,
        label=args.label,
        commands_to_run=commands_to_run,
    )

    print(f"[+] Parsed syscall rows: {summary['total_rows_written']}")
    print(args.output_csv)


def command_check(args: argparse.Namespace) -> None:
    df = load_syscalls(args.csv)
    group_col = get_group_column(df)

    label_counts = df["label"].value_counts(dropna=False).sort_index()
    pid_label_counts = df.groupby(group_col)["label"].first().value_counts(dropna=False).sort_index()
    summary = summarize_dataset(df)

    print("=== DATASET CHECK ===")
    print(f"CSV: {args.csv}")
    print(f"Rows: {summary.rows}")
    print(f"Unique process groups: {summary.unique_pids}")
    print("\nLABEL COUNTS")
    print(label_counts)
    print("\nPROCESS GROUP LABEL COUNTS")
    print(pid_label_counts)

    unique_labels = sorted(df["label"].dropna().unique().tolist())
    if len(unique_labels) < 2:
        print("\n[!] Problem: only one class is present in the dataset.")
        print("[!] Training will not produce meaningful suspicious-behavior detection metrics.")
        print("[!] Add more labeled data for the missing class before running train.")
        return

    pid_label_map = df.groupby(group_col)["label"].first()
    min_pid_count = pid_label_map.value_counts().min()
    if min_pid_count < 2:
        print("\n[!] Warning: at least one class has fewer than 2 unique process groups.")
        print("[!] Holdout split and cross-validation may fail or be unstable.")
        return

    print("\n[+] Dataset looks usable for training.")


def command_train(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    df = load_syscalls(args.csv)
    dataset_summary = summarize_dataset(df)
    split = split_holdout_by_pid(df, test_size=args.test_size, random_state=42)

    print(f"[*] Full dataset shape: {df.shape}")
    print(f"[*] Train shape: {split.train_df.shape} | Test shape: {split.test_df.shape}")

    comparison_results: list[dict[str, Any]] = []
    window_sizes = parse_int_grid(args.window_sizes)
    top_bigrams_grid = parse_int_grid(args.top_bigrams_grid)

    for window_size in window_sizes:
        for top_bigrams in top_bigrams_grid:
            train_windows, y_train, train_groups = sliding_window(split.train_df, window_size)
            test_windows, y_test, _ = sliding_window(split.test_df, window_size)

            if not train_windows or not test_windows:
                print(f"[!] Skipping window_size={window_size}: no train/test windows generated.")
                continue

            extractor = SyscallFeatureExtractor(top_bigrams=top_bigrams, min_bigram_freq=args.min_bigram_freq)
            X_train = extractor.fit_transform(train_windows)
            X_test = extractor.transform(test_windows)

            cv_results = cross_validate_models(
                X=X_train,
                y=y_train,
                groups=train_groups,
                models=build_models(y_train),
                n_splits=args.cv_splits,
            )

            for summary in cv_results:
                print(
                    f"{summary.model_name} | window={window_size} | top_bigrams={top_bigrams} | "
                    f"CV F1={summary.mean_f1:.3f} | CV AUC={summary.mean_auc:.3f}"
                )
                comparison_results.append(
                    {
                        "window_size": window_size,
                        "top_bigrams": top_bigrams,
                        "model_name": summary.model_name,
                        "cv_f1": summary.mean_f1,
                        "cv_auc": summary.mean_auc,
                        "cv_precision": summary.mean_precision,
                        "cv_recall": summary.mean_recall,
                        "feature_count": len(extractor.feature_columns_),
                        "extractor": extractor,
                        "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                    }
                )

    best = choose_best_result(comparison_results)
    print(
        f"\n[+] Selected model: {best['model_name']} | "
        f"window={best['window_size']} | top_bigrams={best['top_bigrams']} | CV F1={best['cv_f1']:.3f}"
    )

    final_model = build_models(best["y_train"])[best["model_name"]]
    final_model.fit(best["X_train"], best["y_train"])
    holdout_metrics = evaluate_holdout(final_model, best["X_test"], best["y_test"])

    print("\n=== HOLDOUT EVALUATION ===")
    print(holdout_metrics["classification_report"])
    print(holdout_metrics["confusion_matrix"])
    print(f"Holdout F1: {holdout_metrics['f1']:.3f}")
    if holdout_metrics["roc_auc"] is not None:
        print(f"Holdout ROC-AUC: {holdout_metrics['roc_auc']:.3f}")

    model_path = os.path.join(args.output_dir, "model.pkl")
    feature_columns_path = os.path.join(args.output_dir, "feature_columns.pkl")
    extractor_path = os.path.join(args.output_dir, "feature_extractor.pkl")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    dataset_summary_path = os.path.join(args.output_dir, "dataset_summary.json")
    training_config_path = os.path.join(args.output_dir, "training_config.json")

    joblib.dump(final_model, model_path)
    joblib.dump(best["extractor"].feature_columns_, feature_columns_path)
    joblib.dump(best["extractor"], extractor_path)

    shap_path = generate_shap_summary(final_model, best["X_test"], args.output_dir)

    serializable_results = [
        {
            "window_size": item["window_size"],
            "top_bigrams": item["top_bigrams"],
            "model_name": item["model_name"],
            "cv_f1": item["cv_f1"],
            "cv_auc": item["cv_auc"],
            "cv_precision": item["cv_precision"],
            "cv_recall": item["cv_recall"],
            "feature_count": item["feature_count"],
        }
        for item in comparison_results
    ]

    cv_results_csv, cv_results_md = save_cv_results_table(serializable_results, args.output_dir)
    cv_plot_path = save_experiment_plot(serializable_results, args.output_dir)
    confusion_plot_path = save_confusion_matrix_plot(holdout_metrics["confusion_matrix"], args.output_dir)

    holdout_rows = []
    for idx, (true_label, pred_label, probability) in enumerate(
        zip(holdout_metrics["true_labels"], holdout_metrics["predictions"], holdout_metrics["probabilities"])
    ):
        holdout_rows.append(
            {
                "sample_index": idx,
                "true_label": true_label,
                "predicted_label": pred_label,
                "malicious_probability": probability,
            }
        )
    holdout_predictions_path = save_predictions(holdout_rows, args.output_dir, "holdout_predictions.csv")

    dataset_summary_payload = {
        "rows": dataset_summary.rows,
        "unique_pids": dataset_summary.unique_pids,
        "class_counts": dataset_summary.class_counts,
        "class_pid_counts": dataset_summary.class_pid_counts,
    }
    training_config_payload = {
        "window_sizes": window_sizes,
        "top_bigrams_grid": top_bigrams_grid,
        "test_size": args.test_size,
        "min_bigram_freq": args.min_bigram_freq,
        "cv_splits": args.cv_splits,
    }
    save_json(dataset_summary_path, dataset_summary_payload)
    save_json(training_config_path, training_config_payload)

    report_path = save_markdown_report(
        output_dir=args.output_dir,
        dataset_summary=dataset_summary_payload,
        train_summary=training_config_payload,
        best_config={
            "model_name": best["model_name"],
            "window_size": best["window_size"],
            "top_bigrams": best["top_bigrams"],
            "feature_count": best["feature_count"],
            "cv_f1": best["cv_f1"],
            "cv_auc": best["cv_auc"],
        },
        holdout_metrics=holdout_metrics,
    )

    save_json(
        metrics_path,
        {
            "selected_model": best["model_name"],
            "selected_window_size": best["window_size"],
            "selected_top_bigrams": best["top_bigrams"],
            "cv_results": serializable_results,
            "holdout_metrics": holdout_metrics,
            "artifacts": {
                "model": model_path,
                "feature_columns": feature_columns_path,
                "feature_extractor": extractor_path,
                "dataset_summary": dataset_summary_path,
                "training_config": training_config_path,
                "cv_results_csv": cv_results_csv,
                "cv_results_markdown": cv_results_md,
                "cv_plot": cv_plot_path,
                "holdout_predictions": holdout_predictions_path,
                "confusion_matrix_plot": confusion_plot_path,
                "report_markdown": report_path,
                "shap_summary": shap_path,
            },
        },
    )

    print("\n[+] Saved artifacts:")
    for path in [
        model_path,
        feature_columns_path,
        extractor_path,
        dataset_summary_path,
        training_config_path,
        cv_results_csv,
        cv_results_md,
        cv_plot_path,
        holdout_predictions_path,
        confusion_plot_path,
        report_path,
        metrics_path,
        shap_path,
    ]:
        print(path)


def command_predict(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    _, model, extractor, window_size = load_artifacts(args.artifacts_dir)
    df = load_syscalls(args.csv, require_labels=False)
    window_rows, pid_rows = predict_from_dataframe(df, model, extractor, window_size)
    windows_path = save_predictions(window_rows, args.output_dir, "window_predictions.csv")
    pid_path = save_predictions(pid_rows, args.output_dir, "pid_predictions.csv")
    suspicious_pids = find_suspicious_pids(pid_rows, args.threshold)
    suspicious_path = save_predictions(suspicious_pids, args.output_dir, "suspicious_pids.csv")
    print("[+] Inference complete.")
    print(windows_path)
    print(pid_path)
    print(suspicious_path)
    print_suspicious_summary(suspicious_pids)


def build_status_payload(source_csv: str, window_rows: list[dict[str, Any]], pid_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    suspicious_pids = find_suspicious_pids(pid_rows, threshold)
    return {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "source_csv": source_csv,
        "window_count": len(window_rows),
        "pid_count": len(pid_rows),
        "threshold": threshold,
        "suspicious_pid_count": len(suspicious_pids),
        "top_suspicious_pids": suspicious_pids[:10],
    }


def print_live_summary(status: dict[str, Any]) -> None:
    print(
        f"[{status['updated_at']}] windows={status['window_count']} "
        f"pids={status['pid_count']} suspicious_pids={status['suspicious_pid_count']}"
    )
    for row in status["top_suspicious_pids"][:5]:
        print(
            "  "
            f"pid={row['pid']} "
            f"max_prob={float(row['malicious_probability_max']):.3f} "
            f"mean_prob={float(row['malicious_probability_mean']):.3f} "
            f"label={row['predicted_label_majority']}"
        )


def command_realtime(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    _, model, extractor, window_size = load_artifacts(args.artifacts_dir)
    last_signature: tuple[float, int] | None = None

    while True:
        if not os.path.exists(args.csv):
            print(f"[!] Waiting for input file: {args.csv}")
            if args.once:
                break
            time.sleep(args.interval)
            continue

        stat = os.stat(args.csv)
        signature = (stat.st_mtime, stat.st_size)
        if signature == last_signature and not args.once:
            time.sleep(args.interval)
            continue

        try:
            df = load_syscalls(args.csv, require_labels=False)
            window_rows, pid_rows = predict_from_dataframe(df, model, extractor, window_size)
        except Exception as exc:
            print(f"[!] Live analysis failed: {exc}")
            if args.once:
                raise
            time.sleep(args.interval)
            continue

        window_path = save_predictions(window_rows, args.output_dir, "live_window_predictions.csv")
        pid_path = save_predictions(pid_rows, args.output_dir, "live_pid_predictions.csv")
        suspicious_pids = find_suspicious_pids(pid_rows, args.threshold)
        suspicious_path = save_predictions(suspicious_pids, args.output_dir, "live_suspicious_pids.csv")
        status = build_status_payload(args.csv, window_rows, pid_rows, args.threshold)
        status["window_predictions_file"] = window_path
        status["pid_predictions_file"] = pid_path
        status["suspicious_pids_file"] = suspicious_path
        status_path = os.path.join(args.output_dir, "live_status.json")
        save_json(status_path, status)
        print_live_summary(status)

        last_signature = signature
        if args.once:
            break
        time.sleep(args.interval)


def command_demo(args: argparse.Namespace) -> None:
    if args.demo_action == "guide":
        print_demo_instructions()
        return

    if args.demo_action in {"benign", "infected", "detect"}:
        ensure_linux_strace()

    if args.demo_action == "benign":
        summary = collect_trace_rows(
            output_csv=args.csv,
            raw_dir=args.raw_dir,
            label=0,
            commands_to_run=[cmd[:] for cmd in BASIC_BENIGN_COMMANDS],
            session_prefix="demo_benign",
        )
        print(f"[+] Benign demo data collected: {summary['total_rows_written']} rows")
        print(args.csv)
        return

    if args.demo_action == "infected":
        summary = collect_trace_rows(
            output_csv=args.csv,
            raw_dir=args.raw_dir,
            label=1,
            commands_to_run=[cmd[:] for cmd in BASIC_BENIGN_COMMANDS],
            session_prefix="demo_infected",
        )
        print(f"[+] Infected demo data collected: {summary['total_rows_written']} rows")
        print(args.csv)
        return

    if args.demo_action == "train":
        train_args = argparse.Namespace(
            csv=args.csv,
            output_dir=args.artifacts_dir,
            window_sizes=args.window_sizes,
            top_bigrams_grid=args.top_bigrams_grid,
            test_size=args.test_size,
            min_bigram_freq=args.min_bigram_freq,
            cv_splits=args.cv_splits,
        )
        command_train(train_args)
        return

    if args.demo_action == "detect":
        temp_csv = args.detect_csv
        collect_trace_rows(
            output_csv=temp_csv,
            raw_dir=args.raw_dir,
            label=None,
            commands_to_run=[cmd[:] for cmd in BASIC_BENIGN_COMMANDS],
            session_prefix="demo_detect",
        )
        predict_args = argparse.Namespace(
            csv=temp_csv,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
        )
        command_predict(predict_args)
        return

    raise ValueError(f"Unknown demo action: {args.demo_action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file syscall ML tool: training, inference, and near-real-time monitoring."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", aliases=["c"], help="Collect syscall traces on Linux using strace.")
    collect_parser.add_argument("--output-csv", default="syscalls.csv", help="CSV file to append parsed syscalls to.")
    collect_parser.add_argument("--raw-dir", default="collector_output", help="Directory to save raw strace logs.")
    collect_parser.add_argument("--label", type=int, default=None, help="Optional label to assign to collected rows.")
    collect_parser.add_argument("-b", "--benign", action="store_true", help="Shortcut for --label 0.")
    collect_parser.add_argument("-s", "--suspicious", action="store_true", help="Shortcut for --label 1.")
    collect_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Target command after '--'. If omitted, a built-in benign command set is collected.",
    )
    collect_parser.set_defaults(func=command_collect)

    check_parser = subparsers.add_parser("check", aliases=["k"], help="Check dataset balance and training readiness.")
    check_parser.add_argument("--csv", default="syscalls.csv", help="Path to dataset CSV.")
    check_parser.set_defaults(func=command_check)

    train_parser = subparsers.add_parser("train", aliases=["t"], help="Train the model and save experiment artifacts.")
    train_parser.add_argument("--csv", default="syscalls.csv", help="Path to training CSV.")
    train_parser.add_argument("--output-dir", default="artifacts", help="Directory to save artifacts.")
    train_parser.add_argument("--window-sizes", default="3,5,10", help="Comma-separated window sizes to compare.")
    train_parser.add_argument("--top-bigrams-grid", default="20,50,100", help="Comma-separated top-bigram values to compare.")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio by pid.")
    train_parser.add_argument("--min-bigram-freq", type=int, default=2, help="Minimum bigram frequency to keep.")
    train_parser.add_argument("--cv-splits", type=int, default=3, help="GroupKFold splits on the training part.")
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", aliases=["p"], help="Run inference on a syscall CSV.")
    predict_parser.add_argument("--csv", default="syscalls.csv", help="Path to input syscall CSV.")
    predict_parser.add_argument("--artifacts-dir", default="artifacts", help="Directory with saved model artifacts.")
    predict_parser.add_argument("--output-dir", default="predictions", help="Directory to save inference outputs.")
    predict_parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for suspicious PID reporting.")
    predict_parser.set_defaults(func=command_predict)

    realtime_parser = subparsers.add_parser("realtime", aliases=["r"], help="Monitor a growing CSV and update live predictions.")
    realtime_parser.add_argument("--csv", default="live_syscalls.csv", help="Path to the live syscall CSV.")
    realtime_parser.add_argument("--artifacts-dir", default="artifacts", help="Directory with trained model artifacts.")
    realtime_parser.add_argument("--output-dir", default="live_output", help="Directory to save live outputs.")
    realtime_parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds.")
    realtime_parser.add_argument("--threshold", type=float, default=0.7, help="Suspicious PID alert threshold.")
    realtime_parser.add_argument("--once", action="store_true", help="Run one analysis pass and exit.")
    realtime_parser.set_defaults(func=command_realtime)

    demo_parser = subparsers.add_parser("demo", help="Easy demo flow for benign vs Diamorphine-infected collection.")
    demo_parser.add_argument(
        "demo_action",
        choices=["guide", "benign", "infected", "train", "detect"],
        help="Demo step to run.",
    )
    demo_parser.add_argument("--csv", default="demo_syscalls.csv", help="Shared demo dataset CSV.")
    demo_parser.add_argument("--raw-dir", default="demo_collector_output", help="Directory for raw demo strace logs.")
    demo_parser.add_argument("--artifacts-dir", default="demo_artifacts", help="Directory for trained demo artifacts.")
    demo_parser.add_argument("--output-dir", default="demo_predictions", help="Directory for demo prediction outputs.")
    demo_parser.add_argument("--detect-csv", default="demo_detect.csv", help="Temporary CSV used by demo detect.")
    demo_parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for suspicious PID reporting.")
    demo_parser.add_argument("--window-sizes", default="3,5,10", help="Comma-separated window sizes to compare.")
    demo_parser.add_argument("--top-bigrams-grid", default="20,50,100", help="Comma-separated top-bigram values to compare.")
    demo_parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio by process group.")
    demo_parser.add_argument("--min-bigram-freq", type=int, default=2, help="Minimum bigram frequency to keep.")
    demo_parser.add_argument("--cv-splits", type=int, default=3, help="GroupKFold splits on the training part.")
    demo_parser.set_defaults(func=command_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

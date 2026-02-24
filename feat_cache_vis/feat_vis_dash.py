"""
Dash app for interactive embedding visualization with image panel.

Usage example:
  python feat_cache_vis/feat_vis_dash.py \
    --cache_dir feat_cache_vis/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_cs/test/seed1 \
    --coords_file vis_umap_coords.npy \
    --image_dir CS_Xplots_2019_2023_test \
    --port 8050
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from flask import send_from_directory

THRESHOLD_STEPS = [5, 10, 15, 20]
DEFAULT_METHODS = ["centroid", "multiprototype"]
LLM_CUTOFF_MIN = 1
LLM_CUTOFF_MAX = 5

SCORE_SOURCE_SPECS = [
    {
        "key": "centroid_5",
        "label": "Centroid 5%",
        "method": "centroid",
        "threshold": 5,
        "arg": "centroid_csv",
        "bottom_col": "is_bottom_5pct",
        "used_in_review_rule": True,
    },
    {
        "key": "mp_5",
        "label": "Multiprototype 5%",
        "method": "multiprototype",
        "threshold": 5,
        "arg": "multiprototype_5pct_csv",
        "bottom_col": "is_bottom_5pct",
        "used_in_review_rule": False,
    },
    {
        "key": "mp_10",
        "label": "Multiprototype 10%",
        "method": "multiprototype",
        "threshold": 10,
        "arg": "multiprototype_10pct_csv",
        "bottom_col": "is_bottom_10pct",
        "used_in_review_rule": True,
    },
    {
        "key": "mp_15",
        "label": "Multiprototype 15%",
        "method": "multiprototype",
        "threshold": 15,
        "arg": "multiprototype_15pct_csv",
        "bottom_col": "is_bottom_15pct",
        "used_in_review_rule": False,
    },
    {
        "key": "mp_20",
        "label": "Multiprototype 20%",
        "method": "multiprototype",
        "threshold": 20,
        "arg": "multiprototype_20pct_csv",
        "bottom_col": "is_bottom_20pct",
        "used_in_review_rule": False,
    },
]
SCORE_SPEC_BY_KEY = {spec["key"]: spec for spec in SCORE_SOURCE_SPECS}

BADGE_COLOR_BY_COUNT = {
    1: "#2A9D8F",
    2: "#E9C46A",
    3: "#F4A261",
    4: "#E76F51",
    5: "#D62828",
}
BADGE_SIZE_BY_COUNT = {1: 9, 2: 11, 3: 13, 4: 15, 5: 17}
MP10_MARGIN_COLUMN = "mp_10__margin_to_other_class"
PREDICTION_COLUMNS = [
    "predicted_label",
    "predicted_word_label",
    "dataset",
    "top3_label_1",
    "top3_prob_1",
    "top3_label_2",
    "top3_prob_2",
    "top3_label_3",
    "top3_prob_3",
    "split",
    "image_path",
    "rationale",
]
EXPORT_COLUMNS = [
    "file_name",
    "ground_truth_num_label",
    "ground_truth_word_label",
    "ground_truth_L2_num_label",
    "predicted_label",
    "predicted_word_label",
    "dataset",
    "top3_label_1",
    "top3_prob_1",
    "top3_label_2",
    "top3_prob_2",
    "top3_label_3",
    "top3_prob_3",
    "split",
    "image_path",
    "rationale",
    "centroid_sim_to_centroid",
    "centroid_pct_rank_in_class",
    "centroid_is_bottom_5pct",
    "mp10_sim_to_centroid",
    "mp10_pct_rank_in_class",
    "mp10_is_bottom_10pct",
    "mp10_margin_to_other_class",
    "reviewer_id",
    "reviewed_at_utc",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dash viewer for cached embedding coords.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Cache folder containing metadata.csv and coords .npy.",
    )
    parser.add_argument(
        "--coords_file",
        type=str,
        required=True,
        help="Coords .npy file name (e.g., vis_umap_coords.npy).",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="URL prefix for images (joined with file_name). Leave empty to disable images.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="Local directory to serve images from at /images (uses file_name).",
    )
    parser.add_argument(
        "--mis_csv",
        type=str,
        default="mis_Qwen_Qwen3-VL-4B-Instruct.csv",
        help="Misclassified CSV with rationale JSON (relative to cache_dir).",
    )
    parser.add_argument(
        "--correct_csv",
        type=str,
        default="correct_Qwen_Qwen3-VL-4B-Instruct.csv",
        help="Correct CSV with rationale JSON (relative to cache_dir).",
    )
    parser.add_argument(
        "--centroid_csv",
        type=str,
        default="scores_centroid.csv",
        help="Centroid score CSV with outlier metrics (relative to cache_dir).",
    )
    parser.add_argument(
        "--multiprototype_5pct_csv",
        type=str,
        default="scores_multiprototype_5pct.csv",
        help="Multi-prototype 5pct score CSV (relative to cache_dir).",
    )
    parser.add_argument(
        "--multiprototype_10pct_csv",
        type=str,
        default="scores_multiprototype_10pct.csv",
        help="Multi-prototype 10pct score CSV (relative to cache_dir).",
    )
    parser.add_argument(
        "--multiprototype_15pct_csv",
        type=str,
        default="scores_multiprototype_15pct.csv",
        help="Multi-prototype 15pct score CSV (relative to cache_dir).",
    )
    parser.add_argument(
        "--multiprototype_20pct_csv",
        type=str,
        default="scores_multiprototype_20pct.csv",
        help="Multi-prototype 20pct score CSV (relative to cache_dir).",
    )
    parser.add_argument(
        "--color_by",
        type=str,
        default="ground_truth_word_label",
        help="Column in metadata.csv to color points.",
    )
    parser.add_argument(
        "--hover",
        type=str,
        default="file_name,ground_truth_word_label,ground_truth_num_label",
        help="Comma-separated metadata columns to show on hover.",
    )
    parser.add_argument(
        "--default_llm_cutoff",
        type=int,
        default=3,
        help="Default LLM score cutoff for review subset mode.",
    )
    parser.add_argument(
        "--review_output_csv",
        type=str,
        default="feat_cache_vis/examiner_confirmed_true_outliers.csv",
        help="Output CSV path for confirmed true outliers.",
    )
    parser.add_argument(
        "--reviewer_id",
        type=str,
        default="unknown",
        help="Reviewer identifier written to confirmed outlier CSV.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dash host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash port.")
    return parser


def _build_image_url(root: str, fname: str) -> str:
    if not root:
        return ""
    root = str(root).rstrip("/")
    return f"{root}/{fname}"


def _resolve_csv_path(cache_dir: str, csv_path: str) -> Path | None:
    if not csv_path:
        return None
    path = Path(csv_path)
    if not path.is_absolute():
        path = Path(cache_dir) / csv_path
    return path


def _score_columns_for_key(source_key: str) -> tuple[str, str, str]:
    return (
        f"{source_key}__flag",
        f"{source_key}__sim_to_centroid",
        f"{source_key}__pct_rank_in_class",
    )


def score_custom_columns() -> list[str]:
    cols: list[str] = []
    for spec in SCORE_SOURCE_SPECS:
        cols.extend(_score_columns_for_key(spec["key"]))
    return cols


def _coerce_bool(value) -> bool | float:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "t"}:
            return True
        if normalized in {"false", "0", "no", "n", "f"}:
            return False
        return np.nan
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return np.nan


def _bool_or_none(value) -> bool | None:
    parsed = _coerce_bool(value)
    if isinstance(parsed, (float, np.floating)) and np.isnan(parsed):
        return None
    return bool(parsed)


def _fmt_num(value, digits: int = 6) -> str:
    if value is None:
        return "(missing)"
    try:
        if pd.isna(value):
            return "(missing)"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct_rank(value) -> str:
    base = _fmt_num(value, digits=6)
    if base == "(missing)":
        return base
    try:
        pct_val = float(value)
    except Exception:
        return base
    return f"{base} ({pct_val * 100:.3f}%)"


def _fmt_bool(value) -> str:
    parsed = _bool_or_none(value)
    if parsed is None:
        return "(missing)"
    return "True" if parsed else "False"


def _resolve_threshold(value) -> int:
    try:
        raw = int(value)
    except Exception:
        return THRESHOLD_STEPS[-1]
    if raw in THRESHOLD_STEPS:
        return raw
    return min(THRESHOLD_STEPS, key=lambda step: abs(step - raw))


def _resolve_llm_cutoff(value) -> int:
    try:
        raw = int(value)
    except Exception:
        raw = 3
    return max(LLM_CUTOFF_MIN, min(LLM_CUTOFF_MAX, raw))


def active_source_keys(selected_methods, threshold_value) -> list[str]:
    methods = set(selected_methods or [])
    threshold = _resolve_threshold(threshold_value)
    return [
        spec["key"]
        for spec in SCORE_SOURCE_SPECS
        if spec["method"] in methods and spec["threshold"] <= threshold
    ]


def _parse_score_from_rationale(raw) -> float | None:
    if not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    score = obj.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def load_llm_score_map(csv_paths: list[Path]) -> dict[str, float]:
    score_map: dict[str, float] = {}
    duplicate_count = 0

    for path in csv_paths:
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "file_name" not in df.columns or "rationale" not in df.columns:
            print(f"[warn] {path.name} missing file_name or rationale; skipping for llm_score.")
            continue

        bad_score = 0
        for row in df.itertuples(index=False):
            fname = getattr(row, "file_name", None)
            raw = getattr(row, "rationale", None)
            if not isinstance(fname, str):
                continue
            if fname in score_map:
                duplicate_count += 1
                continue
            score = _parse_score_from_rationale(raw)
            if score is None:
                bad_score += 1
                continue
            score_map[fname] = score
        if bad_score:
            print(f"[warn] {path.name} has {bad_score} rows with invalid/missing rationale score.")

    if duplicate_count:
        print(f"[warn] LLM score inputs have {duplicate_count} duplicate file_name rows; keeping first.")
    return score_map


def is_review_subset_enabled(values) -> bool:
    return "on" in set(values or [])


def compute_review_mask(df: pd.DataFrame, llm_cutoff: int) -> pd.Series:
    centroid_flag_col = _score_columns_for_key("centroid_5")[0]
    mp10_flag_col = _score_columns_for_key("mp_10")[0]

    rule_centroid = (
        df[centroid_flag_col].fillna(False).astype(bool)
        if centroid_flag_col in df.columns
        else pd.Series(False, index=df.index)
    )
    rule_mp10 = (
        df[mp10_flag_col].fillna(False).astype(bool)
        if mp10_flag_col in df.columns
        else pd.Series(False, index=df.index)
    )
    rule_llm = (
        df["llm_score"].notna() & (df["llm_score"] <= llm_cutoff)
        if "llm_score" in df.columns
        else pd.Series(False, index=df.index)
    )
    return rule_centroid | rule_mp10 | rule_llm


def load_data(cache_dir: str, coords_file: str) -> pd.DataFrame:
    cache_root = Path(cache_dir)
    meta_path = cache_root / "metadata.csv"
    coords_path = cache_root / coords_file

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    if not coords_path.is_file():
        raise FileNotFoundError(f"Missing coords file: {coords_path}")

    metadata = pd.read_csv(meta_path)
    coords = np.load(coords_path)
    if coords.shape[0] != len(metadata):
        raise ValueError(
            f"Row mismatch: coords has {coords.shape[0]} rows, metadata has {len(metadata)} rows."
        )

    df = metadata.copy()
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    return df


def load_rationale_map(csv_path: Path) -> dict[str, dict[str, object]]:
    df = pd.read_csv(csv_path)
    if "file_name" not in df.columns or "rationale" not in df.columns:
        print(f"[warn] {csv_path.name} missing file_name or rationale; skipping.")
        return {}

    mapping: dict[str, dict[str, object]] = {}
    bad = 0
    for row in df.itertuples(index=False):
        fname = getattr(row, "file_name", None)
        raw = getattr(row, "rationale", None)
        if not isinstance(fname, str) or not isinstance(raw, str):
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            bad += 1
            continue
        if isinstance(obj, dict):
            mapping[fname] = {
                "score": obj.get("score", ""),
                "rationale": obj.get("rationale", ""),
            }
        else:
            bad += 1
    if bad:
        print(f"[warn] {bad} rationale rows in {csv_path.name} could not be parsed.")
    return mapping


def _value_or_empty(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def _bool_or_empty(value):
    parsed = _bool_or_none(value)
    if parsed is None:
        return ""
    return bool(parsed)


def load_prediction_map(mis_path: Path | None, correct_path: Path | None) -> dict[str, dict[str, object]]:
    pred_map: dict[str, dict[str, object]] = {}
    conflict_count = 0
    duplicate_count = 0

    for path in [mis_path, correct_path]:
        if path is None or not path.is_file():
            continue
        df = pd.read_csv(path)
        required = {"file_name", "predicted_label", "predicted_word_label"}
        missing = required - set(df.columns)
        if missing:
            print(f"[warn] {path.name} missing columns: {sorted(missing)}; skipping for predictions.")
            continue
        missing_optional = sorted(set(PREDICTION_COLUMNS) - set(df.columns))
        if missing_optional:
            print(f"[warn] {path.name} missing optional prediction columns: {missing_optional}.")

        for row in df.itertuples(index=False):
            fname = getattr(row, "file_name", None)
            if not isinstance(fname, str):
                continue
            entry = {col: _value_or_empty(getattr(row, col, "")) for col in PREDICTION_COLUMNS}
            if fname in pred_map:
                duplicate_count += 1
                current = pred_map[fname]
                if any(str(current.get(col, "")) != str(entry.get(col, "")) for col in PREDICTION_COLUMNS):
                    conflict_count += 1
                continue
            pred_map[fname] = entry

    if duplicate_count:
        print(f"[warn] prediction inputs have {duplicate_count} duplicate file_name rows; keeping first.")
    if conflict_count:
        print(f"[warn] prediction inputs have {conflict_count} conflicting duplicate predictions; kept first.")
    return pred_map


def build_file_row_lookup(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    if "file_name" not in df.columns:
        return {}
    dup_count = int(df["file_name"].duplicated().sum())
    if dup_count:
        print(f"[warn] merged dataframe has {dup_count} duplicate file_name rows; keeping first.")
    dedup = df.drop_duplicates(subset=["file_name"], keep="first")
    return dedup.set_index("file_name").to_dict(orient="index")


def build_export_row(
    file_name: str,
    row_lookup: dict[str, dict[str, object]],
    prediction_map: dict[str, dict[str, object]],
    reviewer_id: str,
) -> dict[str, object]:
    row = row_lookup.get(file_name, {})
    pred = prediction_map.get(file_name, {})

    export_row = {
        "file_name": file_name,
        "ground_truth_num_label": _value_or_empty(row.get("ground_truth_num_label", "")),
        "ground_truth_word_label": _value_or_empty(row.get("ground_truth_word_label", "")),
        "ground_truth_L2_num_label": _value_or_empty(row.get("ground_truth_L2_num_label", "")),
        "predicted_label": _value_or_empty(pred.get("predicted_label", "")),
        "predicted_word_label": _value_or_empty(pred.get("predicted_word_label", "")),
        "dataset": _value_or_empty(pred.get("dataset", "")),
        "top3_label_1": _value_or_empty(pred.get("top3_label_1", "")),
        "top3_prob_1": _value_or_empty(pred.get("top3_prob_1", "")),
        "top3_label_2": _value_or_empty(pred.get("top3_label_2", "")),
        "top3_prob_2": _value_or_empty(pred.get("top3_prob_2", "")),
        "top3_label_3": _value_or_empty(pred.get("top3_label_3", "")),
        "top3_prob_3": _value_or_empty(pred.get("top3_prob_3", "")),
        "split": _value_or_empty(pred.get("split", "")),
        "image_path": _value_or_empty(pred.get("image_path", "")),
        "rationale": _value_or_empty(pred.get("rationale", "")),
        "centroid_sim_to_centroid": _value_or_empty(row.get("centroid_5__sim_to_centroid", "")),
        "centroid_pct_rank_in_class": _value_or_empty(row.get("centroid_5__pct_rank_in_class", "")),
        "centroid_is_bottom_5pct": _bool_or_empty(row.get("centroid_5__flag")),
        "mp10_sim_to_centroid": _value_or_empty(row.get("mp_10__sim_to_centroid", "")),
        "mp10_pct_rank_in_class": _value_or_empty(row.get("mp_10__pct_rank_in_class", "")),
        "mp10_is_bottom_10pct": _bool_or_empty(row.get("mp_10__flag")),
        "mp10_margin_to_other_class": _value_or_empty(row.get(MP10_MARGIN_COLUMN, "")),
        "reviewer_id": reviewer_id or "unknown",
        "reviewed_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    return export_row


def load_existing_confirmed_rows(output_path: Path) -> dict[str, dict[str, object]]:
    if not output_path.is_file():
        return {}

    try:
        df = pd.read_csv(output_path)
    except Exception as exc:
        print(f"[warn] failed to read existing review output CSV '{output_path}': {exc}")
        return {}

    if "file_name" not in df.columns:
        print(f"[warn] existing review output CSV '{output_path}' missing file_name; ignoring preload.")
        return {}

    store: dict[str, dict[str, object]] = {}
    bad_rows = 0
    for row in df.to_dict(orient="records"):
        fname = row.get("file_name")
        if not isinstance(fname, str) or not fname:
            bad_rows += 1
            continue
        normalized = {}
        for col in EXPORT_COLUMNS:
            normalized[col] = _value_or_empty(row.get(col, ""))
        store[fname] = normalized

    if bad_rows:
        print(f"[warn] existing review output CSV '{output_path}' has {bad_rows} invalid rows; ignored.")
    return store


def save_confirmed_rows(store_dict: dict[str, dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(store_dict.values())
    df = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def load_score_source(csv_path: Path, source_key: str, bottom_col: str) -> pd.DataFrame:
    flag_col, sim_col, pct_col = _score_columns_for_key(source_key)
    output_columns = ["file_name", flag_col, sim_col, pct_col]
    include_margin = source_key == "mp_10"
    if include_margin:
        output_columns.append(MP10_MARGIN_COLUMN)
    df = pd.read_csv(csv_path)

    required = {"file_name", "sim_to_centroid", "pct_rank_in_class", bottom_col}
    if include_margin:
        required.add("margin_to_other_class")
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] {csv_path.name} missing columns: {sorted(missing)}; skipping.")
        return pd.DataFrame(columns=output_columns)

    dup_count = int(df["file_name"].duplicated().sum())
    if dup_count:
        print(f"[warn] {csv_path.name} has {dup_count} duplicate file_name rows; keeping first.")
        df = df.drop_duplicates(subset=["file_name"], keep="first")

    flag_series = df[bottom_col].map(_coerce_bool)
    invalid_mask = df[bottom_col].notna() & flag_series.isna()
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(f"[warn] {csv_path.name} has {invalid_count} invalid {bottom_col} values; coerced to False.")

    out = pd.DataFrame(
        {
            "file_name": df["file_name"],
            flag_col: flag_series.fillna(False).astype(bool),
            sim_col: pd.to_numeric(df["sim_to_centroid"], errors="coerce"),
            pct_col: pd.to_numeric(df["pct_rank_in_class"], errors="coerce"),
        }
    )
    if include_margin:
        out[MP10_MARGIN_COLUMN] = pd.to_numeric(df["margin_to_other_class"], errors="coerce")
    return out


def merge_score_sources(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df
    for spec in SCORE_SOURCE_SPECS:
        csv_arg = getattr(args, spec["arg"])
        csv_path = _resolve_csv_path(args.cache_dir, csv_arg)
        if csv_path and csv_path.is_file():
            source_df = load_score_source(csv_path, spec["key"], spec["bottom_col"])
            if not source_df.empty:
                out = out.merge(source_df, on="file_name", how="left")
        elif csv_arg:
            print(f"[warn] {spec['arg']} '{csv_path}' not found; skipping.")
    return out


def compute_overlay_df(df: pd.DataFrame, active_keys: list[str]) -> pd.DataFrame:
    if not active_keys:
        return pd.DataFrame(columns=list(df.columns) + ["active_flag_count", "badge_color", "badge_size"])

    counts = np.zeros(len(df), dtype=int)
    for key in active_keys:
        flag_col, _, _ = _score_columns_for_key(key)
        if flag_col not in df.columns:
            continue
        counts += df[flag_col].fillna(False).astype(bool).astype(int).to_numpy()

    mask = counts > 0
    if not mask.any():
        return pd.DataFrame(columns=list(df.columns) + ["active_flag_count", "badge_color", "badge_size"])

    overlay = df.loc[mask].copy()
    overlay["active_flag_count"] = counts[mask]
    max_bucket = max(BADGE_COLOR_BY_COUNT)
    overlay["badge_color"] = overlay["active_flag_count"].map(BADGE_COLOR_BY_COUNT).fillna(BADGE_COLOR_BY_COUNT[max_bucket])
    overlay["badge_size"] = overlay["active_flag_count"].map(BADGE_SIZE_BY_COUNT).fillna(BADGE_SIZE_BY_COUNT[max_bucket])
    return overlay


def build_scatter_figure(
    df: pd.DataFrame,
    color_col: str | None,
    hover_cols: list[str] | None,
    custom_cols: list[str],
    active_keys: list[str],
) -> go.Figure:
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        hover_data=hover_cols,
        custom_data=custom_cols,
        title="Embedding Viewer",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))

    if color_col:
        for trace in fig.data:
            if getattr(trace, "name", None) is not None:
                trace.legendgroup = str(trace.name)
        fig.update_layout(legend=dict(groupclick="togglegroup"))

    overlay = compute_overlay_df(df, active_keys)
    if overlay.empty:
        return fig

    if color_col:
        for class_name, class_df in overlay.groupby(color_col, dropna=False):
            if class_df.empty:
                continue
            legend_group = "(missing class)" if pd.isna(class_name) else str(class_name)
            fig.add_trace(
                go.Scattergl(
                    x=class_df["x"],
                    y=class_df["y"],
                    mode="markers",
                    name=f"Outlier badges ({legend_group})",
                    legendgroup=legend_group,
                    marker=dict(
                        symbol="circle-open",
                        color=class_df["badge_color"].tolist(),
                        size=class_df["badge_size"].tolist(),
                        line=dict(width=1.5),
                    ),
                    customdata=class_df[custom_cols].to_numpy(),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    else:
        fig.add_trace(
            go.Scattergl(
                x=overlay["x"],
                y=overlay["y"],
                mode="markers",
                name="Outlier badges (active)",
                marker=dict(
                    symbol="circle-open",
                    color=overlay["badge_color"].tolist(),
                    size=overlay["badge_size"].tolist(),
                    line=dict(width=1.5),
                ),
                customdata=overlay[custom_cols].to_numpy(),
                hoverinfo="skip",
                showlegend=True,
            )
        )
    return fig


def main() -> None:
    args = build_argparser().parse_args()
    df = load_data(args.cache_dir, args.coords_file)
    df = merge_score_sources(df, args)

    image_root_url = ""
    image_dir = ""
    if args.image_dir:
        if args.image_root:
            print("[warn] both image_dir and image_root set; using image_dir.")
        image_dir = str(Path(args.image_dir).resolve())
        if not Path(image_dir).is_dir():
            print(f"[warn] image_dir '{image_dir}' not found; image panel disabled.")
            image_dir = ""
        else:
            image_root_url = "/images"
    elif args.image_root:
        image_root_url = args.image_root

    mis_map: dict[str, dict[str, object]] = {}
    correct_map: dict[str, dict[str, object]] = {}
    mis_path = _resolve_csv_path(args.cache_dir, args.mis_csv)
    if mis_path and mis_path.is_file():
        mis_map = load_rationale_map(mis_path)
    elif args.mis_csv:
        print(f"[warn] mis_csv '{mis_path}' not found; skipping.")

    correct_path = _resolve_csv_path(args.cache_dir, args.correct_csv)
    if correct_path and correct_path.is_file():
        correct_map = load_rationale_map(correct_path)
    elif args.correct_csv:
        print(f"[warn] correct_csv '{correct_path}' not found; skipping.")
    prediction_map = load_prediction_map(mis_path, correct_path)

    llm_score_sources = []
    if mis_path and mis_path.is_file():
        llm_score_sources.append(mis_path)
    if correct_path and correct_path.is_file():
        llm_score_sources.append(correct_path)
    llm_score_map = load_llm_score_map(llm_score_sources)
    df["llm_score"] = df["file_name"].map(llm_score_map)

    hover_cols = [c.strip() for c in str(args.hover).split(",") if c.strip()]
    hover_cols = [c for c in hover_cols if c in df.columns]
    if not hover_cols:
        hover_cols = None

    color_col = args.color_by if args.color_by in df.columns else None
    if color_col is None and args.color_by:
        print(f"[warn] color_by column '{args.color_by}' not found; plotting without color.")

    if image_root_url:
        if "file_name" not in df.columns:
            print("[warn] file_name column not found; image panel disabled.")
            df["img_url"] = ""
        else:
            df["img_url"] = df["file_name"].apply(lambda x: _build_image_url(image_root_url, x))
    else:
        df["img_url"] = ""

    score_cols = score_custom_columns()
    if MP10_MARGIN_COLUMN not in df.columns:
        df[MP10_MARGIN_COLUMN] = np.nan
    custom_cols = ["img_url", "file_name", "ground_truth_word_label", "llm_score"] + score_cols
    for col in score_cols:
        if col not in df.columns:
            if col.endswith("__flag"):
                df[col] = False
            else:
                df[col] = np.nan

    custom_idx = {name: idx for idx, name in enumerate(custom_cols)}

    def _custom_get(custom_data, col_name: str, default=None):
        idx = custom_idx[col_name]
        if len(custom_data) <= idx:
            return default
        return custom_data[idx]

    initial_llm_cutoff = _resolve_llm_cutoff(args.default_llm_cutoff)
    initial_active = active_source_keys(DEFAULT_METHODS, THRESHOLD_STEPS[-1])
    initial_fig = build_scatter_figure(df, color_col, hover_cols, custom_cols, initial_active)
    row_lookup = build_file_row_lookup(df)

    review_output_path = Path(args.review_output_csv)
    initial_confirmed_store = load_existing_confirmed_rows(review_output_path)
    initial_status_store = {
        "kind": "info",
        "message": f"Loaded {len(initial_confirmed_store)} confirmed sample(s).",
    }

    app = Dash(__name__)
    if image_dir:

        @app.server.route("/images/<path:filename>")
        def _serve_image(filename: str):
            return send_from_directory(image_dir, filename)

    app.layout = html.Div(
        [
            dcc.Store(id="selected-file-store", data={}),
            dcc.Store(id="confirmed-outliers-store", data=initial_confirmed_store),
            dcc.Store(id="review-status-store", data=initial_status_store),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Outlier Controls", style={"fontWeight": "600", "marginBottom": "6px"}),
                            dcc.Checklist(
                                id="outlier-methods",
                                options=[
                                    {"label": "Centroid", "value": "centroid"},
                                    {"label": "Multi-prototype", "value": "multiprototype"},
                                ],
                                value=DEFAULT_METHODS,
                                inline=True,
                                inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                            ),
                            html.Div(
                                "Threshold (cumulative <= %)",
                                style={"fontSize": "12px", "marginTop": "8px", "marginBottom": "4px"},
                            ),
                            dcc.Slider(
                                id="outlier-threshold",
                                min=min(THRESHOLD_STEPS),
                                max=max(THRESHOLD_STEPS),
                                step=None,
                                value=THRESHOLD_STEPS[-1],
                                marks={step: f"{step}%" for step in THRESHOLD_STEPS},
                            ),
                            html.Hr(style={"margin": "10px 0"}),
                            dcc.Checklist(
                                id="review-subset-only",
                                options=[{"label": "Review subset only", "value": "on"}],
                                value=[],
                                inputStyle={"marginRight": "6px"},
                            ),
                            html.Div(
                                "LLM score cutoff (review rule)",
                                style={"fontSize": "12px", "marginTop": "8px", "marginBottom": "4px"},
                            ),
                            dcc.Slider(
                                id="review-llm-cutoff",
                                min=LLM_CUTOFF_MIN,
                                max=LLM_CUTOFF_MAX,
                                step=1,
                                value=initial_llm_cutoff,
                                marks={v: str(v) for v in range(LLM_CUTOFF_MIN, LLM_CUTOFF_MAX + 1)},
                            ),
                        ],
                        style={"padding": "10px 12px", "borderBottom": "1px solid #ddd"},
                    ),
                    dcc.Graph(id="embed-graph", figure=initial_fig, style={"height": "79vh"}),
                ],
                style={"flex": "3", "minWidth": "0"},
            ),
            html.Div(
                [
                    html.Div("Selected Image", style={"fontWeight": "600", "marginBottom": "8px"}),
                    html.Img(
                        id="img-view",
                        style={"width": "100%", "height": "auto", "border": "1px solid #eee"},
                    ),
                    html.Div(
                        [
                            html.Button("Mark True Outlier", id="mark-true-outlier-btn", n_clicks=0),
                            html.Button(
                                "Unmark",
                                id="unmark-true-outlier-btn",
                                n_clicks=0,
                                style={"marginLeft": "8px"},
                            ),
                        ],
                        style={"marginTop": "10px"},
                    ),
                    html.Div(id="review-status-text", style={"marginTop": "8px", "fontSize": "12px"}),
                    html.Div(id="img-caption", style={"marginTop": "8px", "fontSize": "12px", "color": "#555"}),
                ],
                style={"flex": "1", "borderLeft": "1px solid #ddd", "padding": "12px", "overflowY": "auto"},
            ),
        ],
        style={"display": "flex", "height": "100vh", "gap": "8px"},
    )

    @app.callback(
        Output("embed-graph", "figure"),
        Input("outlier-methods", "value"),
        Input("outlier-threshold", "value"),
        Input("review-subset-only", "value"),
        Input("review-llm-cutoff", "value"),
    )
    def _update_graph(selected_methods, threshold_value, review_subset_values, llm_cutoff_value):
        llm_cutoff = _resolve_llm_cutoff(llm_cutoff_value)
        review_only = is_review_subset_enabled(review_subset_values)
        if review_only:
            review_mask = compute_review_mask(df, llm_cutoff)
            df_plot = df.loc[review_mask].copy()
        else:
            df_plot = df
        active_keys = active_source_keys(selected_methods, threshold_value)
        return build_scatter_figure(df_plot, color_col, hover_cols, custom_cols, active_keys)

    @app.callback(
        Output("selected-file-store", "data"),
        Input("embed-graph", "clickData"),
        State("selected-file-store", "data"),
    )
    def _update_selected_file(click_data, current_selected):
        if not click_data or "points" not in click_data or not click_data["points"]:
            return no_update
        custom = click_data["points"][0].get("customdata", [])
        fname = _custom_get(custom, "file_name", "")
        if not fname:
            return no_update
        if current_selected and current_selected.get("file_name") == fname:
            return no_update
        return {"file_name": fname}

    @app.callback(
        Output("confirmed-outliers-store", "data"),
        Output("review-status-store", "data"),
        Input("mark-true-outlier-btn", "n_clicks"),
        Input("unmark-true-outlier-btn", "n_clicks"),
        State("selected-file-store", "data"),
        State("confirmed-outliers-store", "data"),
    )
    def _update_confirmed_store(mark_clicks, unmark_clicks, selected_file_data, confirmed_store_data):
        trigger = ctx.triggered_id
        if trigger not in {"mark-true-outlier-btn", "unmark-true-outlier-btn"}:
            return no_update, no_update

        selected_file = (selected_file_data or {}).get("file_name", "")
        if not selected_file:
            return no_update, {"kind": "error", "message": "No selected sample. Click a point first."}

        current_store = dict(confirmed_store_data or {})
        action_message = ""

        if trigger == "mark-true-outlier-btn":
            export_row = build_export_row(selected_file, row_lookup, prediction_map, args.reviewer_id)
            current_store[selected_file] = export_row
            action_message = f"Marked {selected_file} as true outlier."
        else:
            if selected_file in current_store:
                del current_store[selected_file]
                action_message = f"Unmarked {selected_file}."
            else:
                action_message = f"{selected_file} was not marked."

        try:
            save_confirmed_rows(current_store, review_output_path)
        except Exception as exc:
            return no_update, {"kind": "error", "message": f"Failed to save CSV: {exc}"}

        return current_store, {
            "kind": "success",
            "message": f"{action_message} Saved {len(current_store)} confirmed sample(s).",
        }

    @app.callback(
        Output("review-status-text", "children"),
        Input("review-status-store", "data"),
    )
    def _render_review_status(status_data):
        status = status_data or {}
        kind = str(status.get("kind", "info")).lower()
        message = str(status.get("message", ""))
        prefix = {"success": "[ok]", "error": "[error]", "info": "[info]"}.get(kind, "[info]")
        if not message:
            return ""
        return f"{prefix} {message}"

    @app.callback(
        Output("img-view", "src"),
        Output("img-caption", "children"),
        Input("selected-file-store", "data"),
        Input("outlier-methods", "value"),
        Input("outlier-threshold", "value"),
        Input("review-subset-only", "value"),
        Input("review-llm-cutoff", "value"),
        Input("confirmed-outliers-store", "data"),
        Input("review-status-store", "data"),
    )
    def _render_panel(
        selected_file_data,
        selected_methods,
        threshold_value,
        review_subset_values,
        llm_cutoff_value,
        confirmed_store_data,
        review_status_data,
    ):
        selected_file = (selected_file_data or {}).get("file_name", "")
        if not selected_file:
            return "", "Click a point to load image."

        row = row_lookup.get(selected_file, {})
        img_url = _value_or_empty(row.get("img_url", ""))
        gt_word = _value_or_empty(row.get("ground_truth_word_label", ""))
        llm_score = row.get("llm_score", None)
        llm_cutoff = _resolve_llm_cutoff(llm_cutoff_value)

        mis_entry = mis_map.get(selected_file)
        correct_entry = correct_map.get(selected_file)
        if mis_entry and correct_entry:
            prediction = "ambiguous (found in mis + correct)"
            entry = mis_entry
        elif mis_entry:
            prediction = "misclassification"
            entry = mis_entry
        elif correct_entry:
            prediction = "correct classification"
            entry = correct_entry
        else:
            prediction = "unknown (not found)"
            entry = None

        def _format_entry(score_entry: dict[str, object] | None):
            if score_entry is None:
                return [
                    html.Div("LLM score: (not found)"),
                    html.Div("LLM rationale: (not found)"),
                ]
            score = score_entry.get("score", "")
            rationale = score_entry.get("rationale", "")
            return [
                html.Div(f"LLM score: {score}" if score != "" else "LLM score: (missing)"),
                html.Div(
                    f"LLM rationale: {rationale}" if rationale != "" else "LLM rationale: (missing)",
                    style={"whiteSpace": "pre-wrap"},
                ),
            ]

        active_keys = active_source_keys(selected_methods, threshold_value)
        active_labels: list[str] = []
        for key in active_keys:
            flag_col, _, _ = _score_columns_for_key(key)
            if _bool_or_none(row.get(flag_col)) is True:
                active_labels.append(SCORE_SPEC_BY_KEY[key]["label"])

        active_flag_summary = f"Active outlier flags: {len(active_labels)} / {len(active_keys)}"
        flagged_by_summary = "Flagged by: " + (", ".join(active_labels) if active_labels else "(none)")

        centroid_flag_col = _score_columns_for_key("centroid_5")[0]
        mp10_flag_col = _score_columns_for_key("mp_10")[0]
        rule_centroid = _bool_or_none(row.get(centroid_flag_col)) is True
        rule_mp10 = _bool_or_none(row.get(mp10_flag_col)) is True
        rule_llm = False
        try:
            if llm_score is not None and not pd.isna(llm_score):
                rule_llm = float(llm_score) <= llm_cutoff
        except Exception:
            rule_llm = False
        review_match = rule_centroid or rule_mp10 or rule_llm
        review_rule_labels = []
        if rule_centroid:
            review_rule_labels.append("centroid_bottom5")
        if rule_mp10:
            review_rule_labels.append("mp10_bottom10")
        if rule_llm:
            review_rule_labels.append(f"llm<={llm_cutoff}")
        review_rules_line = "Review rules active: " + (", ".join(review_rule_labels) if review_rule_labels else "(none)")
        review_mode_line = "Review subset only: " + ("on" if is_review_subset_enabled(review_subset_values) else "off")

        confirmed_store = dict(confirmed_store_data or {})
        confirmed_count = len(confirmed_store)
        is_confirmed = selected_file in confirmed_store
        review_status_message = str((review_status_data or {}).get("message", ""))

        table_rows = []
        td_style = {"borderBottom": "1px solid #e6e6e6", "padding": "4px 6px", "verticalAlign": "top"}
        for spec in SCORE_SOURCE_SPECS:
            flag_col, sim_col, pct_col = _score_columns_for_key(spec["key"])
            flag_value = row.get(flag_col)
            row_style = {"backgroundColor": "#fff8e6"} if _bool_or_none(flag_value) is True else {}
            table_rows.append(
                html.Tr(
                    [
                        html.Td(spec["label"], style=td_style),
                        html.Td("Yes" if spec.get("used_in_review_rule") else "No", style=td_style),
                        html.Td(_fmt_bool(flag_value), style=td_style),
                        html.Td(_fmt_num(row.get(sim_col), digits=6), style=td_style),
                        html.Td(_fmt_pct_rank(row.get(pct_col)), style=td_style),
                    ],
                    style=row_style,
                )
            )

        table = html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Source", style={"textAlign": "left", "padding": "4px 6px"}),
                            html.Th("Used in Review Rule", style={"textAlign": "left", "padding": "4px 6px"}),
                            html.Th("Bottom Flag", style={"textAlign": "left", "padding": "4px 6px"}),
                            html.Th("sim_to_centroid", style={"textAlign": "left", "padding": "4px 6px"}),
                            html.Th("pct_rank_in_class", style={"textAlign": "left", "padding": "4px 6px"}),
                        ]
                    )
                ),
                html.Tbody(table_rows),
            ],
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px", "marginTop": "6px"},
        )

        caption = [
            html.Div(f"file_name: {selected_file}" if selected_file else "file_name: (missing)"),
            html.Div(
                f"ground_truth_word_label: {gt_word}"
                if gt_word != ""
                else "ground_truth_word_label: (missing)"
            ),
            html.Div(f"prediction: {prediction}"),
            *_format_entry(entry),
            html.Hr(),
            html.Div(active_flag_summary, style={"fontWeight": "600"}),
            html.Div(flagged_by_summary),
            html.Div(f"Review subset match: {'True' if review_match else 'False'}"),
            html.Div(review_rules_line),
            html.Div(f"Current LLM cutoff: {llm_cutoff}"),
            html.Div(review_mode_line),
            html.Div(f"Confirmed true outlier: {'Yes' if is_confirmed else 'No'}"),
            html.Div(f"Current confirmed count: {confirmed_count}"),
            html.Div(f"Last save status: {review_status_message or '(none)'}"),
            html.Hr(),
            table,
        ]
        return img_url, caption

    print(f"[dash] running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

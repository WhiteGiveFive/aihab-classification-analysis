from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = APP_DIR / "CS_Xplots_2019_2023_test"
DEFAULT_CORRECT_CSV = APP_DIR / "correctly_classified_samples_test.csv"
DEFAULT_MIS_CSV = APP_DIR / "misclassified_samples_test.csv"

st.set_page_config(page_title="AIHAB Classification Gallery", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(path_str: str, label: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_csv(path)
    df["classification"] = label
    return df


def unique_options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    values = df[column].dropna().astype(str).unique().tolist()
    return sorted(values)


def apply_filters(
    df: pd.DataFrame,
    classifications: Iterable[str],
    datasets: Iterable[str],
    gt_labels: Iterable[str],
    pred_labels: Iterable[str],
    filename_query: str,
) -> pd.DataFrame:
    filtered = df
    if classifications:
        filtered = filtered[filtered["classification"].isin(classifications)]
    if datasets and "dataset" in filtered.columns:
        filtered = filtered[filtered["dataset"].isin(datasets)]
    if gt_labels and "ground_truth_word_label" in filtered.columns:
        filtered = filtered[filtered["ground_truth_word_label"].isin(gt_labels)]
    if pred_labels and "predicted_word_label" in filtered.columns:
        filtered = filtered[filtered["predicted_word_label"].isin(pred_labels)]
    if filename_query:
        filtered = filtered[filtered["file_name"].str.contains(filename_query, case=False, na=False)]
    return filtered


def build_caption(row: pd.Series) -> str:
    gt = row.get("ground_truth_word_label", "")
    pred = row.get("predicted_word_label", "")
    return f"{row['file_name']} | GT: {gt} | Pred: {pred} | {row['classification']}"


def load_image(path: Path, max_width: int | None) -> Image.Image | None:
    if not path.exists():
        return None
    image = Image.open(path)
    if max_width:
        image.thumbnail((max_width, max_width))
    return image


st.title("AIHAB Classification Gallery")
st.caption("Local, self-hosted viewer for correct vs misclassified samples.")

with st.sidebar.expander("Paths", expanded=False):
    dataset_dir = Path(st.text_input("Dataset folder", str(DEFAULT_DATASET_DIR))).expanduser()
    correct_csv = Path(st.text_input("Correct CSV", str(DEFAULT_CORRECT_CSV))).expanduser()
    mis_csv = Path(st.text_input("Mis CSV", str(DEFAULT_MIS_CSV))).expanduser()

if not correct_csv.exists() or not mis_csv.exists():
    st.error("CSV files not found. Check the Paths section in the sidebar.")
    st.stop()

correct_df = load_csv(str(correct_csv), "correct", correct_csv.stat().st_mtime)
mis_df = load_csv(str(mis_csv), "mis", mis_csv.stat().st_mtime)

combined = pd.concat([correct_df, mis_df], ignore_index=True)

st.subheader("Filters")

classification_options = ["correct", "mis"]
dataset_options = unique_options(combined, "dataset")
gt_options = unique_options(combined, "ground_truth_word_label")
pred_options = unique_options(combined, "predicted_word_label")

with st.sidebar:
    classifications = st.multiselect("Classification", classification_options, default=classification_options)
    datasets = st.multiselect("Dataset", dataset_options, default=dataset_options)
    gt_labels = st.multiselect("Ground truth label", gt_options, default=[])
    pred_labels = st.multiselect("Predicted label", pred_options, default=[])
    filename_query = st.text_input("Filename contains")
    sort_by = st.selectbox("Sort by", ["file_name", "ground_truth_word_label", "predicted_word_label"])
    sort_ascending = st.radio("Sort order", ["asc", "desc"], horizontal=True) == "asc"

    st.markdown("---")
    page_size = st.selectbox("Images per page", [24, 48, 96, 120], index=0)
    columns = st.slider("Gallery columns", min_value=2, max_value=6, value=4)
    max_width = st.slider("Max image width (px)", min_value=256, max_value=1024, value=512, step=64)
    show_table = st.checkbox("Show filtered table", value=False)

filtered = apply_filters(
    combined,
    classifications,
    datasets,
    gt_labels,
    pred_labels,
    filename_query,
)

if sort_by in filtered.columns:
    filtered = filtered.sort_values(sort_by, ascending=sort_ascending, na_position="last")

count_correct = int((filtered["classification"] == "correct").sum())
count_mis = int((filtered["classification"] == "mis").sum())

metrics_cols = st.columns(3)
metrics_cols[0].metric("Filtered total", len(filtered))
metrics_cols[1].metric("Correct", count_correct)
metrics_cols[2].metric("Misclassified", count_mis)

if filtered.empty:
    st.info("No samples match the current filters.")
    st.stop()

max_page = max(1, math.ceil(len(filtered) / page_size))
page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)
start = (page - 1) * page_size
end = start + page_size
page_df = filtered.iloc[start:end]

st.write(f"Showing {start + 1}-{min(end, len(filtered))} of {len(filtered)}")

rows = list(page_df.iterrows())
columns_list = st.columns(columns)

for idx, (_, row) in enumerate(rows):
    col = columns_list[idx % columns]
    img_path = dataset_dir / row["file_name"]
    image = load_image(img_path, max_width)
    if image is None:
        col.warning(f"Missing: {row['file_name']}")
        continue
    col.image(image, caption=build_caption(row), use_container_width=True)

if show_table:
    st.dataframe(page_df, use_container_width=True)

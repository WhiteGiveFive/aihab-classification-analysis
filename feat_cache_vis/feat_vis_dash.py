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
from dash import Dash, dcc, html, Input, Output
from flask import send_from_directory


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
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dash host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash port.")
    return parser


def _build_image_url(root: str, fname: str) -> str:
    if not root:
        return ""
    root = str(root).rstrip("/")
    return f"{root}/{fname}"


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


def _resolve_csv_path(cache_dir: str, csv_path: str) -> Path | None:
    if not csv_path:
        return None
    path = Path(csv_path)
    if not path.is_absolute():
        path = Path(cache_dir) / csv_path
    return path


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


def load_centroid_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"file_name", "sim_to_centroid", "pct_rank_in_class", "is_bottom_5pct"}
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] {csv_path.name} missing columns: {sorted(missing)}; skipping.")
        return pd.DataFrame(columns=["file_name", "sim_to_centroid", "pct_rank_in_class", "is_bottom_5pct"])
    return df[list(required)]


def main() -> None:
    args = build_argparser().parse_args()
    df = load_data(args.cache_dir, args.coords_file)

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

    centroid_path = _resolve_csv_path(args.cache_dir, args.centroid_csv)
    if centroid_path and centroid_path.is_file():
        centroid_df = load_centroid_scores(centroid_path)
        if not centroid_df.empty:
            df = df.merge(centroid_df, on="file_name", how="left")
    elif args.centroid_csv:
        print(f"[warn] centroid_csv '{centroid_path}' not found; skipping.")

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

    custom_cols = [
        "img_url",
        "file_name",
        "ground_truth_word_label",
        "sim_to_centroid",
        "pct_rank_in_class",
        "is_bottom_5pct",
    ]
    for col in custom_cols:
        if col not in df.columns:
            df[col] = False if col == "is_bottom_5pct" else np.nan

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

    if "is_bottom_5pct" in df.columns:
        bottom_mask = df["is_bottom_5pct"] == True
        df_bottom = df[bottom_mask]
        if not df_bottom.empty:
            if color_col:
                for trace in fig.data:
                    if getattr(trace, "name", None) is not None:
                        trace.legendgroup = str(trace.name)
                for class_name, class_df in df_bottom.groupby(color_col, dropna=False):
                    if class_df.empty:
                        continue
                    legend_group = "(missing class)" if pd.isna(class_name) else str(class_name)
                    fig.add_trace(
                        go.Scattergl(
                            x=class_df["x"],
                            y=class_df["y"],
                            mode="markers",
                            name=f"Bottom 5% ({legend_group})",
                            legendgroup=legend_group,
                            marker=dict(
                                size=10,
                                symbol="circle-open",
                                # For open symbols, Plotly uses marker.color for the outline.
                                color="#FF5500",
                            ),
                            customdata=class_df[custom_cols].to_numpy(),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                fig.update_layout(legend=dict(groupclick="togglegroup"))
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=df_bottom["x"],
                        y=df_bottom["y"],
                        mode="markers",
                        name="Bottom 5% (centroid)",
                        marker=dict(
                            size=10,
                            symbol="circle-open",
                            # For open symbols, Plotly uses marker.color for the outline.
                            color="#FF5500",
                        ),
                        customdata=df_bottom[custom_cols].to_numpy(),
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )

    app = Dash(__name__)
    if image_dir:
        @app.server.route("/images/<path:filename>")
        def _serve_image(filename: str):
            return send_from_directory(image_dir, filename)
    app.layout = html.Div(
        [
            html.Div(
                dcc.Graph(id="embed-graph", figure=fig, style={"height": "85vh"}),
                style={"flex": "3", "minWidth": "0"},
            ),
            html.Div(
                [
                    html.Div("Selected Image", style={"fontWeight": "600", "marginBottom": "8px"}),
                    html.Img(
                        id="img-view",
                        style={"width": "100%", "height": "auto", "border": "1px solid #eee"},
                    ),
                    html.Div(id="img-caption", style={"marginTop": "8px", "fontSize": "12px", "color": "#555"}),
                ],
                style={"flex": "1", "borderLeft": "1px solid #ddd", "padding": "12px"},
            ),
        ],
        style={"display": "flex", "height": "100vh", "gap": "8px"},
    )

    @app.callback(
        Output("img-view", "src"),
        Output("img-caption", "children"),
        Input("embed-graph", "clickData"),
    )
    def _on_click(click_data):
        if not click_data or "points" not in click_data or not click_data["points"]:
            return "", "Click a point to load image."
        custom = click_data["points"][0].get("customdata", [])
        img_url = custom[0] if len(custom) > 0 else ""
        fname = custom[1] if len(custom) > 1 else ""
        gt_word = custom[2] if len(custom) > 2 else ""
        sim_to_centroid = custom[3] if len(custom) > 3 else None
        pct_rank = custom[4] if len(custom) > 4 else None
        mis_entry = mis_map.get(fname)
        correct_entry = correct_map.get(fname)
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

        def _format_entry(entry: dict[str, object] | None):
            if entry is None:
                return [
                    html.Div("LLM score: (not found)"),
                    html.Div("LLM rationale: (not found)"),
                ]
            score = entry.get("score", "")
            rationale = entry.get("rationale", "")
            return [
                html.Div(
                    f"LLM score: {score}"
                    if score != ""
                    else "LLM score: (missing)"
                ),
                html.Div(
                    f"LLM rationale: {rationale}"
                    if rationale != ""
                    else "LLM rationale: (missing)",
                    style={"whiteSpace": "pre-wrap"},
                ),
            ]

        def _fmt_num(val, digits: int = 6) -> str:
            if val is None:
                return "(missing)"
            try:
                if pd.isna(val):
                    return "(missing)"
            except Exception:
                pass
            if isinstance(val, (int, np.integer)):
                return str(val)
            if isinstance(val, (float, np.floating)):
                return f"{val:.{digits}f}"
            return str(val)

        pct_rank_str = _fmt_num(pct_rank, digits=6)
        if pct_rank_str not in ("(missing)",):
            try:
                pct_val = float(pct_rank)
                pct_rank_str = f"{pct_rank_str} ({pct_val * 100:.3f}%)"
            except Exception:
                pass

        caption = [
            html.Div(f"file_name: {fname}" if fname else "file_name: (missing)"),
            html.Div(
                f"ground_truth_word_label: {gt_word}"
                if gt_word != ""
                else "ground_truth_word_label: (missing)"
            ),
            html.Div(f"prediction: {prediction}"),
            *_format_entry(entry),
            html.Div(f"sim_to_centroid: {_fmt_num(sim_to_centroid, digits=6)}"),
            html.Div(f"pct_rank_in_class: {pct_rank_str}"),
        ]
        return img_url, caption

    print(f"[dash] running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

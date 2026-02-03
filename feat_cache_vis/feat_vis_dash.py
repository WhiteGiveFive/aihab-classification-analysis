"""
Dash app for interactive embedding visualization with image panel.

Usage example:
  python feat_cache_vis/feat_vis_dash.py \
    --cache_dir feat_cache_vis/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_cs/test/seed1 \
    --coords_file vis_umap_coords.npy \
    --image_dir CS_Xplots_2019_2023_test \
    --port 8050
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import plotly.express as px
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

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        hover_data=hover_cols,
        custom_data=["img_url", "file_name"],
        title="Embedding Viewer",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))

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
        return img_url, fname

    print(f"[dash] running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

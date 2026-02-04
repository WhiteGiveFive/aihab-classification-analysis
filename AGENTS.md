# Repository Guidelines

This project implements tools that analyse and visualise habitat data.

## Project Structure & Module Organization
- `CS_Xplots_2019_2023_test/` holds the primary dataset (JPEG images, two CSVs, and a `.ipynb_checkpoints/` directory).
- The repository currently has no source-code or test directories; contributions are primarily data updates.
- Keep assets in place; downstream tooling expects stable paths and filenames.

## App Development Principles (Self-Hosted, Restricted Data)
- Keep all data local; the app must be self-hosted and should not upload images or CSVs to third-party services.
- Optimize for large datasets (3GB+): lazy-load images, paginate results, and cache CSV parsing.
- Favor Python-first tooling; default to Streamlit unless stronger auth or UI needs justify Dash/FastAPI.
- Plan for multi-model comparisons: treat each model’s CSV outputs as separate sources and support set operations (e.g., correct-by-all, misclassified-by-all).
- Preserve stable paths and filenames to avoid breaking downstream tools and data links.

## Build, Test, and Development Commands
- No build or test scripts are defined in this repository.
- If you add analysis code or notebooks, document the exact commands and dependencies here (e.g., `python scripts/analyze.py`, `jupyter lab`).

## Coding Style & Naming Conventions
- Preserve existing filenames for images and CSVs; do not rename existing data.
- Image naming pattern: `ATT<id>_<plot>_<year>_photo<index>-<timestamp>[suffix].jpg`  
  Example: `ATT1000_642X1_2021_photo1-20210817-104314.jpg`
- CSV naming pattern: descriptive and date-stamped, e.g., `CS_Xplots_2019_23_NEW02OCT24.csv`.
- If you introduce code, follow standard conventions for the chosen language (e.g., Python 4-space indentation) and note any formatter or linter you adopt.

## Testing Guidelines
- No automated tests are included.
- If you add code, include a minimal validation or test step and document how to run it.

## Commit & Pull Request Guidelines
- Git history is empty; use short, imperative commit summaries that describe the data change (e.g., “Add ATT1200 plot images”).
- In pull requests, include:
  - A brief description of what changed and why.
  - The data source and any transformations or filters applied.
  - Any manual validation or checks performed.

## Data Handling & Integrity
- Avoid re-encoding images unless required to prevent quality loss and repo bloat.
- Keep CSVs and image sets in sync; note any deltas or exclusions.
- Do not add transient artifacts (e.g., new `.ipynb_checkpoints/` contents) unless required for reproducibility.

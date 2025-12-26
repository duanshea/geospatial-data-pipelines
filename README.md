# geospatial-data-pipelines

Small, practical geospatial data pipeline exercises: **ingest → transform → analyze → export**.

This repo focuses on building **reproducible** geospatial ETL workflows with clean code structure, clear inputs/outputs, and short design notes.

---

## What’s inside

Typical pipelines in this repo may include:
- Data ingestion (local files / public sources / APIs)
- Geo-referencing & alignment (optional)
- Feature extraction / index computation
- Time-series analysis & change detection (optional)
- QA/QC + reporting
- Export to standard formats (GeoTIFF / GeoJSON / CSV)

> **Note on data:** Large or licensed datasets are **not** included. See [Data access](#data-access) for how to provide your own inputs.

---

## Repository structure

- `data/` — inputs (empty by default; use your own data or add links)
- `notebooks/` — exploration and prototyping
- `src/` — reusable code (pipelines, utilities, core modules)
- `outputs/` — generated artifacts (ignored by git if large)
- `docs/` — short write-ups (design choices, experiment notes, figures)

Recommended conventions:
- Each exercise/pipeline should have a **single entry point** (script or notebook)
- Each pipeline writes outputs into its own subfolder under `outputs/`

---

## Quick start

### 1) Create the environment

Option A (conda, recommended):
```bash
conda env create -f environment.yml
conda activate geo-pipelines

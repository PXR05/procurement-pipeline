# Procurement analysis

Pipeline for Indonesian IT procurement: data cleaning, K-Means clustering, network analysis, and BPK PDF extraction.

Requires uv and JDK 17+. Set `JAVA_HOME`, then run from this folder:

```sh
uv sync --locked
uv run --locked python scripts/reproduce.py --raw-archive "/path/to/opentender-2023.jsonl.gz" --context-dir "/path/to/context"
```

The context folder needs `apbd_fiscal_context_2023.csv`, `lkpp_open_data_aggregates_2023.csv`, and either `bpk_audit_findings_2023.json` or PDFs indexed by `bpk_sources.json`. Use [the PDF index example](examples/bpk_sources.json); page numbers start at 1. Scanned PDFs need OCR first. Results are written to `data/` and `output/`.

For Docker, place the archive at `inputs/opentender-2023.jsonl.gz` and context files at `inputs/context/`:

```sh
docker compose up -d --wait namenode datanode
docker compose run --build --rm pipeline
```

Optional database and Airflow services:

```sh
docker compose --profile serving up -d --wait
docker compose --profile orchestration up -d --build
```

Copy `.env.example` to `.env` to change paths, ports, or local passwords. Trigger the `procurement` DAG in Airflow at localhost:8080. The serving profile starts empty databases; the dashboard uses its existing snapshot.

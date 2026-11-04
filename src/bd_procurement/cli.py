import json
import shutil
from pathlib import Path

import typer

from bd_procurement.analytics.graph import export_graphml
from bd_procurement.analytics.quality import profile_procurement_quality
from bd_procurement.config import (
    BRONZE_OPENTENDER_DIR,
    CHARTS_DIR,
    MANIFESTS_DIR,
    OUTPUT_DIR,
    SILVER_DIR,
)
from bd_procurement.ingestion.bpk import extract_bpk_sources
from bd_procurement.ingestion.context import prepare_context
from bd_procurement.ingestion.manifest import create_manifest
from bd_procurement.processing.gold_builder import build_gold_marts
from bd_procurement.processing.silver_builder import (
    load_silver_from_parquet,
    process_bronze_to_silver,
)
from bd_procurement.visualizer.charts import (
    plot_hhi_distribution,
    plot_it_subcategories,
    plot_network_subgraph,
    plot_seasonality,
    plot_top_suppliers,
)

app = typer.Typer(help="Procurement analysis.")


@app.command(help="Extract review candidates from indexed BPK PDF pages.")
def extract_bpk(
    sources: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, help="bpk_sources.json file."
    ),
    output: Path = typer.Option(..., help="Destination JSON file."),
):
    records = extract_bpk_sources(sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Extracted {len(records)} records: {output}")


@app.command(help="Import the 2023 archive and context files.")
def ingest(
    raw_archive: Path | None = typer.Option(None, help="2023 OCDS JSONL.GZ file."),
    context_dir: Path | None = typer.Option(
        None, help="Folder containing the three context files."
    ),
    sample_context: bool = typer.Option(False, help="Use generated context samples."),
):
    if (context_dir is None) == (not sample_context):
        raise typer.BadParameter("Choose --context-dir or --sample-context.")
    archive_path = (
        raw_archive
        if raw_archive is not None
        else BRONZE_OPENTENDER_DIR / "opentender-2023.jsonl.gz"
    )
    if not archive_path.is_file():
        raise typer.BadParameter(
            f"Archive not found: {archive_path}", param_hint="--raw-archive"
        )
    try:
        context_paths = prepare_context(context_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--context-dir") from exc
    BRONZE_OPENTENDER_DIR.mkdir(parents=True, exist_ok=True)
    target = BRONZE_OPENTENDER_DIR / "opentender-2023.jsonl.gz"
    if archive_path.resolve() != target.resolve():
        shutil.copy2(archive_path, target)
    create_manifest(
        source_id="opentender_ocds",
        source_url="https://data.open-contracting.org/en/publication/101",
        filepath=target,
        period="2023",
        format_type="JSONL.GZ",
        license_type="ODbL",
        output_dir=MANIFESTS_DIR,
    )
    for path in context_paths:
        create_manifest(
            source_id=path.parent.name,
            source_url="user-provided"
            if context_dir is not None
            else "generated-sample",
            filepath=path,
            period="2023",
            format_type=path.suffix.lstrip(".").upper(),
            license_type="User supplied" if context_dir is not None else "Sample data",
            extra_metadata={"sample_data": context_dir is None},
            output_dir=MANIFESTS_DIR,
        )
    print(
        f"Imported {target.name}; context: {'supplied' if context_dir is not None else 'sample'}."
    )


@app.command(help="Build tables, graphs, and charts.")
def process(
    max_records: int | None = typer.Option(None, min=1, help="Optional record limit."),
    recompute_silver: bool = typer.Option(
        True, help="Rebuild tables from the archive."
    ),
):
    archive = BRONZE_OPENTENDER_DIR / "opentender-2023.jsonl.gz"
    if (
        not recompute_silver
        and max_records is None
        and (SILVER_DIR / "fact_tender.parquet").exists()
    ):
        silver = load_silver_from_parquet()
    else:
        silver = process_bronze_to_silver(archive, max_records=max_records)
    print(
        f"Processed {silver['processed_records']:,} records; quarantined {silver['quarantined_records']:,}."
    )
    gold = build_gold_marts(silver)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "mart_buyer_concentration",
        "mart_supplier_dominance",
        "mart_review_priority",
    ):
        (OUTPUT_DIR / f"{name}.json").write_text(
            json.dumps(gold[name], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    (OUTPUT_DIR / "graph_analytics_summary.json").write_text(
        json.dumps(gold["graph_analytics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    export_graphml(gold["projected_graph"], OUTPUT_DIR / "procurement_network.graphml")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_hhi_distribution(gold["mart_buyer_concentration"])
    plot_top_suppliers(gold["mart_supplier_dominance"])
    plot_seasonality(silver["fact_tenders"])
    plot_it_subcategories(silver["fact_tenders"])
    plot_network_subgraph(gold["projected_graph"])
    quality = profile_procurement_quality(
        silver["fact_tenders"],
        silver["fact_awards"],
        silver["bridge_award_suppliers"],
        silver["dim_organizations"],
        quarantine_count=silver["quarantined_records"],
    )
    (OUTPUT_DIR / "data_quality_scorecard.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    print(f"Results: {OUTPUT_DIR}")


@app.command(help="Import data and rebuild the analysis.")
def run_all(
    raw_archive: Path | None = typer.Option(None, help="2023 OCDS JSONL.GZ file."),
    context_dir: Path | None = typer.Option(
        None, help="Folder containing the three context files."
    ),
    sample_context: bool = typer.Option(False, help="Use generated context samples."),
    max_records: int | None = typer.Option(None, min=1, help="Optional record limit."),
    recompute_silver: bool = typer.Option(
        True, help="Rebuild tables from the archive."
    ),
):
    ingest(
        raw_archive=raw_archive, context_dir=context_dir, sample_context=sample_context
    )
    process(max_records=max_records, recompute_silver=recompute_silver)


if __name__ == "__main__":
    app()

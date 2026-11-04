import json
import shutil
from pathlib import Path

from bd_procurement.config import RAW_DIR
from bd_procurement.ingestion.bpk import (
    extract_bpk_sources,
    generate_sample_bpk_audit_findings,
)
from bd_procurement.ingestion.djpk import (
    generate_sample_djpk_fiscal_data,
    load_djpk_fiscal_data,
)
from bd_procurement.ingestion.lkpp import (
    generate_sample_lkpp_aggregates,
    load_lkpp_aggregates,
)

CONTEXT_FILES = {
    "djpk": ("apbd_fiscal_context_2023.csv", generate_sample_djpk_fiscal_data),
    "bpk": ("bpk_audit_findings_2023.json", generate_sample_bpk_audit_findings),
    "lkpp": ("lkpp_open_data_aggregates_2023.csv", generate_sample_lkpp_aggregates),
}


def prepare_context(context_dir: Path | None = None) -> list[Path]:
    bpk_from_pdf = (
        context_dir is not None and (context_dir / "bpk_sources.json").is_file()
    )
    bpk = None
    if context_dir is not None:
        missing = [
            name
            for name, _ in CONTEXT_FILES.values()
            if not (name == CONTEXT_FILES["bpk"][0] and bpk_from_pdf)
            if not (context_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing context files in {context_dir}: {', '.join(missing)}"
            )
        try:
            djpk = load_djpk_fiscal_data(context_dir / CONTEXT_FILES["djpk"][0])
            lkpp = load_lkpp_aggregates(context_dir / CONTEXT_FILES["lkpp"][0])
            bpk = (
                extract_bpk_sources(context_dir / "bpk_sources.json")
                if bpk_from_pdf
                else json.loads(
                    (context_dir / CONTEXT_FILES["bpk"][0]).read_text(
                        encoding="utf-8-sig"
                    )
                )
            )
            if not djpk or not lkpp or not isinstance(bpk, list):
                raise ValueError("Expected DJPK/LKPP rows and a BPK JSON array.")
            for row in bpk:
                for field in (
                    "report_id",
                    "entity_name",
                    "page_number",
                    "issue_type",
                    "quote",
                ):
                    if field not in row:
                        raise ValueError(f"BPK record is missing {field}.")
            for resource in ("LKPP-AGG-2023-NASIONAL", "LKPP-AGG-2023-TI-ESTIMATE"):
                rows = [row for row in lkpp if row["resource_id"] == resource]
                if not rows or any(
                    row["total_tenders"] <= 0 or row["total_value_idr"] <= 0
                    for row in rows
                ):
                    raise ValueError(f"LKPP needs positive totals for {resource}.")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid context data: {exc}") from exc
    paths = []
    for directory, (name, generate_sample) in CONTEXT_FILES.items():
        target = RAW_DIR / directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if context_dir is None:
            generate_sample(target)
        elif directory == "bpk" and bpk_from_pdf:
            target.write_text(
                json.dumps(bpk, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        elif (context_dir / name).resolve() != target.resolve():
            shutil.copy2(context_dir / name, target)
        paths.append(target)
    return paths

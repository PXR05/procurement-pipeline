import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from bd_procurement.config import (
    MAX_VALID_YEAR,
    MIN_VALID_YEAR,
    QUARANTINE_DIR,
    SILVER_DIR,
    VALID_CURRENCY,
)
from bd_procurement.ingestion.opentender import (
    extract_tender_and_awards,
    stream_opentender_records,
)
from bd_procurement.processing.classifier import ITDomainClassifier
from bd_procurement.processing.normalizer import (
    build_organization_dimension_record,
)


def parse_iso_date(date_str: Optional[str]) -> Tuple[Optional[datetime], bool]:
    if not date_str:
        return None, False
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        is_valid = MIN_VALID_YEAR <= dt.year <= MAX_VALID_YEAR
        return dt, is_valid
    except Exception:
        return None, False


def save_as_parquet(records: List[Dict[str, Any]], target_file: Path):
    target_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(target_file), compression="snappy")


def load_silver_from_parquet(silver_dir: Optional[Path] = None) -> Dict[str, Any]:
    dir_path = silver_dir or SILVER_DIR
    tenders_file = dir_path / "fact_tender.parquet"
    awards_file = dir_path / "fact_award.parquet"
    bridge_file = dir_path / "bridge_award_supplier.parquet"
    orgs_file = dir_path / "dim_organization.parquet"

    if not (
        tenders_file.exists()
        and awards_file.exists()
        and bridge_file.exists()
        and orgs_file.exists()
    ):
        raise FileNotFoundError("One or more Silver Parquet files missing.")

    tenders = pq.read_table(str(tenders_file)).to_pylist()
    awards = pq.read_table(str(awards_file)).to_pylist()
    bridge = pq.read_table(str(bridge_file)).to_pylist()
    orgs = pq.read_table(str(orgs_file)).to_pylist()

    return {
        "processed_records": len(tenders),
        "quarantined_records": 0,
        "fact_tenders": tenders,
        "fact_awards": awards,
        "bridge_award_suppliers": bridge,
        "dim_organizations": orgs,
    }


def process_bronze_to_silver(
    archive_path: Path,
    output_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    out_dir = output_dir or SILVER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    classifier = ITDomainClassifier()

    fact_tenders = []
    fact_awards = []
    bridge_award_suppliers = []
    dim_organizations = {}

    quarantine_count = 0
    quarantine_records = []
    processed_count = 0

    for line_no, record, err in stream_opentender_records(
        archive_path, max_records=max_records
    ):
        if err or not record:
            quarantine_count += 1
            if len(quarantine_records) < 500:
                quarantine_records.append({"line": line_no, "error": err})
            continue

        processed_count += 1
        extracted = extract_tender_and_awards(record)
        tender_raw = extracted["tender"]
        awards_raw = extracted["awards"]

        pub_dt, date_valid = parse_iso_date(tender_raw.get("date_published"))
        pub_year = pub_dt.year if pub_dt else None
        pub_month = pub_dt.month if pub_dt else None
        pub_quarter = ((pub_month - 1) // 3 + 1) if pub_month else None
        quarter_4_flag = bool(pub_quarter == 4) if pub_quarter else False

        it_meta = classifier.classify(
            tender_raw["tender_title"], tender_raw["main_category"]
        )

        buyer_raw_id = tender_raw["buyer_id"]
        buyer_raw_name = tender_raw["buyer_name"]
        buyer_record = build_organization_dimension_record(
            buyer_raw_id, buyer_raw_name, "buyer"
        )
        buyer_key = buyer_record["org_key"]
        dim_organizations[buyer_key] = buyer_record

        proc_entity_raw_id = tender_raw["procuring_entity_id"]
        proc_entity_raw_name = tender_raw["procuring_entity_name"]
        if proc_entity_raw_name:
            pe_rec = build_organization_dimension_record(
                proc_entity_raw_id, proc_entity_raw_name, "procuring_entity"
            )
            proc_entity_key = pe_rec["org_key"]
            dim_organizations[proc_entity_key] = pe_rec
        else:
            proc_entity_key = buyer_key

        tender_key = f"t_{tender_raw['ocid']}_{tender_raw['tender_id']}"

        fact_tenders.append(
            {
                "tender_key": tender_key,
                "ocid": tender_raw["ocid"],
                "release_id": tender_raw["release_id"],
                "tender_id": tender_raw["tender_id"],
                "tender_title": tender_raw["tender_title"],
                "tender_status": tender_raw["tender_status"],
                "date_published": tender_raw["date_published"],
                "pub_year": pub_year,
                "pub_month": pub_month,
                "pub_quarter": pub_quarter,
                "quarter_4_flag": quarter_4_flag,
                "date_valid": date_valid,
                "main_category": tender_raw["main_category"],
                "procurement_method": tender_raw["procurement_method"],
                "buyer_key": buyer_key,
                "procuring_entity_key": proc_entity_key,
                "estimated_value": tender_raw["estimated_value"],
                "min_value": tender_raw["min_value"],
                "currency": tender_raw["currency"],
                "is_it": it_meta["is_it"],
                "it_subcategories": json.dumps(it_meta["subcategories"]),
                "it_primary_subcat": it_meta["primary_subcategory"],
                "it_confidence": it_meta["confidence"],
            }
        )

        for aw in awards_raw:
            aw_dt, aw_date_valid = parse_iso_date(aw.get("award_date"))
            aw_amount = aw.get("award_amount")
            aw_currency = aw.get("award_currency") or "IDR"

            is_amount_valid = (
                isinstance(aw_amount, (int, float))
                and aw_amount > 0
                and aw_currency == VALID_CURRENCY
            )

            award_key = f"aw_{tender_raw['ocid']}_{aw['award_id']}"
            suppliers = aw.get("suppliers") or []
            supplier_count = len(suppliers)

            fact_awards.append(
                {
                    "award_key": award_key,
                    "tender_key": tender_key,
                    "award_id": aw["award_id"],
                    "award_title": aw["award_title"],
                    "award_status": aw["award_status"],
                    "award_date": aw["award_date"],
                    "award_year": aw_dt.year if aw_dt else pub_year,
                    "award_month": aw_dt.month if aw_dt else pub_month,
                    "award_amount": aw_amount,
                    "award_currency": aw_currency,
                    "is_amount_valid": is_amount_valid,
                    "supplier_count": supplier_count,
                }
            )

            allocated_amount = (
                (aw_amount / supplier_count)
                if (is_amount_valid and supplier_count > 0)
                else None
            )
            for s in suppliers:
                s_rec = build_organization_dimension_record(
                    s["raw_supplier_id"], s["raw_supplier_name"], "supplier"
                )
                s_key = s_rec["org_key"]
                dim_organizations[s_key] = s_rec

                bridge_award_suppliers.append(
                    {
                        "award_key": award_key,
                        "tender_key": tender_key,
                        "supplier_key": s_key,
                        "allocated_amount": allocated_amount,
                        "buyer_key": buyer_key,
                        "is_it": it_meta["is_it"],
                        "it_primary_subcat": it_meta["primary_subcategory"],
                        "award_year": aw_dt.year if aw_dt else pub_year,
                        "quarter_4_flag": quarter_4_flag,
                    }
                )

    save_as_parquet(fact_tenders, out_dir / "fact_tender.parquet")
    save_as_parquet(fact_awards, out_dir / "fact_award.parquet")
    save_as_parquet(bridge_award_suppliers, out_dir / "bridge_award_supplier.parquet")

    clean_orgs = []
    for o in dim_organizations.values():
        clean_o = {k: v for k, v in o.items() if k != "attributes"}
        clean_orgs.append(clean_o)
    save_as_parquet(clean_orgs, out_dir / "dim_organization.parquet")

    quarantine_file = QUARANTINE_DIR / "quarantine_records.json"
    with open(quarantine_file, "w", encoding="utf-8") as f:
        json.dump(quarantine_records, f, indent=2)

    return {
        "processed_records": processed_count,
        "quarantined_records": quarantine_count,
        "fact_tenders": fact_tenders,
        "fact_awards": fact_awards,
        "bridge_award_suppliers": bridge_award_suppliers,
        "dim_organizations": list(dim_organizations.values()),
    }

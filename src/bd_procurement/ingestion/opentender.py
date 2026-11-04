import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


def stream_opentender_records(
    archive_path: Path, max_records: Optional[int] = None
) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    if not archive_path.exists():
        raise FileNotFoundError(f"OpenTender archive not found at: {archive_path}")

    line_count = 0
    with gzip.open(archive_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line_count += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                yield line_count, record, None
            except Exception as e:
                yield line_count, None, str(e)

            if max_records and line_count >= max_records:
                break


def extract_tender_and_awards(record: Dict[str, Any]) -> Dict[str, Any]:
    ocid = record.get("ocid")
    release_id = record.get("id")
    buyer = record.get("buyer") or {}
    tender = record.get("tender") or {}
    awards = record.get("awards") or []

    tender_value_obj = tender.get("value") or {}
    tender_min_val_obj = tender.get("minValue") or {}

    val_amount = tender_value_obj.get("amount")
    min_amount = tender_min_val_obj.get("amount")

    val_currency = (
        tender_value_obj.get("currency") or tender_min_val_obj.get("currency") or "IDR"
    )

    procuring_entity = tender.get("procuringEntity") or {}

    tender_data = {
        "ocid": ocid,
        "release_id": release_id,
        "tender_id": str(tender.get("id") or release_id),
        "tender_title": str(tender.get("title") or "").strip(),
        "tender_status": tender.get("status"),
        "date_published": tender.get("datePublished"),
        "main_category": tender.get("mainProcurementCategory"),
        "procurement_method": tender.get("procurementMethod"),
        "buyer_id": str(buyer.get("id") or ""),
        "buyer_name": str(buyer.get("name") or "").strip(),
        "procuring_entity_id": str(procuring_entity.get("id") or ""),
        "procuring_entity_name": str(procuring_entity.get("name") or "").strip(),
        "estimated_value": float(val_amount)
        if isinstance(val_amount, (int, float))
        else None,
        "min_value": float(min_amount)
        if isinstance(min_amount, (int, float))
        else None,
        "currency": val_currency,
        "raw_record_keys": list(record.keys()),
    }

    parsed_awards = []
    for award in awards:
        award_val_obj = award.get("value") or {}
        award_amount = award_val_obj.get("amount")
        award_currency = award_val_obj.get("currency") or "IDR"

        suppliers = award.get("suppliers") or []
        parsed_suppliers = []
        for s in suppliers:
            parsed_suppliers.append(
                {
                    "raw_supplier_id": str(s.get("id") or "").strip(),
                    "raw_supplier_name": str(s.get("name") or "").strip(),
                }
            )

        parsed_awards.append(
            {
                "award_id": str(award.get("id") or ""),
                "award_title": str(
                    award.get("title") or tender_data["tender_title"]
                ).strip(),
                "award_status": award.get("status"),
                "award_date": award.get("date"),
                "award_amount": float(award_amount)
                if isinstance(award_amount, (int, float))
                else None,
                "award_currency": award_currency,
                "suppliers": parsed_suppliers,
            }
        )

    return {
        "tender": tender_data,
        "awards": parsed_awards,
    }

from collections import Counter
from typing import Any, Dict, List


def profile_procurement_quality(
    tenders: List[Dict[str, Any]],
    awards: List[Dict[str, Any]],
    bridge: List[Dict[str, Any]],
    orgs: List[Dict[str, Any]],
    quarantine_count: int = 0,
) -> Dict[str, Any]:
    total_tenders = len(tenders)
    total_awards = len(awards)
    total_bridge = len(bridge)
    total_orgs = len(orgs)

    tender_with_date = sum(1 for t in tenders if t.get("date_published"))
    tender_with_buyer = sum(1 for t in tenders if t.get("buyer_key"))
    tender_with_est = sum(
        1
        for t in tenders
        if (t.get("estimated_value") is not None or t.get("min_value") is not None)
    )

    award_with_date = sum(1 for a in awards if a.get("award_date"))
    award_with_amount = sum(1 for a in awards if a.get("award_amount") is not None)
    award_with_supplier = sum(1 for a in awards if a.get("supplier_count", 0) > 0)

    valid_dates = sum(1 for t in tenders if t.get("date_valid"))
    valid_awards = sum(1 for a in awards if a.get("is_amount_valid"))
    idr_currency_tenders = sum(1 for t in tenders if t.get("currency") == "IDR")

    unique_tenders = len(set(t["tender_key"] for t in tenders))
    unique_awards = len(set(a["award_key"] for a in awards))

    it_tenders = sum(1 for t in tenders if t.get("is_it"))
    q4_tenders = sum(1 for t in tenders if t.get("quarter_4_flag"))

    subcat_counts = Counter(
        t.get("it_primary_subcat") for t in tenders if t.get("is_it")
    )

    quality_report = {
        "summary": {
            "total_records_ingested": total_tenders + quarantine_count,
            "tenders_parsed": total_tenders,
            "awards_parsed": total_awards,
            "bridge_allocations": total_bridge,
            "organizations_cataloged": total_orgs,
            "quarantined_records": quarantine_count,
            "it_procurement_tenders": it_tenders,
            "it_procurement_ratio": round(it_tenders / total_tenders, 4)
            if total_tenders
            else 0,
        },
        "dimensions": {
            "completeness": {
                "tender_date_published_pct": round(
                    tender_with_date / total_tenders * 100, 2
                )
                if total_tenders
                else 0,
                "tender_buyer_id_pct": round(tender_with_buyer / total_tenders * 100, 2)
                if total_tenders
                else 0,
                "tender_estimated_or_min_value_pct": round(
                    tender_with_est / total_tenders * 100, 2
                )
                if total_tenders
                else 0,
                "award_date_pct": round(award_with_date / total_awards * 100, 2)
                if total_awards
                else 0,
                "award_positive_amount_pct": round(
                    award_with_amount / total_awards * 100, 2
                )
                if total_awards
                else 0,
                "award_supplier_identified_pct": round(
                    award_with_supplier / total_awards * 100, 2
                )
                if total_awards
                else 0,
            },
            "validity": {
                "valid_publication_dates_pct": round(
                    valid_dates / total_tenders * 100, 2
                )
                if total_tenders
                else 0,
                "valid_award_amounts_pct": round(valid_awards / total_awards * 100, 2)
                if total_awards
                else 0,
                "idr_currency_pct": round(idr_currency_tenders / total_tenders * 100, 2)
                if total_tenders
                else 0,
            },
            "uniqueness": {
                "tender_uniqueness_pct": round(unique_tenders / total_tenders * 100, 2)
                if total_tenders
                else 0,
                "award_uniqueness_pct": round(unique_awards / total_awards * 100, 2)
                if total_awards
                else 0,
                "duplicate_tender_keys_count": total_tenders - unique_tenders,
                "duplicate_award_keys_count": total_awards - unique_awards,
            },
            "consistency": {
                "q4_rush_tenders_count": q4_tenders,
                "q4_rush_ratio_pct": round(q4_tenders / valid_dates * 100, 2)
                if valid_dates
                else 0,
                "it_subcategories_distribution": dict(subcat_counts),
            },
        },
    }

    return quality_report

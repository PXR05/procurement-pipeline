from collections import defaultdict
from typing import Any, Dict, List, Optional

from bd_procurement.config import (
    AWARD_ESTIMATE_RATIO_ANOMALOUS_LOW,
    AWARD_ESTIMATE_RATIO_NEAR_CEILING,
    HHI_HIGH_CONCENTRATION_THRESHOLD,
    HHI_MODERATE_CONCENTRATION_THRESHOLD,
    MIN_AWARDS_FOR_CONCENTRATION_EVAL,
    SUPPLIER_TOP1_SHARE_ALERT,
)


def compute_buyer_concentration_metrics(
    bridge_records: List[Dict[str, Any]], filter_it_only: bool = True
) -> List[Dict[str, Any]]:

    grouped = defaultdict(
        lambda: {
            "total_value": 0.0,
            "award_count": 0,
            "supplier_values": defaultdict(float),
            "supplier_counts": defaultdict(int),
            "q4_awards": 0,
        }
    )

    for row in bridge_records:
        if filter_it_only and not row.get("is_it"):
            continue

        val = row.get("allocated_amount")
        if val is None or val <= 0:
            continue

        b_key = row["buyer_key"]
        yr = row.get("award_year") or 2023
        key = (b_key, yr)

        grouped[key]["total_value"] += val
        grouped[key]["award_count"] += 1
        s_key = row["supplier_key"]
        grouped[key]["supplier_values"][s_key] += val
        grouped[key]["supplier_counts"][s_key] += 1
        if row.get("quarter_4_flag"):
            grouped[key]["q4_awards"] += 1

    results = []
    for (buyer_key, yr), data in grouped.items():
        total_val = data["total_value"]
        total_awards = data["award_count"]
        supplier_vals = data["supplier_values"]

        if total_val <= 0:
            continue

        shares = [val / total_val for val in supplier_vals.values()]
        sorted_shares = sorted(shares, reverse=True)

        hhi = sum(s**2 for s in shares)
        top1_share = sorted_shares[0] if sorted_shares else 0.0
        top3_share = (
            sum(sorted_shares[:3]) if len(sorted_shares) >= 3 else sum(sorted_shares)
        )
        q4_ratio = (data["q4_awards"] / total_awards) if total_awards > 0 else 0.0

        is_low_sample = total_awards < MIN_AWARDS_FOR_CONCENTRATION_EVAL

        if is_low_sample:
            concentration_class = "LOW_SAMPLE"
        elif hhi >= HHI_HIGH_CONCENTRATION_THRESHOLD:
            concentration_class = "HIGH"
        elif hhi >= HHI_MODERATE_CONCENTRATION_THRESHOLD:
            concentration_class = "MODERATE"
        else:
            concentration_class = "LOW_COMPETITIVE"

        results.append(
            {
                "buyer_key": buyer_key,
                "year": yr,
                "total_procured_value": total_val,
                "total_awards_count": total_awards,
                "distinct_suppliers_count": len(supplier_vals),
                "hhi": round(hhi, 4),
                "top1_supplier_share": round(top1_share, 4),
                "top3_supplier_share": round(top3_share, 4),
                "q4_awards_count": data["q4_awards"],
                "q4_awards_ratio": round(q4_ratio, 4),
                "is_low_sample": is_low_sample,
                "concentration_class": concentration_class,
            }
        )

    return results


def compute_buyer_supplier_pair_metrics(
    bridge_records: List[Dict[str, Any]], filter_it_only: bool = True
) -> List[Dict[str, Any]]:
    pairs = defaultdict(
        lambda: {
            "total_value": 0.0,
            "award_count": 0,
            "years_active": set(),
            "subcategories": set(),
        }
    )

    for row in bridge_records:
        if filter_it_only and not row.get("is_it"):
            continue

        val = row.get("allocated_amount")
        if val is None or val <= 0:
            continue

        pair_key = (row["buyer_key"], row["supplier_key"])
        pairs[pair_key]["total_value"] += val
        pairs[pair_key]["award_count"] += 1
        if row.get("award_year"):
            pairs[pair_key]["years_active"].add(row["award_year"])
        if row.get("it_primary_subcat"):
            pairs[pair_key]["subcategories"].add(row["it_primary_subcat"])

    output = []
    for (b_key, s_key), data in pairs.items():
        output.append(
            {
                "buyer_key": b_key,
                "supplier_key": s_key,
                "total_won_value": data["total_value"],
                "awards_won_count": data["award_count"],
                "years_active_count": len(data["years_active"]),
                "subcategories_count": len(data["subcategories"]),
                "is_repeat_winner": data["award_count"] >= 3,
            }
        )

    return output


def evaluate_review_priority(
    tender_row: Dict[str, Any],
    buyer_metric: Optional[Dict[str, Any]],
    pair_metric: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    reasons = []
    score = 0.0

    if buyer_metric and not buyer_metric.get("is_low_sample"):
        hhi = buyer_metric.get("hhi", 0.0)
        if hhi >= HHI_HIGH_CONCENTRATION_THRESHOLD:
            score += 3.0
            reasons.append(f"HIGH_HHI_CONCENTRATION ({hhi:.2f})")
        top1 = buyer_metric.get("top1_supplier_share", 0.0)
        if top1 >= SUPPLIER_TOP1_SHARE_ALERT:
            score += 2.0
            reasons.append(f"DOMINANT_TOP1_VENDOR ({top1 * 100:.1f}%)")

    if pair_metric and pair_metric.get("is_repeat_winner"):
        count = pair_metric.get("awards_won_count", 0)
        score += 2.5
        reasons.append(f"REPEAT_WINNER_RELATIONSHIP ({count} awards)")

    if tender_row.get("quarter_4_flag"):
        score += 1.0
        reasons.append("YEAR_END_Q4_PUBLICATION")

    est = tender_row.get("min_value") or tender_row.get("estimated_value")
    award_amt = tender_row.get("award_amount")
    if est and award_amt and est > 0:
        ratio = award_amt / est
        if ratio >= AWARD_ESTIMATE_RATIO_NEAR_CEILING and ratio <= 1.05:
            score += 2.0
            reasons.append(f"AWARD_NEAR_CEILING_HPS ({ratio * 100:.1f}%)")
        elif ratio <= AWARD_ESTIMATE_RATIO_ANOMALOUS_LOW:
            score += 1.5
            reasons.append(f"ABNORMALLY_LOW_RATIO ({ratio * 100:.1f}%)")

    priority_level = "LOW"
    if score >= 6.0:
        priority_level = "HIGH"
    elif score >= 3.5:
        priority_level = "MEDIUM"

    return {
        "priority_score": round(score, 1),
        "priority_level": priority_level,
        "reasons": reasons,
        "needs_review": bool(score >= 3.5),
    }

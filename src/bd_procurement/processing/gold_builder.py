import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from bd_procurement.analytics.graph import (
    build_projected_buyer_supplier_graph,
    run_graph_analytics,
)
from bd_procurement.analytics.indicators import (
    compute_buyer_concentration_metrics,
    compute_buyer_supplier_pair_metrics,
    evaluate_review_priority,
)
from bd_procurement.config import GOLD_DIR


def save_as_parquet(records: List[Dict[str, Any]], target_file: Path):
    target_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(target_file), compression="snappy")


def build_gold_marts(
    silver_data: Dict[str, Any], output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    out_dir = output_dir or GOLD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fact_tenders = silver_data["fact_tenders"]
    fact_awards = silver_data["fact_awards"]
    bridge_records = silver_data["bridge_award_suppliers"]
    dim_orgs = silver_data["dim_organizations"]

    org_map = {o["org_key"]: o for o in dim_orgs}
    award_map = {a["award_key"]: a for a in fact_awards}

    bridge_by_tender = defaultdict(list)
    for br in bridge_records:
        bridge_by_tender[br["tender_key"]].append(br)

    buyer_concentration = compute_buyer_concentration_metrics(
        bridge_records, filter_it_only=True
    )

    for bc in buyer_concentration:
        b_info = org_map.get(bc["buyer_key"], {})
        bc["buyer_name"] = b_info.get("canonical_name", bc["buyer_key"])
        bc["raw_buyer_name"] = b_info.get("raw_name", "")

    buyer_conc_map = {bc["buyer_key"]: bc for bc in buyer_concentration}

    pair_metrics = compute_buyer_supplier_pair_metrics(
        bridge_records, filter_it_only=True
    )
    for pm in pair_metrics:
        b_info = org_map.get(pm["buyer_key"], {})
        s_info = org_map.get(pm["supplier_key"], {})
        pm["buyer_name"] = b_info.get("canonical_name", pm["buyer_key"])
        pm["supplier_name"] = s_info.get("canonical_name", pm["supplier_key"])

    pair_map = {(pm["buyer_key"], pm["supplier_key"]): pm for pm in pair_metrics}

    supplier_agg = {}
    for pm in pair_metrics:
        s_key = pm["supplier_key"]
        if s_key not in supplier_agg:
            s_info = org_map.get(s_key, {})
            supplier_agg[s_key] = {
                "supplier_key": s_key,
                "supplier_name": s_info.get("canonical_name", s_key),
                "raw_supplier_name": s_info.get("raw_name", ""),
                "legal_form": s_info.get("legal_form", "unknown"),
                "total_won_value": 0.0,
                "total_awards_count": 0,
                "distinct_buyers_count": 0,
                "buyers_served": set(),
            }
        supplier_agg[s_key]["total_won_value"] += pm["total_won_value"]
        supplier_agg[s_key]["total_awards_count"] += pm["awards_won_count"]
        supplier_agg[s_key]["buyers_served"].add(pm["buyer_key"])

    mart_supplier_dominance = []
    for s_data in supplier_agg.values():
        s_data["distinct_buyers_count"] = len(s_data.pop("buyers_served"))
        mart_supplier_dominance.append(s_data)

    mart_supplier_dominance.sort(key=lambda x: x["total_won_value"], reverse=True)

    mart_review_priority = []
    for t in fact_tenders:
        if not t.get("is_it"):
            continue

        b_key = t["buyer_key"]
        b_metric = buyer_conc_map.get(b_key)

        matching_bridge = bridge_by_tender.get(t["tender_key"], [])
        for br in matching_bridge:
            s_key = br["supplier_key"]
            p_metric = pair_map.get((b_key, s_key))
            aw = award_map.get(br["award_key"], {})

            tender_eval_data = {
                **t,
                "award_amount": aw.get("award_amount"),
            }

            eval_res = evaluate_review_priority(tender_eval_data, b_metric, p_metric)
            if eval_res["needs_review"]:
                mart_review_priority.append(
                    {
                        "tender_key": t["tender_key"],
                        "ocid": t["ocid"],
                        "tender_id": t["tender_id"],
                        "tender_title": t["tender_title"],
                        "it_primary_subcat": t["it_primary_subcat"],
                        "buyer_key": b_key,
                        "buyer_name": org_map.get(b_key, {}).get(
                            "canonical_name", b_key
                        ),
                        "supplier_key": s_key,
                        "supplier_name": org_map.get(s_key, {}).get(
                            "canonical_name", s_key
                        ),
                        "award_amount": aw.get("award_amount"),
                        "estimated_or_min_value": t.get("min_value")
                        or t.get("estimated_value"),
                        "priority_score": eval_res["priority_score"],
                        "priority_level": eval_res["priority_level"],
                        "review_reasons": json.dumps(eval_res["reasons"]),
                    }
                )

    mart_review_priority.sort(key=lambda x: x["priority_score"], reverse=True)

    save_as_parquet(buyer_concentration, out_dir / "mart_buyer_concentration.parquet")
    save_as_parquet(
        mart_supplier_dominance, out_dir / "mart_supplier_dominance.parquet"
    )
    save_as_parquet(pair_metrics, out_dir / "mart_buyer_supplier_pairs.parquet")
    save_as_parquet(mart_review_priority, out_dir / "mart_review_priority.parquet")

    proj_graph = build_projected_buyer_supplier_graph(pair_metrics, dim_orgs)
    graph_insights = run_graph_analytics(proj_graph)

    return {
        "mart_buyer_concentration": buyer_concentration,
        "mart_supplier_dominance": mart_supplier_dominance,
        "mart_buyer_supplier_pairs": pair_metrics,
        "mart_review_priority": mart_review_priority,
        "graph_analytics": graph_insights,
        "projected_graph": proj_graph,
    }

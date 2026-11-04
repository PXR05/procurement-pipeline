import json
import re

import pandas as pd

from bd_procurement.config import DATA_DIR, GOLD_DIR, OUTPUT_DIR, SILVER_DIR
from bd_procurement.ingestion.djpk import load_djpk_fiscal_data
from bd_procurement.ingestion.lkpp import load_lkpp_aggregates


def run_multi_source_integration():
    silver_dir = SILVER_DIR
    gold_dir = GOLD_DIR
    out_dir = OUTPUT_DIR
    for directory in (silver_dir, gold_dir, out_dir):
        directory.mkdir(parents=True, exist_ok=True)

    djpk_rows = load_djpk_fiscal_data(
        DATA_DIR / "raw" / "djpk" / "apbd_fiscal_context_2023.csv"
    )
    lkpp_rows = load_lkpp_aggregates(
        DATA_DIR / "raw" / "lkpp" / "lkpp_open_data_aggregates_2023.csv"
    )
    bpk_rows = json.loads(
        (DATA_DIR / "raw" / "bpk" / "bpk_audit_findings_2023.json").read_text(
            encoding="utf-8-sig"
        )
    )

    df_djpk = pd.DataFrame(djpk_rows)
    df_lkpp = pd.DataFrame(lkpp_rows)
    df_bpk = pd.DataFrame(bpk_rows)
    for field in ("report_id", "entity_name", "page_number", "issue_type", "quote"):
        if field not in df_bpk:
            df_bpk[field] = pd.Series(dtype="object")

    df_djpk.to_parquet(silver_dir / "context_djpk_apbd.parquet", index=False)
    df_lkpp.to_parquet(silver_dir / "context_lkpp_aggregates.parquet", index=False)
    df_bpk.to_parquet(silver_dir / "context_bpk_findings.parquet", index=False)

    df_tenders = pd.read_parquet(silver_dir / "fact_tender.parquet")
    df_awards = pd.read_parquet(silver_dir / "fact_award.parquet")
    df_all_buyers = pd.read_parquet(gold_dir / "ml_buyer_year_all.parquet")
    df_k4 = pd.read_parquet(gold_dir / "ml_buyer_year_clustered_k4.parquet")

    ot_total_completed_tenders = int((df_tenders["tender_status"] == "complete").sum())
    ot_total_it_tenders = int(df_tenders["is_it"].sum())
    ot_total_it_value = float(df_all_buyers["total_procured_value"].sum())

    lkpp_non_it = df_lkpp[df_lkpp["resource_id"] == "LKPP-AGG-2023-NASIONAL"]
    lkpp_national_tenders = int(lkpp_non_it["total_tenders"].sum())
    lkpp_national_value = float(lkpp_non_it["total_value_idr"].sum())

    lkpp_it_row = df_lkpp[df_lkpp["resource_id"] == "LKPP-AGG-2023-TI-ESTIMATE"].iloc[0]
    lkpp_it_tenders = int(lkpp_it_row["total_tenders"])
    lkpp_it_value = float(lkpp_it_row["total_value_idr"])

    reconciliation = [
        {
            "scope": "Total Paket Selesai Nasional (2023)",
            "lkpp_tenders": lkpp_national_tenders,
            "opentender_tenders": ot_total_completed_tenders,
            "tender_coverage_pct": round(
                ot_total_completed_tenders / lkpp_national_tenders * 100, 2
            ),
            "lkpp_value_idr": lkpp_national_value,
            "opentender_value_idr": float(df_awards["award_amount"].dropna().sum()),
        },
        {
            "scope": "Subset Pengadaan TI Nasional (2023)",
            "lkpp_tenders": lkpp_it_tenders,
            "opentender_tenders": ot_total_it_tenders,
            "tender_coverage_pct": round(
                ot_total_it_tenders / lkpp_it_tenders * 100, 2
            ),
            "lkpp_value_idr": lkpp_it_value,
            "opentender_value_idr": ot_total_it_value,
            "value_coverage_pct": round(ot_total_it_value / lkpp_it_value * 100, 2),
        },
    ]
    pd.DataFrame(reconciliation).to_parquet(
        gold_dir / "mart_lkpp_reconciliation.parquet", index=False
    )

    buyers_merged = df_all_buyers.merge(
        df_k4[["buyer_key", "cluster_id"]], on="buyer_key", how="left"
    )
    buyers_merged["cluster_label"] = buyers_merged["cluster_id"].apply(
        lambda x: f"Cluster {int(x)}" if pd.notna(x) else "Sampel Kecil (<3 Award)"
    )

    def normalize_entity(name):
        text = re.sub(r"[^\w\s]", " ", str(name).casefold())
        return " ".join(text.replace("pemerintah daerah", "pemerintah").split())

    buyer_keys = {}
    for row in buyers_merged.to_dict(orient="records"):
        for field in ("buyer_name", "raw_buyer_name"):
            if row.get(field):
                buyer_keys[normalize_entity(row[field])] = row["buyer_key"]
    entities = dict.fromkeys(
        df_bpk["entity_name"].tolist() + df_djpk["region_name"].tolist()
    )
    entity_mapping = {
        name: buyer_keys[normalize_entity(name)]
        for name in entities
        if normalize_entity(name) in buyer_keys
    }

    djpk_it_modal = df_djpk[
        df_djpk["account_name"].str.contains("Modal Peralatan dan Mesin", na=False)
    ].copy()
    djpk_barang_jasa = df_djpk[
        df_djpk["account_name"].str.contains("Barang dan Jasa", na=False)
    ].copy()

    joined_records = []
    for entity_label, b_key in entity_mapping.items():
        b_row = buyers_merged[buyers_merged["buyer_key"] == b_key]
        if b_row.empty:
            continue
        b = b_row.iloc[0]

        d_it = djpk_it_modal[djpk_it_modal["region_name"] == entity_label]
        d_bj = djpk_barang_jasa[djpk_barang_jasa["region_name"] == entity_label]
        apbd_it_alloc = (
            float(d_it.iloc[0]["budget_allocated"]) if not d_it.empty else None
        )
        apbd_it_real = (
            float(d_it.iloc[0]["budget_realized"]) if not d_it.empty else None
        )
        apbd_bj_real = (
            float(d_bj.iloc[0]["budget_realized"]) if not d_bj.empty else None
        )
        ot_share_of_apbd_it_pct = (
            round(float(b["total_procured_value"]) / apbd_it_real * 100, 2)
            if apbd_it_real
            else None
        )

        bpk_match = df_bpk[df_bpk["entity_name"] == entity_label]
        bpk_report_id = bpk_match.iloc[0]["report_id"] if not bpk_match.empty else None
        bpk_page = (
            int(bpk_match.iloc[0]["page_number"]) if not bpk_match.empty else None
        )
        bpk_issue = bpk_match.iloc[0]["issue_type"] if not bpk_match.empty else None
        bpk_quote = bpk_match.iloc[0]["quote"] if not bpk_match.empty else None

        joined_records.append(
            {
                "buyer_key": b_key,
                "buyer_name": b["buyer_name"],
                "cluster_label": b["cluster_label"],
                "ot_it_awards": int(b["unique_awards_count"]),
                "ot_it_value_idr": round(float(b["total_procured_value"]), 2),
                "ot_hhi": round(float(b["hhi"]), 4),
                "apbd_it_modal_allocated_idr": apbd_it_alloc,
                "apbd_it_modal_realized_idr": apbd_it_real,
                "apbd_barang_jasa_realized_idr": apbd_bj_real,
                "ot_tender_to_apbd_it_realized_pct": ot_share_of_apbd_it_pct,
                "bpk_report_id": bpk_report_id,
                "bpk_page_number": bpk_page,
                "bpk_issue_type": bpk_issue,
                "bpk_quote": bpk_quote,
                "bpk_findings": bpk_match.to_dict(orient="records"),
            }
        )

    df_joined = pd.DataFrame(joined_records)
    df_joined.to_parquet(gold_dir / "mart_multi_source_context.parquet", index=False)

    summary = {
        "lkpp_reconciliation": reconciliation,
        "joined_entities": joined_records,
        "unmatched_entities": [name for name in entities if name not in entity_mapping],
    }
    (out_dir / "multi_source_integration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Context results: {out_dir / 'multi_source_integration.json'}")


if __name__ == "__main__":
    run_multi_source_integration()

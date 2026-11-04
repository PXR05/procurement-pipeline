from __future__ import annotations

import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from bd_procurement.config import (
    GOLD_DIR,
    MIN_AWARDS_FOR_CONCENTRATION_EVAL,
    SILVER_DIR,
)

FEATURE_COLS = [
    "log_total_value",
    "log_award_count",
    "suppliers_per_award",
    "q4_award_share",
    "share_hardware",
    "share_software",
    "share_network",
    "share_services",
]

SCALE_ONLY_COLS = [
    "log_total_value",
    "log_award_count",
]


def build_ml_buyer_year_dataset(
    silver_dir: Optional[Path] = None,
    gold_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    s_dir = silver_dir or SILVER_DIR
    g_dir = gold_dir or GOLD_DIR
    g_dir.mkdir(parents=True, exist_ok=True)

    tenders_raw = pq.read_table(str(s_dir / "fact_tender.parquet")).to_pylist()
    awards_raw = pq.read_table(str(s_dir / "fact_award.parquet")).to_pylist()
    bridge_raw = pq.read_table(str(s_dir / "bridge_award_supplier.parquet")).to_pylist()
    orgs_raw = pq.read_table(str(s_dir / "dim_organization.parquet")).to_pylist()

    org_map = {o["org_key"]: o for o in orgs_raw}

    tender_map: Dict[str, Dict[str, Any]] = {}
    for t in tenders_raw:
        tender_map[t["tender_key"]] = t

    award_map: Dict[str, Dict[str, Any]] = {}
    for a in awards_raw:
        award_map[a["award_key"]] = a

    award_suppliers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for br in bridge_raw:
        if (
            br.get("allocated_amount")
            and br["allocated_amount"] > 0
            and br.get("supplier_key")
        ):
            award_suppliers[br["award_key"]].append(br)

    valid_it_awards: List[Dict[str, Any]] = []
    for aw_key, aw in award_map.items():
        if not aw.get("is_amount_valid"):
            continue
        t = tender_map.get(aw["tender_key"])
        if not t or not t.get("is_it"):
            continue
        sup_rows = award_suppliers.get(aw_key, [])
        if not sup_rows:
            continue
        valid_it_awards.append(aw)

    grouped: Dict[Tuple[str, int], Dict[str, Any]] = defaultdict(
        lambda: {
            "total_value": 0.0,
            "unique_awards": set(),
            "q4_awards_by_award_date": 0,
            "q4_awards_by_pub_date": 0,
            "subcat_counts": Counter(),
            "supplier_values": defaultdict(float),
            "supplier_award_counts": defaultdict(int),
            "sample_records": [],
        }
    )

    for aw in valid_it_awards:
        aw_key = aw["award_key"]
        t = tender_map[aw["tender_key"]]
        b_key = t["buyer_key"]
        yr = int(aw.get("award_year") or t.get("pub_year") or 2023)
        key = (b_key, yr)

        aw_month = aw.get("award_month")
        if aw_month is None:
            aw_month = t.get("pub_month")
        is_q4_award = bool(aw_month in (10, 11, 12))
        is_q4_pub = bool(t.get("quarter_4_flag"))

        subcat = t.get("it_primary_subcat") or "hardware"
        sup_rows = award_suppliers[aw_key]

        g = grouped[key]
        g["unique_awards"].add(aw_key)
        if is_q4_award:
            g["q4_awards_by_award_date"] += 1
        if is_q4_pub:
            g["q4_awards_by_pub_date"] += 1
        g["subcat_counts"][subcat] += 1

        for sr in sup_rows:
            alloc = float(sr["allocated_amount"])
            s_key = sr["supplier_key"]
            g["total_value"] += alloc
            g["supplier_values"][s_key] += alloc
            g["supplier_award_counts"][s_key] += 1

        if len(g["sample_records"]) < 5:
            first_sup_key = sup_rows[0]["supplier_key"]
            g["sample_records"].append(
                {
                    "ocid": t["ocid"],
                    "tender_id": t["tender_id"],
                    "award_id": aw["award_id"],
                    "tender_title": t["tender_title"],
                    "it_primary_subcat": subcat,
                    "award_date": aw["award_date"],
                    "award_month": aw_month,
                    "award_amount": float(aw["award_amount"]),
                    "supplier_key": first_sup_key,
                    "supplier_name": org_map.get(first_sup_key, {}).get(
                        "canonical_name", first_sup_key
                    ),
                }
            )

    all_observations: List[Dict[str, Any]] = []
    for (b_key, yr), g in grouped.items():
        n_awards = len(g["unique_awards"])
        total_val = g["total_value"]
        if n_awards == 0 or total_val <= 0:
            continue

        n_suppliers = len(g["supplier_values"])
        suppliers_per_award = n_suppliers / n_awards
        q4_share = g["q4_awards_by_award_date"] / n_awards
        q4_pub_share = g["q4_awards_by_pub_date"] / n_awards

        sc = g["subcat_counts"]
        share_hw = sc.get("hardware", 0) / n_awards
        share_sw = sc.get("software", 0) / n_awards
        share_net = sc.get("infrastructure_network", 0) / n_awards
        share_srv = sc.get("services_security", 0) / n_awards

        shares = [v / total_val for v in g["supplier_values"].values()]
        sorted_shares = sorted(shares, reverse=True)
        hhi = sum(s**2 for s in shares)
        top1_share = sorted_shares[0] if sorted_shares else 0.0
        top3_share = (
            sum(sorted_shares[:3]) if len(sorted_shares) >= 3 else sum(sorted_shares)
        )

        b_info = org_map.get(b_key, {})
        is_low_sample = n_awards < MIN_AWARDS_FOR_CONCENTRATION_EVAL

        all_observations.append(
            {
                "obs_id": f"{b_key}_{yr}",
                "buyer_key": b_key,
                "buyer_name": b_info.get("canonical_name", b_key),
                "raw_buyer_name": b_info.get("raw_name", b_key),
                "year": yr,
                "total_procured_value": round(total_val, 2),
                "unique_awards_count": n_awards,
                "distinct_suppliers_count": n_suppliers,
                "log_total_value": round(math.log1p(total_val), 6),
                "log_award_count": round(math.log1p(n_awards), 6),
                "suppliers_per_award": round(suppliers_per_award, 6),
                "q4_awards_count": g["q4_awards_by_award_date"],
                "q4_award_share": round(q4_share, 6),
                "q4_pub_share": round(q4_pub_share, 6),
                "cnt_hardware": sc.get("hardware", 0),
                "cnt_software": sc.get("software", 0),
                "cnt_network": sc.get("infrastructure_network", 0),
                "cnt_services": sc.get("services_security", 0),
                "share_hardware": round(share_hw, 6),
                "share_software": round(share_sw, 6),
                "share_network": round(share_net, 6),
                "share_services": round(share_srv, 6),
                "hhi": round(hhi, 4),
                "top1_supplier_share": round(top1_share, 4),
                "top3_supplier_share": round(top3_share, 4),
                "is_low_sample": is_low_sample,
            }
        )

    all_observations.sort(key=lambda x: x["total_procured_value"], reverse=True)
    filtered_observations = [o for o in all_observations if not o["is_low_sample"]]
    if not filtered_observations:
        raise ValueError(
            "No eligible buyers. Clustering needs buyers with at least three valid IT awards."
        )

    pq.write_table(
        pa.Table.from_pylist(all_observations),
        str(g_dir / "ml_buyer_year_all.parquet"),
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist(filtered_observations),
        str(g_dir / "ml_buyer_year_filtered.parquet"),
        compression="snappy",
    )

    trace_obs = None
    for obs in filtered_observations:
        if (
            4 <= obs["unique_awards_count"] <= 10
            and obs["cnt_software"] >= 1
            and obs["cnt_hardware"] >= 1
        ):
            trace_obs = obs
            break
    if trace_obs is None:
        trace_obs = filtered_observations[0]

    trace_key = (trace_obs["buyer_key"], trace_obs["year"])
    trace_awards = grouped[trace_key]["sample_records"]

    funnel_summary = {
        "raw_records_ingested": len(tenders_raw),
        "unique_tenders_valid": len(tender_map),
        "unique_awards_all_valid": len(award_map),
        "bridge_rows_all": len(bridge_raw),
        "it_tenders_total": sum(1 for t in tender_map.values() if t.get("is_it")),
        "it_tenders_planning": sum(
            1
            for t in tender_map.values()
            if t.get("is_it") and t.get("tender_status") == "planning"
        ),
        "it_tenders_complete": sum(
            1
            for t in tender_map.values()
            if t.get("is_it") and t.get("tender_status") == "complete"
        ),
        "it_unique_awards_valid": len(valid_it_awards),
        "it_total_value_idr": round(
            sum(o["total_procured_value"] for o in all_observations), 2
        ),
        "buyer_year_all_count": len(all_observations),
        "buyer_year_low_sample_count": len(all_observations)
        - len(filtered_observations),
        "buyer_year_low_sample_1_award": sum(
            1 for o in all_observations if o["unique_awards_count"] == 1
        ),
        "buyer_year_low_sample_2_awards": sum(
            1 for o in all_observations if o["unique_awards_count"] == 2
        ),
        "buyer_year_filtered_count": len(filtered_observations),
        "filtered_awards_covered": sum(
            o["unique_awards_count"] for o in filtered_observations
        ),
        "filtered_value_covered_idr": round(
            sum(o["total_procured_value"] for o in filtered_observations), 2
        ),
        "q4_it_awards_by_award_date": sum(
            o["q4_awards_count"] for o in all_observations
        ),
        "q4_it_awards_by_pub_date": sum(
            g["q4_awards_by_pub_date"] for g in grouped.values()
        ),
        "traceable_example": {
            "observation": trace_obs,
            "sample_awards": trace_awards,
        },
    }

    return {
        "all_observations": all_observations,
        "filtered_observations": filtered_observations,
        "funnel_summary": funnel_summary,
    }


def align_cluster_labels(
    ref_centroids: np.ndarray, target_centroids: np.ndarray, target_labels: np.ndarray
) -> Tuple[np.ndarray, Dict[int, int]]:
    k = ref_centroids.shape[0]
    cost_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            cost_matrix[i, j] = np.linalg.norm(ref_centroids[i] - target_centroids[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping[int(lbl)] for lbl in target_labels], dtype=int)
    return aligned, mapping


def run_kmeans_experiments(
    observations: List[Dict[str, Any]],
    k_values: Tuple[int, ...] = (2, 3, 4, 5, 6),
    seeds: Tuple[int, ...] = (42, 7, 123, 2024, 999),
    primary_seed: int = 42,
) -> Dict[str, Any]:
    if not k_values or min(k_values) < 2:
        raise ValueError("Cluster counts must be at least two.")
    if len(seeds) < 2 or primary_seed not in seeds:
        raise ValueError("Supply at least two seeds, including the primary seed.")
    if len(observations) <= max(k_values):
        raise ValueError(
            f"Clustering k={max(k_values)} needs at least {max(k_values) + 1} eligible buyer-year observations; found {len(observations)}."
        )
    primary_index = seeds.index(primary_seed)
    df = pd.DataFrame(observations)
    X_raw = df[FEATURE_COLS].to_numpy(dtype=float)
    if not np.isfinite(X_raw).all():
        raise ValueError("Clustering features contain missing or non-finite values.")
    if len(np.unique(X_raw, axis=0)) < max(k_values):
        raise ValueError(
            f"Clustering needs at least {max(k_values)} distinct feature rows."
        )
    X_scale_raw = df[SCALE_ONLY_COLS].to_numpy(dtype=float)

    t0 = time.perf_counter()
    sk_scaler = SklearnStandardScaler()
    X_sk_scaled = sk_scaler.fit_transform(X_raw)
    sk_prep_ms = (time.perf_counter() - t0) * 1000.0

    sk_scale_only_scaler = SklearnStandardScaler()
    X_sk_scale_only = sk_scale_only_scaler.fit_transform(X_scale_raw)

    sk_results_by_k: Dict[int, Dict[str, Any]] = {}
    ablation_by_k: Dict[int, Dict[str, Any]] = {}

    for k in k_values:
        seed_sil_euclid = []
        seed_sil_sqeuclid = []
        seed_inertias = []
        seed_labels_list = []
        train_times_ms = []
        infer_times_ms = []
        primary_model = None
        primary_labels = None

        for sd in seeds:
            km = SklearnKMeans(
                n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=sd
            )
            t_fit0 = time.perf_counter()
            km.fit(X_sk_scaled)
            t_fit_ms = (time.perf_counter() - t_fit0) * 1000.0

            t_pred0 = time.perf_counter()
            lbls = km.predict(X_sk_scaled)
            t_pred_ms = (time.perf_counter() - t_pred0) * 1000.0

            sil_e = float(silhouette_score(X_sk_scaled, lbls, metric="euclidean"))
            sil_sq = float(silhouette_score(X_sk_scaled, lbls, metric="sqeuclidean"))

            seed_sil_euclid.append(sil_e)
            seed_sil_sqeuclid.append(sil_sq)
            seed_inertias.append(float(km.inertia_))
            seed_labels_list.append(lbls)
            train_times_ms.append(t_fit_ms)
            infer_times_ms.append(t_pred_ms)

            if sd == primary_seed:
                primary_model = km
                primary_labels = lbls

        aris = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                aris.append(
                    float(adjusted_rand_score(seed_labels_list[i], seed_labels_list[j]))
                )

        sk_results_by_k[k] = {
            "k": k,
            "silhouette_euclidean_primary": round(seed_sil_euclid[primary_index], 4),
            "silhouette_euclidean_mean": round(float(np.mean(seed_sil_euclid)), 4),
            "silhouette_euclidean_std": round(float(np.std(seed_sil_euclid)), 4),
            "silhouette_sqeuclidean_primary": round(
                seed_sil_sqeuclid[primary_index], 4
            ),
            "silhouette_sqeuclidean_mean": round(float(np.mean(seed_sil_sqeuclid)), 4),
            "silhouette_sqeuclidean_std": round(float(np.std(seed_sil_sqeuclid)), 4),
            "inertia_primary": round(seed_inertias[primary_index], 2),
            "seed_stability_ari_mean": round(float(np.mean(aris)), 4),
            "seed_stability_ari_min": round(float(np.min(aris)), 4),
            "prep_time_ms": round(sk_prep_ms, 3),
            "train_time_ms_mean": round(float(np.mean(train_times_ms)), 3),
            "infer_time_ms_mean": round(float(np.mean(infer_times_ms)), 3),
            "centroids": primary_model.cluster_centers_.tolist(),
            "labels": primary_labels.tolist(),
            "cluster_sizes": {
                int(c): int(cnt)
                for c, cnt in sorted(Counter(primary_labels.tolist()).items())
            },
        }

        km_scale = SklearnKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=primary_seed,
        )
        lbls_scale = km_scale.fit_predict(X_sk_scale_only)
        sil_scale_e = float(
            silhouette_score(X_sk_scale_only, lbls_scale, metric="euclidean")
        )
        cross_ari = float(adjusted_rand_score(primary_labels, lbls_scale))
        ablation_by_k[k] = {
            "k": k,
            "scale_only_silhouette_euclidean": round(sil_scale_e, 4),
            "full_features_silhouette_euclidean": round(
                seed_sil_euclid[primary_index], 4
            ),
            "ari_scale_vs_full": round(cross_ari, 4),
        }

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.ml.clustering import KMeans as SparkKMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml.feature import StandardScaler as SparkStandardScaler
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import SparkSession

    t_spark_init0 = time.perf_counter()
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ProcurementKMeans")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    spark_init_ms = (time.perf_counter() - t_spark_init0) * 1000.0

    spark_results_by_k: Dict[int, Dict[str, Any]] = {}
    spark_version = spark.version

    try:
        t_sp_prep0 = time.perf_counter()
        sdf_raw = spark.createDataFrame(df[["obs_id"] + FEATURE_COLS])
        assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
        sdf_vec = assembler.transform(sdf_raw)
        sp_scaler = SparkStandardScaler(
            inputCol="raw_features", outputCol="features", withStd=True, withMean=True
        )
        sp_scaler_model = sp_scaler.fit(sdf_vec)
        sdf_scaled = sp_scaler_model.transform(sdf_vec).cache()
        _ = sdf_scaled.count()
        sp_prep_ms = (time.perf_counter() - t_sp_prep0) * 1000.0

        evaluator_sq = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )

        n_obs = len(df)
        ddof_scale = math.sqrt((n_obs - 1) / n_obs)

        for k in k_values:
            sp_sil_euclid = []
            sp_sil_sqeuclid = []
            sp_Evaluator_sq = []
            sp_wssse = []
            sp_seed_labels = []
            sp_train_ms = []
            sp_infer_ms = []
            sp_primary_labels = None
            sp_primary_centroids = None

            for sd in seeds:
                skm = SparkKMeans(
                    featuresCol="features",
                    predictionCol="prediction",
                    k=k,
                    initMode="k-means||",
                    initSteps=5,
                    maxIter=300,
                    seed=sd,
                )
                t_fit0 = time.perf_counter()
                smodel = skm.fit(sdf_scaled)
                t_fit_ms = (time.perf_counter() - t_fit0) * 1000.0

                t_pred0 = time.perf_counter()
                scored = smodel.transform(sdf_scaled)
                preds_rows = scored.select("obs_id", "prediction").collect()
                t_pred_ms = (time.perf_counter() - t_pred0) * 1000.0

                eval_sq_val = float(evaluator_sq.evaluate(scored))
                pred_map = {r["obs_id"]: int(r["prediction"]) for r in preds_rows}
                lbls_arr = np.array([pred_map[oid] for oid in df["obs_id"]], dtype=int)

                sil_e = float(
                    silhouette_score(X_sk_scaled, lbls_arr, metric="euclidean")
                )
                sil_sq = float(
                    silhouette_score(X_sk_scaled, lbls_arr, metric="sqeuclidean")
                )

                sp_sil_euclid.append(sil_e)
                sp_sil_sqeuclid.append(sil_sq)
                sp_Evaluator_sq.append(eval_sq_val)
                sp_wssse.append(float(smodel.summary.trainingCost))
                sp_seed_labels.append(lbls_arr)
                sp_train_ms.append(t_fit_ms)
                sp_infer_ms.append(t_pred_ms)

                if sd == primary_seed:
                    sp_primary_centroids = (
                        np.array(smodel.clusterCenters(), dtype=float) / ddof_scale
                    )
                    sp_primary_labels = lbls_arr

            sp_aris = []
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    sp_aris.append(
                        float(adjusted_rand_score(sp_seed_labels[i], sp_seed_labels[j]))
                    )

            ref_c = np.array(sk_results_by_k[k]["centroids"], dtype=float)
            ref_lbls = np.array(sk_results_by_k[k]["labels"], dtype=int)
            aligned_sp_lbls, sp_to_sk_map = align_cluster_labels(
                ref_c, sp_primary_centroids, sp_primary_labels
            )

            cross_ari = float(adjusted_rand_score(ref_lbls, aligned_sp_lbls))
            exact_match_pct = float(np.mean(ref_lbls == aligned_sp_lbls) * 100.0)

            spark_results_by_k[k] = {
                "k": k,
                "silhouette_euclidean_primary": round(sp_sil_euclid[primary_index], 4),
                "silhouette_euclidean_mean": round(float(np.mean(sp_sil_euclid)), 4),
                "silhouette_euclidean_std": round(float(np.std(sp_sil_euclid)), 4),
                "silhouette_sqeuclidean_primary": round(
                    sp_sil_sqeuclid[primary_index], 4
                ),
                "spark_evaluator_sqeuclidean_primary": round(
                    sp_Evaluator_sq[primary_index], 4
                ),
                "wssse_primary": round(sp_wssse[primary_index], 2),
                "seed_stability_ari_mean": round(float(np.mean(sp_aris)), 4),
                "seed_stability_ari_min": round(float(np.min(sp_aris)), 4),
                "prep_time_ms": round(sp_prep_ms, 3),
                "train_time_ms_mean": round(float(np.mean(sp_train_ms)), 3),
                "infer_time_ms_mean": round(float(np.mean(sp_infer_ms)), 3),
                "aligned_labels": aligned_sp_lbls.tolist(),
                "raw_to_aligned_map": sp_to_sk_map,
                "aligned_cluster_sizes": {
                    int(c): int(cnt)
                    for c, cnt in sorted(Counter(aligned_sp_lbls.tolist()).items())
                },
                "ari_with_sklearn": round(cross_ari, 4),
                "exact_agreement_pct_with_sklearn": round(exact_match_pct, 2),
            }
    finally:
        spark.stop()

    return {
        "n_observations": len(df),
        "feature_columns": FEATURE_COLS,
        "k_values": k_values,
        "seeds": seeds,
        "primary_seed": primary_seed,
        "scaler_means": {
            col: round(float(m), 6) for col, m in zip(FEATURE_COLS, sk_scaler.mean_)
        },
        "scaler_stds": {
            col: round(float(s), 6) for col, s in zip(FEATURE_COLS, sk_scaler.scale_)
        },
        "sklearn_by_k": sk_results_by_k,
        "spark_by_k": spark_results_by_k,
        "ablation_by_k": ablation_by_k,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "sklearn_version": __import__("sklearn").__version__,
            "scipy_version": __import__("scipy").__version__,
            "pyspark_version": spark_version,
            "spark_master": "local[*]",
            "spark_init_time_ms": round(spark_init_ms, 2),
        },
    }

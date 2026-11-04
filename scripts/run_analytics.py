import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bd_procurement.analytics.clustering import (
    build_ml_buyer_year_dataset,
    run_kmeans_experiments,
)
from bd_procurement.analytics.graph import (
    export_graphml,
)
from bd_procurement.config import (
    CHARTS_DIR,
    GOLD_DIR,
    OUTPUT_DIR,
    SILVER_DIR,
)

CLUSTER_NAMES = {
    0: "Cluster 0: Belanja Besar Dominan Software & Jaringan",
    1: "Cluster 1: Belanja Menengah Hardware Rutin Tersebar",
    2: "Cluster 2: Frekuensi Tinggi Hardware & Pemenang Berulang",
    3: "Cluster 3: Spesialis Layanan/Keamanan & Kontrak Q4",
}

CLUSTER_SHORT_NAMES = {
    0: "C0: Skala Besar Software/Jaringan",
    1: "C1: Hardware Rutin Tersebar",
    2: "C2: Hardware Frekuensi Tinggi",
    3: "C3: Layanan/Keamanan & Q4",
}

CLUSTER_COLORS = {
    0: "#2563eb",
    1: "#059669",
    2: "#d97706",
    3: "#dc2626",
    -1: "#94a3b8",
}


def compute_cluster_profiles(observations, labels, spark_labels):
    df = pd.DataFrame(observations)
    df["cluster_id"] = labels
    df["spark_cluster_id"] = spark_labels
    df["cluster_name"] = [CLUSTER_NAMES.get(int(c), f"Cluster {c}") for c in labels]

    profiles = []
    for cid in sorted(df["cluster_id"].unique()):
        sub = df[df["cluster_id"] == cid]
        top_buyers = (
            sub.sort_values("total_procured_value", ascending=False)
            .head(5)[
                [
                    "buyer_key",
                    "buyer_name",
                    "total_procured_value",
                    "unique_awards_count",
                    "distinct_suppliers_count",
                    "hhi",
                    "top1_supplier_share",
                    "q4_award_share",
                    "share_hardware",
                    "share_software",
                    "share_network",
                    "share_services",
                ]
            ]
            .to_dict(orient="records")
        )

        profiles.append(
            {
                "cluster_id": int(cid),
                "cluster_name": CLUSTER_NAMES.get(int(cid), f"Cluster {cid}"),
                "count": int(len(sub)),
                "pct_of_cohort": round(float(len(sub) / len(df) * 100.0), 2),
                "mean_total_value_idr": round(
                    float(sub["total_procured_value"].mean()), 2
                ),
                "median_total_value_idr": round(
                    float(sub["total_procured_value"].median()), 2
                ),
                "sum_total_value_idr": round(
                    float(sub["total_procured_value"].sum()), 2
                ),
                "mean_unique_awards": round(
                    float(sub["unique_awards_count"].mean()), 2
                ),
                "median_unique_awards": round(
                    float(sub["unique_awards_count"].median()), 2
                ),
                "mean_distinct_suppliers": round(
                    float(sub["distinct_suppliers_count"].mean()), 2
                ),
                "mean_suppliers_per_award": round(
                    float(sub["suppliers_per_award"].mean()), 4
                ),
                "mean_q4_award_share": round(float(sub["q4_award_share"].mean()), 4),
                "mean_share_hardware": round(float(sub["share_hardware"].mean()), 4),
                "mean_share_software": round(float(sub["share_software"].mean()), 4),
                "mean_share_network": round(float(sub["share_network"].mean()), 4),
                "mean_share_services": round(float(sub["share_services"].mean()), 4),
                "mean_hhi": round(float(sub["hhi"].mean()), 4),
                "median_hhi": round(float(sub["hhi"].median()), 4),
                "mean_top1_share": round(float(sub["top1_supplier_share"].mean()), 4),
                "mean_top3_share": round(float(sub["top3_supplier_share"].mean()), 4),
                "top_buyers": top_buyers,
            }
        )
    return profiles, df


def analyze_clustered_graph(
    silver_dir: Path,
    filtered_df: pd.DataFrame,
    all_obs: list[dict],
) -> tuple[nx.Graph, dict]:
    tenders = {
        t["tender_key"]: t
        for t in pq.read_table(str(silver_dir / "fact_tender.parquet")).to_pylist()
    }
    awards = {
        a["award_key"]: a
        for a in pq.read_table(str(silver_dir / "fact_award.parquet")).to_pylist()
    }
    bridge = pq.read_table(
        str(silver_dir / "bridge_award_supplier.parquet")
    ).to_pylist()
    orgs = {
        o["org_key"]: o
        for o in pq.read_table(str(silver_dir / "dim_organization.parquet")).to_pylist()
    }

    buyer_cluster_map = {}
    for _, row in filtered_df.iterrows():
        buyer_cluster_map[row["buyer_key"]] = int(row["cluster_id"])

    pair_data = defaultdict(
        lambda: {
            "unique_awards": set(),
            "total_value": 0.0,
            "subcategories": Counter(),
            "sample_awards": [],
        }
    )

    for br in bridge:
        if not br.get("is_it"):
            continue
        val = br.get("allocated_amount")
        if not val or val <= 0:
            continue
        b_key = br["buyer_key"]
        s_key = br["supplier_key"]
        aw_key = br["award_key"]
        t_key = br["tender_key"]
        aw = awards.get(aw_key, {})
        t = tenders.get(t_key, {})

        pd_entry = pair_data[(b_key, s_key)]
        pd_entry["unique_awards"].add(aw_key)
        pd_entry["total_value"] += float(val)
        if br.get("it_primary_subcat"):
            pd_entry["subcategories"][br["it_primary_subcat"]] += 1
        if len(pd_entry["sample_awards"]) < 3:
            pd_entry["sample_awards"].append(
                {
                    "ocid": t.get("ocid", ""),
                    "tender_id": t.get("tender_id", ""),
                    "award_id": aw.get("award_id", ""),
                    "tender_title": t.get("tender_title", ""),
                    "award_date": aw.get("award_date", ""),
                    "amount": float(val),
                    "subcat": br.get("it_primary_subcat", ""),
                }
            )

    G = nx.Graph()
    for (b_key, s_key), pinfo in pair_data.items():
        cnt = len(pinfo["unique_awards"])
        val = pinfo["total_value"]
        cid = buyer_cluster_map.get(b_key, -1)
        cname = CLUSTER_SHORT_NAMES.get(cid, "Low-Sample (<3 Award)")

        if not G.has_node(b_key):
            b_org = orgs.get(b_key, {})
            G.add_node(
                b_key,
                node_type="buyer",
                name=b_org.get("canonical_name", b_key),
                cluster_id=cid,
                cluster_label=cname,
            )
        if not G.has_node(s_key):
            s_org = orgs.get(s_key, {})
            G.add_node(
                s_key,
                node_type="supplier",
                name=s_org.get("canonical_name", s_key),
                cluster_id=-2,
                cluster_label="Supplier",
            )
        G.add_edge(
            b_key,
            s_key,
            weight=round(val, 2),
            contract_count=cnt,
            primary_subcat=pinfo["subcategories"].most_common(1)[0][0]
            if pinfo["subcategories"]
            else "hardware",
        )

    deg_dict = dict(G.degree())
    deg_cent = nx.degree_centrality(G)
    pagerank = nx.pagerank(G, weight="contract_count")
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_cc = components[0] if components else set()

    for n in G.nodes():
        G.nodes[n]["degree"] = deg_dict[n]
        G.nodes[n]["degree_centrality"] = round(float(deg_cent[n]), 6)
        G.nodes[n]["pagerank"] = round(float(pagerank[n]), 6)

    top10_pr = [
        {
            "node_key": n,
            "name": G.nodes[n]["name"],
            "node_type": G.nodes[n]["node_type"],
            "cluster_id": G.nodes[n]["cluster_id"],
            "degree": deg_dict[n],
            "pagerank": round(float(pr), 6),
            "degree_centrality": round(float(deg_cent[n]), 6),
        }
        for n, pr in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    top10_deg = [
        {
            "node_key": n,
            "name": G.nodes[n]["name"],
            "node_type": G.nodes[n]["node_type"],
            "cluster_id": G.nodes[n]["cluster_id"],
            "degree": deg_dict[n],
            "degree_centrality": round(float(dc), 6),
            "pagerank": round(float(pagerank[n]), 6),
        }
        for n, dc in sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    supplier_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "supplier"]
    buyer_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "buyer"]

    supplier_Bridge_rows = []
    for s in supplier_nodes:
        neighbors = list(G.neighbors(s))
        clusters_served = sorted(
            {
                G.nodes[b]["cluster_id"]
                for b in neighbors
                if G.nodes[b]["cluster_id"] >= 0
            }
        )
        all_clusters_served = sorted({G.nodes[b]["cluster_id"] for b in neighbors})
        total_contracts = sum(G[s][b]["contract_count"] for b in neighbors)
        total_val = sum(G[s][b]["weight"] for b in neighbors)
        buyer_names = [
            f"{G.nodes[b]['name']} (C{G.nodes[b]['cluster_id']}: {G[s][b]['contract_count']} kontrak)"
            if G.nodes[b]["cluster_id"] >= 0
            else f"{G.nodes[b]['name']} (<3 award: {G[s][b]['contract_count']} kontrak)"
            for b in neighbors
        ]
        supplier_Bridge_rows.append(
            {
                "supplier_key": s,
                "supplier_name": G.nodes[s]["name"],
                "distinct_buyers": len(neighbors),
                "filtered_clusters_served": clusters_served,
                "filtered_clusters_count": len(clusters_served),
                "all_clusters_count": len(all_clusters_served),
                "total_contracts": total_contracts,
                "total_value_idr": round(total_val, 2),
                "pagerank": round(float(pagerank[s]), 6),
                "degree_centrality": round(float(deg_cent[s]), 6),
                "buyers_detail": buyer_names[:6],
            }
        )

    supplier_Bridge_rows.sort(
        key=lambda x: (x["distinct_buyers"], x["total_value_idr"]), reverse=True
    )

    multi_buyer_suppliers = [
        r for r in supplier_Bridge_rows if r["distinct_buyers"] >= 2
    ]

    cross_cluster_suppliers = [
        r for r in supplier_Bridge_rows if r["filtered_clusters_count"] >= 2
    ]

    intra_cluster_shared = {}
    for cid in [0, 1, 2, 3]:
        c_buyers = [b for b in buyer_nodes if G.nodes[b]["cluster_id"] == cid]
        sup_counter = Counter()
        for b in c_buyers:
            for s in G.neighbors(b):
                sup_counter[s] += 1
        shared_s = [
            {
                "supplier_key": s,
                "supplier_name": G.nodes[s]["name"],
                "buyers_in_cluster": cnt,
                "total_buyers_overall": deg_dict[s],
            }
            for s, cnt in sup_counter.most_common()
            if cnt >= 2
        ]
        intra_cluster_shared[cid] = {
            "shared_suppliers_count": len(shared_s),
            "top_shared_suppliers": shared_s[:5],
        }

    cluster_network_stats = []
    for cid in [0, 1, 2, 3, -1]:
        c_buyers = [b for b in buyer_nodes if G.nodes[b]["cluster_id"] == cid]
        if not c_buyers:
            continue
        c_edges = [(b, s, G[b][s]) for b in c_buyers for s in G.neighbors(b)]
        repeat_edges_2plus = [e for e in c_edges if e[2]["contract_count"] >= 2]
        repeat_edges_3plus = [e for e in c_edges if e[2]["contract_count"] >= 3]
        max_edge = max(c_edges, key=lambda e: (e[2]["contract_count"], e[2]["weight"]))

        cluster_network_stats.append(
            {
                "cluster_id": cid,
                "cluster_label": CLUSTER_SHORT_NAMES.get(cid, "Low-Sample (<3 Award)"),
                "buyer_nodes_count": len(c_buyers),
                "edges_count": len(c_edges),
                "unique_suppliers_connected": len({e[1] for e in c_edges}),
                "mean_buyer_degree": round(
                    float(np.mean([deg_dict[b] for b in c_buyers])), 2
                ),
                "mean_buyer_pagerank": round(
                    float(np.mean([pagerank[b] for b in c_buyers])), 6
                ),
                "repeat_edges_2plus_count": len(repeat_edges_2plus),
                "repeat_edges_2plus_pct": round(
                    len(repeat_edges_2plus) / len(c_edges) * 100.0, 2
                )
                if c_edges
                else 0.0,
                "repeat_edges_3plus_count": len(repeat_edges_3plus),
                "repeat_edges_3plus_pct": round(
                    len(repeat_edges_3plus) / len(c_edges) * 100.0, 2
                )
                if c_edges
                else 0.0,
                "max_repeat_edge": {
                    "buyer_key": max_edge[0],
                    "buyer_name": G.nodes[max_edge[0]]["name"],
                    "supplier_key": max_edge[1],
                    "supplier_name": G.nodes[max_edge[1]]["name"],
                    "contract_count": max_edge[2]["contract_count"],
                    "total_value_idr": max_edge[2]["weight"],
                    "sample_awards": pair_data[(max_edge[0], max_edge[1])][
                        "sample_awards"
                    ],
                },
            }
        )

    all_edges_sorted_by_cnt = sorted(
        [(b, s, pinfo) for (b, s), pinfo in pair_data.items()],
        key=lambda x: (len(x[2]["unique_awards"]), x[2]["total_value"]),
        reverse=True,
    )
    top_repeat_examples = []
    for b, s, pinfo in all_edges_sorted_by_cnt[:5]:
        top_repeat_examples.append(
            {
                "buyer_key": b,
                "buyer_name": G.nodes[b]["name"],
                "buyer_cluster_id": G.nodes[b]["cluster_id"],
                "supplier_key": s,
                "supplier_name": G.nodes[s]["name"],
                "contract_count": len(pinfo["unique_awards"]),
                "total_value_idr": round(pinfo["total_value"], 2),
                "sample_awards": pinfo["sample_awards"],
            }
        )

    bridge_supplier_traces = []
    for br_sup in cross_cluster_suppliers[:5]:
        s_key = br_sup["supplier_key"]
        contracts_by_buyer = []
        for b_key in G.neighbors(s_key):
            pinfo = pair_data[(b_key, s_key)]
            contracts_by_buyer.append(
                {
                    "buyer_key": b_key,
                    "buyer_name": G.nodes[b_key]["name"],
                    "cluster_id": G.nodes[b_key]["cluster_id"],
                    "contract_count": len(pinfo["unique_awards"]),
                    "total_value_idr": round(pinfo["total_value"], 2),
                    "sample_awards": pinfo["sample_awards"][:2],
                }
            )
        bridge_supplier_traces.append(
            {
                **br_sup,
                "contracts_by_buyer": contracts_by_buyer,
            }
        )

    summary = {
        "total_nodes": G.number_of_nodes(),
        "buyer_nodes_count": len(buyer_nodes),
        "supplier_nodes_count": len(supplier_nodes),
        "total_edges": G.number_of_edges(),
        "total_unique_it_awards": sum(
            d["contract_count"] for _, _, d in G.edges(data=True)
        ),
        "connected_components_count": len(components),
        "largest_component_nodes": len(largest_cc),
        "largest_component_pct": round(
            len(largest_cc) / G.number_of_nodes() * 100.0, 2
        ),
        "second_largest_component_nodes": len(components[1])
        if len(components) > 1
        else 0,
        "pagerank_differs_from_degree": bool(
            any(abs(pagerank[n] - deg_cent[n]) > 1e-5 for n in G.nodes())
        ),
        "pagerank_top10": top10_pr,
        "degree_centrality_top10": top10_deg,
        "multi_buyer_suppliers_count": len(multi_buyer_suppliers),
        "cross_cluster_bridge_suppliers_count": len(cross_cluster_suppliers),
        "top_bridge_suppliers": bridge_supplier_traces,
        "intra_cluster_shared": intra_cluster_shared,
        "cluster_network_stats": cluster_network_stats,
        "top_repeat_edge_examples": top_repeat_examples,
    }
    return G, summary


def generate_architecture_diagram_v2(out_path: Path, funnel: dict):
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    ax.text(
        50,
        47.2,
        "Arsitektur Solusi Big Data Pengadaan TI: Komponen Berjalan vs. Rencana Pengembangan",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )

    def draw_box(
        x,
        y,
        w,
        h,
        title,
        subtitle,
        bg="#f8fafc",
        border="#1e293b",
        dashed=False,
        title_color="#0f172a",
    ):
        ls = "--" if dashed else "-"
        lw = 1.4 if dashed else 1.8
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=bg,
            edgecolor=border,
            linestyle=ls,
            linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h * 0.65,
            title,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.3,
            color="#334155",
        )

    ax.text(
        11,
        42.5,
        "1. Sumber Data",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1e3a8a",
    )
    draw_box(
        2.5,
        27.5,
        17,
        12.5,
        "OpenTender OCDS (Aktif)",
        f"2023: {funnel['raw_records_ingested']:,} record\nFormat: OCDS JSONL.GZ",
        bg="#dbeafe",
        border="#1d4ed8",
        dashed=False,
        title_color="#1e3a8a",
    )
    draw_box(
        2.5,
        9.5,
        17,
        15.5,
        "Sumber Konteks (Ilustrasi)",
        "• DJPK APBD (CSV contoh lokal)\n• BPK IHPS (JSON contoh lokal)\n• LKPP Agregat (CSV contoh)\n[Belum diintegrasikan ke ML]",
        bg="#f8fafc",
        border="#64748b",
        dashed=True,
        title_color="#475569",
    )

    ax.text(
        39,
        42.5,
        "2. Ingesti & Penyimpanan (Lakehouse)",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1e3a8a",
    )
    draw_box(
        24.5,
        31.0,
        29,
        9.0,
        "Bronze Layer & Manifest SHA-256 (Aktif)",
        "Pembacaan streaming JSONL.GZ, validasi skema,\npencatatan checksum SHA-256 & karantina",
        bg="#fef3c7",
        border="#b45309",
        dashed=False,
        title_color="#78350f",
    )
    draw_box(
        24.5,
        20.0,
        29,
        9.0,
        "Silver Layer: Tabel Parquet Bersih (Aktif)",
        "fact_tender (133.846), fact_award (75.619),\nbridge_award_supplier, dim_organization (29.657)",
        bg="#e0f2fe",
        border="#0369a1",
        dashed=False,
        title_color="#0c4a6e",
    )
    draw_box(
        24.5,
        9.5,
        29,
        8.5,
        "Penyimpanan Lokal & HDFS Single-Node (Aktif)",
        "Parquet Snappy lokal + sinkronisasi Docker HDFS\n(1 NameNode, 1 DataNode, replication=1)",
        bg="#f1f5f9",
        border="#334155",
        dashed=False,
        title_color="#0f172a",
    )

    ax.text(
        69,
        42.5,
        "3. Pemrosesan, ML & Graf",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1e3a8a",
    )
    draw_box(
        58.0,
        31.0,
        22,
        9.0,
        "Gold & Dataset ML (Aktif)",
        f"Python + PyArrow\nInstansi–Tahun (2023)\nN={funnel['buyer_year_all_count']} (N={funnel['buyer_year_filtered_count']} >=3 award)",
        bg="#dcfce7",
        border="#15803d",
        dashed=False,
        title_color="#14532d",
    )
    draw_box(
        58.0,
        20.0,
        22,
        9.0,
        "K-Means Clustering (Aktif)",
        "• scikit-learn (Lokal)\n• Spark MLlib (Mode local[*])\nEvaluasi Silhouette & ARI",
        bg="#dcfce7",
        border="#15803d",
        dashed=False,
        title_color="#14532d",
    )
    draw_box(
        58.0,
        9.5,
        22,
        8.5,
        "Graph Analytics (Aktif)",
        "NetworkX (2.891 node, 2.698 edge)\nPageRank, Degree, Komponen\n& Pemetaan Cluster Buyer",
        bg="#dcfce7",
        border="#15803d",
        dashed=False,
        title_color="#14532d",
    )

    ax.text(
        90.5,
        42.5,
        "4. Lanjutan",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#475569",
    )
    draw_box(
        83.5,
        29.5,
        14.5,
        10.5,
        "Orkestrasi\n(Rencana)",
        "Apache Airflow\nPenjadwalan\nbatch otomatis",
        bg="#f8fafc",
        border="#64748b",
        dashed=True,
        title_color="#475569",
    )
    draw_box(
        83.5,
        17.5,
        14.5,
        10.0,
        "Basis Data Saji\n(Rencana)",
        "PostgreSQL &\nNeo4j Graph DB",
        bg="#f8fafc",
        border="#64748b",
        dashed=True,
        title_color="#475569",
    )
    draw_box(
        83.5,
        9.5,
        14.5,
        6.5,
        "Keluaran Aktif",
        "Parquet, JSON,\nGraphML & Grafik",
        bg="#ede9fe",
        border="#6d28d9",
        dashed=False,
        title_color="#4c1d95",
    )

    arrow_kw = dict(arrowstyle="->", color="#334155", lw=1.5, mutation_scale=12)
    ax.annotate("", xy=(24.2, 35.0), xytext=(20.0, 35.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(39.0, 29.5), xytext=(39.0, 30.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(39.0, 18.5), xytext=(39.0, 19.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(57.7, 35.0), xytext=(54.0, 25.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(69.0, 29.5), xytext=(69.0, 30.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(69.0, 18.5), xytext=(69.0, 19.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(83.2, 13.0), xytext=(80.5, 13.0), arrowprops=arrow_kw)

    solid_patch = mpatches.Patch(
        facecolor="#dbeafe",
        edgecolor="#1d4ed8",
        linewidth=1.8,
        label="Komponen Aktif (Garis Penuh)",
    )
    dashed_patch = mpatches.Patch(
        facecolor="#f8fafc",
        edgecolor="#64748b",
        linewidth=1.4,
        linestyle="--",
        label="Komponen Rencana / Data Ilustrasi Lokal (Garis Putus-Putus)",
    )
    ax.legend(
        handles=[solid_patch, dashed_patch],
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.01),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_kmeans_evaluation_chart(res_filtered: dict, out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300)

    ks = [2, 3, 4, 5, 6]
    sk_euclid = [
        res_filtered["sklearn_by_k"][k]["silhouette_euclidean_primary"] for k in ks
    ]
    sp_euclid = [
        res_filtered["spark_by_k"][k]["silhouette_euclidean_primary"] for k in ks
    ]
    sk_sq = [
        res_filtered["sklearn_by_k"][k]["silhouette_sqeuclidean_primary"] for k in ks
    ]
    ari_sk_sp = [res_filtered["spark_by_k"][k]["ari_with_sklearn"] for k in ks]

    ax1.plot(
        ks,
        sk_euclid,
        marker="o",
        lw=2.2,
        color="#2563eb",
        label="scikit-learn Silhouette (Euclidean)",
    )
    ax1.plot(
        ks,
        sp_euclid,
        marker="s",
        lw=2.0,
        linestyle="--",
        color="#dc2626",
        label="Spark MLlib Silhouette (Euclidean)",
    )
    ax1.plot(
        ks,
        sk_sq,
        marker="^",
        lw=1.8,
        linestyle=":",
        color="#059669",
        label="scikit-learn Silhouette (Sq. Euclidean)",
    )
    ax1.plot(
        ks,
        ari_sk_sp,
        marker="D",
        lw=1.8,
        linestyle="-.",
        color="#7c3aed",
        label="Kesesuaian scikit-learn vs Spark (ARI)",
    )

    ax1.axvline(
        4,
        color="#f59e0b",
        linestyle="--",
        alpha=0.8,
        lw=1.5,
        label="Pilihan Model (k=4, ARI=0,9679)",
    )
    ax1.set_xticks(ks)
    ax1.set_ylim(0.1, 1.05)
    ax1.set_xlabel("Jumlah Cluster (k)", fontsize=10)
    ax1.set_ylabel("Nilai Metrik (Silhouette / ARI)", fontsize=10)
    ax1.set_title(
        "(a) Kualitas Pengelompokan & Kesesuaian Antar-Framework",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.legend(loc="lower left", fontsize=7.8, frameon=True)

    sk_train = [res_filtered["sklearn_by_k"][k]["train_time_ms_mean"] for k in ks]
    sp_train = [res_filtered["spark_by_k"][k]["train_time_ms_mean"] for k in ks]
    sk_infer = [res_filtered["sklearn_by_k"][k]["infer_time_ms_mean"] for k in ks]
    sp_infer = [res_filtered["spark_by_k"][k]["infer_time_ms_mean"] for k in ks]

    x = np.arange(len(ks))
    w = 0.2
    ax2.bar(
        x - 1.5 * w,
        sk_train,
        width=w,
        color="#2563eb",
        label="scikit-learn Pelatihan (ms)",
    )
    ax2.bar(
        x - 0.5 * w,
        sk_infer,
        width=w,
        color="#93c5fd",
        label="scikit-learn Inferensi (ms)",
    )
    ax2.bar(
        x + 0.5 * w,
        sp_train,
        width=w,
        color="#dc2626",
        label="Spark MLlib Pelatihan (ms)",
    )
    ax2.bar(
        x + 1.5 * w,
        sp_infer,
        width=w,
        color="#fca5a5",
        label="Spark MLlib Inferensi (ms)",
    )

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"k={k}" for k in ks])
    ax2.set_xlabel("Konfigurasi Jumlah Cluster (k)", fontsize=10)
    ax2.set_ylabel("Durasi Eksekusi (milidetik, skala log)", fontsize=10)
    ax2.set_title(
        "(b) Perbandingan Waktu Eksekusi: scikit-learn vs. Spark MLlib",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=7.8, frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_cluster_profiles_chart(
    profiles_k4: list[dict], filtered_df: pd.DataFrame, out_path: Path
):
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.7), dpi=300)

    c_labels = [f"Cluster {p['cluster_id']}\n(n={p['count']})" for p in profiles_k4]
    hw = [p["mean_share_hardware"] * 100 for p in profiles_k4]
    sw = [p["mean_share_software"] * 100 for p in profiles_k4]
    net = [p["mean_share_network"] * 100 for p in profiles_k4]
    srv = [p["mean_share_services"] * 100 for p in profiles_k4]
    q4 = [p["mean_q4_award_share"] * 100 for p in profiles_k4]

    x = np.arange(len(profiles_k4))
    ax1.bar(x, hw, width=0.52, color="#3b82f6", label="Hardware (%)")
    ax1.bar(x, sw, width=0.52, bottom=hw, color="#10b981", label="Software (%)")
    b_net = np.array(hw) + np.array(sw)
    ax1.bar(x, net, width=0.52, bottom=b_net, color="#f59e0b", label="Jaringan (%)")
    b_srv = b_net + np.array(net)
    ax1.bar(
        x,
        srv,
        width=0.52,
        bottom=b_srv,
        color="#8b5cf6",
        label="Layanan & Keamanan (%)",
    )

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        x,
        q4,
        color="#dc2626",
        marker="D",
        lw=2.2,
        markersize=7,
        label="Proporsi Award Q4 (%)",
    )
    ax1_twin.set_ylim(0, 30)
    ax1_twin.set_ylabel("Proporsi Award Kuartal IV (%)", color="#dc2626", fontsize=9.5)
    ax1_twin.grid(False)

    ax1.set_xticks(x)
    ax1.set_xticklabels(c_labels, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Komposisi Rata-Rata Jenis Pengadaan TI (%)", fontsize=9.5)
    ax1.set_title(
        "(a) Komposisi Kategori TI & Proporsi Q4 per Cluster",
        fontsize=10.5,
        fontweight="bold",
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=7.8,
        frameon=True,
    )

    for cid in sorted(filtered_df["cluster_id"].unique()):
        sub = filtered_df[filtered_df["cluster_id"] == cid]
        ax2.scatter(
            sub["total_procured_value"] / 1e9,
            sub["hhi"],
            s=30 + sub["unique_awards_count"] * 2.2,
            c=CLUSTER_COLORS[int(cid)],
            alpha=0.78,
            edgecolors="white",
            linewidth=0.5,
            label=f"{CLUSTER_SHORT_NAMES[int(cid)]} (HHI={sub['hhi'].mean():.2f})",
        )

    ax2.set_xscale("log")
    ax2.axhline(
        0.25,
        color="#ef4444",
        linestyle="--",
        lw=1.2,
        alpha=0.7,
        label="HHI >= 0,25 (Terkonsentrasi)",
    )
    ax2.set_xlabel(
        "Total Nilai Pengadaan TI Instansi–Tahun (Miliar IDR, skala log)", fontsize=9.5
    )
    ax2.set_ylabel("Indeks HHI (Indikator Deskriptif Pasca-Clustering)", fontsize=9.5)
    ax2.set_title(
        "(b) Distribusi Skala Nilai vs. Konsentrasi HHI Antar-Cluster",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=7.5,
        frameon=True,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _short_org_label(name: str) -> str:
    replacements = [
        ("Kementerian ", "Kemen. "),
        ("Pemerintah Daerah Provinsi ", "Pemprov "),
        ("Pemerintah Daerah Kabupaten ", "Pemkab "),
        ("Pemerintah Daerah Kota ", "Pemkot "),
        ("Kepolisian Negara Republik Indonesia", "Polri"),
        ("Kejaksaan Agung Republik Indonesia", "Kejaksaan RI"),
        ("Badan Siber Dan Sandi Negara", "BSSN"),
        ("Badan Informasi Geospasial", "BIG"),
        ("Badan Meteorologi Klimatologi Dan Geofisika", "BMKG"),
        ("Pusat Pelaporan Dan Analisis Transaksi Keuangan", "PPATK"),
        ("Komisi Pemberantasan Korupsi", "KPK"),
        ("PT Telekomunikasi Indonesia Tbk", "PT Telkom Indonesia"),
    ]
    out = name
    for old, new in replacements:
        out = out.replace(old, new)
    return out[:24]


def generate_clustered_network_chart(G: nx.Graph, out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    supplier_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "supplier"]
    bridge_suppliers = []
    for s in supplier_nodes:
        nbrs = [b for b in G.neighbors(s) if G.nodes[b]["cluster_id"] >= 0]
        clusters = {G.nodes[b]["cluster_id"] for b in nbrs}
        if len(clusters) >= 2 and len(nbrs) >= 3:
            bridge_suppliers.append(s)

    bridge_suppliers.sort(
        key=lambda s: (
            G.nodes[s]["degree"],
            sum(G[s][b]["weight"] for b in G.neighbors(s)),
        ),
        reverse=True,
    )
    top_bridge = bridge_suppliers[:12]

    selected_buyers = set()
    for s in top_bridge:
        for b in G.neighbors(s):
            if G.nodes[b]["cluster_id"] >= 0:
                selected_buyers.add(b)

    for cid in [0, 1, 2, 3]:
        c_buyers = [
            b
            for b in G.nodes()
            if G.nodes[b]["node_type"] == "buyer" and G.nodes[b]["cluster_id"] == cid
        ]
        c_buyers.sort(
            key=lambda b: (G.nodes[b]["pagerank"], G.nodes[b]["degree"]), reverse=True
        )
        for b in c_buyers[:5]:
            selected_buyers.add(b)

    repeat_suppliers = []
    for b in list(selected_buyers):
        nbr_s = [
            s
            for s in G.neighbors(b)
            if s not in top_bridge and G[b][s].get("contract_count", 1) >= 3
        ]
        nbr_s.sort(
            key=lambda s: (G[b][s]["contract_count"], G[b][s]["weight"]), reverse=True
        )
        for s in nbr_s[:1]:
            if s not in repeat_suppliers:
                repeat_suppliers.append(s)
    repeat_suppliers = repeat_suppliers[:10]

    sub_nodes = selected_buyers | set(top_bridge) | set(repeat_suppliers)
    subG = G.subgraph(sub_nodes).copy()
    subG.remove_nodes_from(list(nx.isolates(subG)))

    pos = {}
    quadrant_angles = {
        0: (np.deg2rad(112), np.deg2rad(168)),
        1: (np.deg2rad(12), np.deg2rad(68)),
        2: (np.deg2rad(192), np.deg2rad(248)),
        3: (np.deg2rad(292), np.deg2rad(348)),
    }
    radius_buyer = 3.25
    for cid in [0, 1, 2, 3]:
        c_nodes = sorted(
            [
                n
                for n in subG.nodes()
                if subG.nodes[n]["node_type"] == "buyer"
                and subG.nodes[n]["cluster_id"] == cid
            ],
            key=lambda n: subG.nodes[n]["pagerank"],
            reverse=True,
        )
        if not c_nodes:
            continue
        ang_start, ang_end = quadrant_angles[cid]
        angles = np.linspace(ang_start, ang_end, len(c_nodes))
        for idx, n in enumerate(c_nodes):
            r_offset = radius_buyer + (0.28 if idx % 2 == 1 else -0.15)
            pos[n] = np.array(
                [r_offset * np.cos(angles[idx]), r_offset * np.sin(angles[idx])]
            )

    b_nodes = [n for n in top_bridge if n in subG]
    if b_nodes:
        b_angles = np.linspace(0, 2 * np.pi, len(b_nodes), endpoint=False) + np.deg2rad(
            15
        )
        for idx, s in enumerate(b_nodes):
            rx, ry = (1.15, 0.95) if idx % 2 == 0 else (0.72, 0.60)
            pos[s] = np.array([rx * np.cos(b_angles[idx]), ry * np.sin(b_angles[idx])])

    r_nodes = [n for n in repeat_suppliers if n in subG]
    for idx, s in enumerate(r_nodes):
        nbrs = [b for b in subG.neighbors(s) if b in pos]
        if nbrs:
            b_vec = np.mean([pos[b] for b in nbrs], axis=0)
            norm = np.linalg.norm(b_vec) or 1.0
            unit = b_vec / norm
            tangent = np.array([-unit[1], unit[0]]) * (0.22 if idx % 2 == 0 else -0.22)
            pos[s] = unit * 2.15 + tangent
        else:
            pos[s] = np.array([0.0, 0.0])

    fig, ax = plt.subplots(figsize=(12.2, 8.6), dpi=300)

    edges = list(subG.edges(data=True))
    single_edges = [(u, v) for u, v, d in edges if d.get("contract_count", 1) == 1]
    repeat_edges = [(u, v) for u, v, d in edges if d.get("contract_count", 1) >= 2]
    repeat_widths = [
        1.4 + min(subG[u][v].get("contract_count", 2) * 0.45, 3.2)
        for u, v in repeat_edges
    ]

    nx.draw_networkx_edges(
        subG,
        pos,
        edgelist=single_edges,
        width=0.9,
        edge_color="#cbd5e1",
        alpha=0.65,
        ax=ax,
    )
    nx.draw_networkx_edges(
        subG,
        pos,
        edgelist=repeat_edges,
        width=repeat_widths,
        edge_color="#475569",
        alpha=0.85,
        ax=ax,
    )

    for cid in [0, 1, 2, 3]:
        c_nodelist = [
            n
            for n in subG.nodes()
            if subG.nodes[n]["node_type"] == "buyer"
            and subG.nodes[n]["cluster_id"] == cid
        ]
        if c_nodelist:
            sizes = [250 + min(G.nodes[n]["degree"] * 4.2, 380) for n in c_nodelist]
            nx.draw_networkx_nodes(
                subG,
                pos,
                nodelist=c_nodelist,
                node_color=CLUSTER_COLORS[cid],
                node_shape="o",
                node_size=sizes,
                alpha=0.94,
                edgecolors="white",
                linewidths=1.3,
                label=f"Buyer {CLUSTER_SHORT_NAMES[cid]}",
                ax=ax,
            )

    if b_nodes:
        nx.draw_networkx_nodes(
            subG,
            pos,
            nodelist=b_nodes,
            node_color="#7c3aed",
            node_shape="D",
            node_size=210,
            alpha=0.95,
            edgecolors="white",
            linewidths=1.1,
            label="Supplier Jembatan Lintas-Cluster (Bridge >=3 Buyer)",
            ax=ax,
        )
    if r_nodes:
        nx.draw_networkx_nodes(
            subG,
            pos,
            nodelist=r_nodes,
            node_color="#475569",
            node_shape="^",
            node_size=165,
            alpha=0.90,
            edgecolors="white",
            linewidths=1.0,
            label="Supplier Kontrak Berulang Intra-Cluster (>=3 Award)",
            ax=ax,
        )

    for n in subG.nodes():
        x, y = pos[n]
        ntype = subG.nodes[n]["node_type"]
        lbl = _short_org_label(subG.nodes[n]["name"])
        if ntype == "buyer":
            norm = np.hypot(x, y) or 1.0
            lx, ly = x + (x / norm) * 0.23, y + (y / norm) * 0.18
            ha = "left" if x >= 0 else "right"
            ax.text(
                lx,
                ly,
                lbl,
                fontsize=7.5,
                fontweight="bold",
                color="#0f172a",
                ha=ha,
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )
        elif n in b_nodes:
            ax.text(
                x,
                y - 0.14,
                lbl,
                fontsize=7.0,
                fontweight="bold",
                color="#4c1d95",
                ha="center",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="#f5f3ff",
                    edgecolor="#ddd6fe",
                    alpha=0.88,
                ),
            )
        elif n in r_nodes:
            cnt = max(subG[n][b].get("contract_count", 1) for b in subG.neighbors(n))
            ax.text(
                x,
                y + 0.13,
                f"{lbl} ({cnt}x)",
                fontsize=6.6,
                color="#334155",
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )

    quad_titles = [
        (-2.8, 3.85, "Cluster 0: Skala Besar Software & Jaringan", CLUSTER_COLORS[0]),
        (2.8, 3.85, "Cluster 1: Hardware Rutin Tersebar", CLUSTER_COLORS[1]),
        (-2.8, -3.85, "Cluster 2: Hardware Frekuensi Tinggi", CLUSTER_COLORS[2]),
        (2.8, -3.85, "Cluster 3: Layanan Sistem, Keamanan & Q4", CLUSTER_COLORS[3]),
    ]
    for qx, qy, qtext, qcol in quad_titles:
        ax.text(
            qx,
            qy,
            qtext,
            fontsize=8.8,
            fontweight="bold",
            color=qcol,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#f8fafc",
                edgecolor=qcol,
                lw=1.0,
                alpha=0.9,
            ),
        )

    n_buyers_sub = sum(1 for n in subG.nodes() if subG.nodes[n]["node_type"] == "buyer")
    n_sups_sub = sum(
        1 for n in subG.nodes() if subG.nodes[n]["node_type"] == "supplier"
    )
    ax.set_title(
        f"Subgraf Inti Jaringan Instansi–Penyedia Pengadaan TI (2023) Berdasarkan Cluster K-Means (k=4)\n"
        f"[Subgraf: {subG.number_of_nodes()} Node ({n_buyers_sub} Buyer, {n_sups_sub} Supplier) & {subG.number_of_edges()} Edge; Garis Tebal = Kontrak Berulang]",
        fontsize=10.8,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(-4.65, 4.65)
    ax.set_ylim(-4.35, 4.45)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=True,
        fontsize=8.0,
    )
    ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    print("Building buyer-year data.")
    ds = build_ml_buyer_year_dataset()
    funnel = ds["funnel_summary"]

    print(f"K-Means: {len(ds['filtered_observations'])} filtered observations.")
    res_filtered = run_kmeans_experiments(ds["filtered_observations"])

    print(f"K-Means: {len(ds['all_observations'])} total observations.")
    res_all = run_kmeans_experiments(ds["all_observations"])

    profiles_k4, filtered_df = compute_cluster_profiles(
        ds["filtered_observations"],
        res_filtered["sklearn_by_k"][4]["labels"],
        res_filtered["spark_by_k"][4]["aligned_labels"],
    )
    profiles_k3, _ = compute_cluster_profiles(
        ds["filtered_observations"],
        res_filtered["sklearn_by_k"][3]["labels"],
        res_filtered["spark_by_k"][3]["aligned_labels"],
    )
    profiles_all_k4, _ = compute_cluster_profiles(
        ds["all_observations"],
        res_all["sklearn_by_k"][4]["labels"],
        res_all["spark_by_k"][4]["aligned_labels"],
    )

    pq.write_table(
        pa.Table.from_pandas(filtered_df),
        str(GOLD_DIR / "ml_buyer_year_clustered_k4.parquet"),
        compression="snappy",
    )

    print("Building the cluster graph.")
    G, graph_summary = analyze_clustered_graph(
        SILVER_DIR, filtered_df, ds["all_observations"]
    )
    export_graphml(G, OUTPUT_DIR / "procurement_network_clustered.graphml")

    with open(OUTPUT_DIR / "graph_analytics_summary.json", "w", encoding="utf-8") as f:
        json.dump(graph_summary, f, indent=2, ensure_ascii=False)

    print("Saving charts.")
    f_arch = generate_architecture_diagram_v2(
        CHARTS_DIR / "architecture_diagram_v2.png", funnel
    )
    f_eval = generate_kmeans_evaluation_chart(
        res_filtered, CHARTS_DIR / "kmeans_evaluation_k2_k6.png"
    )
    f_prof = generate_cluster_profiles_chart(
        profiles_k4, filtered_df, CHARTS_DIR / "kmeans_cluster_profiles.png"
    )
    f_net = generate_clustered_network_chart(
        G, CHARTS_DIR / "clustered_buyer_supplier_network.png"
    )

    print(
        f"   Generated charts:\n   - {f_arch}\n   - {f_eval}\n   - {f_prof}\n   - {f_net}"
    )

    out_summary = {
        "funnel": funnel,
        "filtered_experiment": res_filtered,
        "all_experiment": res_all,
        "profiles_filtered_k3": profiles_k3,
        "profiles_filtered_k4": profiles_k4,
        "profiles_all_k4": profiles_all_k4,
        "graph_summary": graph_summary,
    }
    with open(OUTPUT_DIR / "ml_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2, ensure_ascii=False)
    print(f"Results: {OUTPUT_DIR / 'ml_experiment_results.json'}")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from bd_procurement.config import CHARTS_DIR


def set_plot_style():
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.autolayout": True,
            "figure.dpi": 300,
        }
    )


def plot_hhi_distribution(
    buyer_concentration: List[Dict[str, Any]], output_path: Optional[Path] = None
) -> Path:
    set_plot_style()
    out = output_path or (CHARTS_DIR / "buyer_hhi_distribution.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    valid_hhi = [
        b["hhi"]
        for b in buyer_concentration
        if not b.get("is_low_sample") and b.get("hhi") is not None
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if valid_hhi:
        n, bins, patches = ax.hist(
            valid_hhi, bins=15, color="#2563eb", edgecolor="white", alpha=0.85
        )
        ax.axvline(
            0.15,
            color="#f59e0b",
            linestyle="--",
            linewidth=1.5,
            label="Ambang Moderat (0.15)",
        )
        ax.axvline(
            0.25,
            color="#ef4444",
            linestyle="--",
            linewidth=1.5,
            label="Ambang Konsentrasi Tinggi (0.25)",
        )
        ax.set_xlabel("Herfindahl-Hirschman Index (HHI)")
        ax.set_ylabel("Jumlah Instansi (Buyer)")
        ax.set_title(
            "Distribusi Konsentrasi Penyedia TI Berdasarkan HHI (Instansi N >= 3)"
        )
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "Data sampel tidak mencukupi", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_top_suppliers(
    supplier_dominance: List[Dict[str, Any]],
    top_n: int = 10,
    output_path: Optional[Path] = None,
) -> Path:
    set_plot_style()
    out = output_path or (CHARTS_DIR / "top_it_suppliers.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    top_suppliers = supplier_dominance[:top_n]
    names = [s["supplier_name"][:25] for s in top_suppliers][::-1]
    values = [(s["total_won_value"] / 1e9) for s in top_suppliers][::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(names, values, color="#0284c7", edgecolor="none", height=0.65)
    ax.set_xlabel("Total Nilai Award Pengadaan TI (Miliar IDR)")
    ax.set_title(f"Top {top_n} Penyedia TI Berdasarkan Akumulasi Nilai Award (2023)")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"Rp{width:.1f} M",
            va="center",
            ha="left",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_seasonality(
    tenders: List[Dict[str, Any]], output_path: Optional[Path] = None
) -> Path:
    set_plot_style()
    out = output_path or (CHARTS_DIR / "procurement_seasonality_q4.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    months = range(1, 13)
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Mei",
        "Jun",
        "Jul",
        "Ags",
        "Sep",
        "Okt",
        "Nov",
        "Des",
    ]
    all_tenders_by_m = {m: 0 for m in months}
    it_tenders_by_m = {m: 0 for m in months}

    for t in tenders:
        if not t.get("date_valid") or not t.get("pub_month"):
            continue
        m = t["pub_month"]
        if 1 <= m <= 12:
            all_tenders_by_m[m] += 1
            if t.get("is_it"):
                it_tenders_by_m[m] += 1

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax2 = ax1.twinx()

    x = list(months)
    ax1.bar(
        x,
        [all_tenders_by_m[m] for m in x],
        color="#94a3b8",
        alpha=0.5,
        label="Semua Pengadaan (Total)",
    )
    ax2.plot(
        x,
        [it_tenders_by_m[m] for m in x],
        color="#dc2626",
        marker="o",
        linewidth=2,
        label="Pengadaan TI",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(month_names)
    ax1.set_xlabel("Bulan Publikasi")
    ax1.set_ylabel("Jumlah Seluruh Tender", color="#475569")
    ax2.set_ylabel("Jumlah Tender TI", color="#dc2626")
    ax1.set_title(
        "Pola Temporal Publikasi Tender: Indikator Year-End Rush (Kuartal IV)"
    )

    ax1.axvspan(9.5, 12.5, color="#fee2e2", alpha=0.4, label="Kuartal IV (Okt-Des)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_it_subcategories(
    tenders: List[Dict[str, Any]], output_path: Optional[Path] = None
) -> Path:
    set_plot_style()
    out = output_path or (CHARTS_DIR / "it_subcategories_breakdown.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    from collections import Counter

    subcats = Counter(t["it_primary_subcat"] for t in tenders if t.get("is_it"))

    labels = [
        f"{k.replace('_', ' ').title()}\n({v:,} paket)" for k, v in subcats.items()
    ]
    vals = list(subcats.values())
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(7, 5))
    if vals:
        wedges, texts, autotexts = ax.pie(
            vals,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors[: len(vals)],
            textprops=dict(color="#1f2937", fontsize=9),
        )
        for at in autotexts:
            at.set_color("white")
            at.set_weight("bold")

        centre_circle = plt.Circle((0, 0), 0.60, fc="white")
        fig.gca().add_artist(centre_circle)
        ax.set_title("Distribusi Subkategori Pengadaan Teknologi Informasi (2023)")
    else:
        ax.text(0.5, 0.5, "Tidak ada data subkategori TI", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_network_subgraph(
    projected_graph: nx.Graph, top_n_nodes: int = 25, output_path: Optional[Path] = None
) -> Path:
    set_plot_style()
    out = output_path or (CHARTS_DIR / "buyer_supplier_network_subgraph.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(projected_graph) == 0:
        return out

    degrees = dict(projected_graph.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[
        :top_n_nodes
    ]
    sub_nodes = set(n for n, d in sorted_nodes)
    subG = projected_graph.subgraph(sub_nodes)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(subG, k=0.5, seed=42)

    buyer_nodes = [n for n in subG.nodes() if subG.nodes[n].get("node_type") == "buyer"]
    supplier_nodes = [
        n for n in subG.nodes() if subG.nodes[n].get("node_type") == "supplier"
    ]

    nx.draw_networkx_nodes(
        subG,
        pos,
        nodelist=buyer_nodes,
        node_color="#3b82f6",
        node_size=350,
        alpha=0.9,
        label="Buyer / Instansi",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        subG,
        pos,
        nodelist=supplier_nodes,
        node_color="#10b981",
        node_size=250,
        alpha=0.9,
        label="Supplier / Penyedia",
        ax=ax,
    )
    nx.draw_networkx_edges(subG, pos, edge_color="#cbd5e1", width=1.2, alpha=0.7, ax=ax)

    labels = {n: subG.nodes[n].get("name", n)[:16] for n in sub_nodes}
    nx.draw_networkx_labels(
        subG, pos, labels=labels, font_size=7, font_family="sans-serif", ax=ax
    )

    ax.set_title("Subgraf Jaringan Relasi Buyer-Supplier Pengadaan TI Terkoneksi")
    ax.legend(loc="upper right")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out

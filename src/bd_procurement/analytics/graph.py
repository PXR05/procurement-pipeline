from pathlib import Path
from typing import Any, Dict, List

import networkx as nx


def build_bipartite_procurement_graph(
    tenders: List[Dict[str, Any]],
    awards: List[Dict[str, Any]],
    bridge: List[Dict[str, Any]],
    orgs: List[Dict[str, Any]],
    filter_it_only: bool = True,
) -> nx.DiGraph:
    G = nx.DiGraph()

    for o in orgs:
        G.add_node(
            o["org_key"],
            node_type=o["org_type"],
            canonical_name=o.get("canonical_name", ""),
            raw_name=o.get("raw_name", ""),
            legal_form=o.get("legal_form", ""),
        )

    for t in tenders:
        if filter_it_only and not t.get("is_it"):
            continue

        t_key = t["tender_key"]
        G.add_node(
            t_key,
            node_type="tender",
            title=t.get("tender_title", "")[:80],
            status=t.get("tender_status", ""),
            year=t.get("pub_year"),
            primary_subcat=t.get("it_primary_subcat", ""),
        )

        b_key = t.get("buyer_key")
        if b_key and G.has_node(b_key):
            G.add_edge(b_key, t_key, relationship="ISSUES")

        subcat = t.get("it_primary_subcat")
        if subcat:
            cat_key = f"cat_{subcat}"
            if not G.has_node(cat_key):
                G.add_node(cat_key, node_type="category", name=subcat)
            G.add_edge(t_key, cat_key, relationship="IN_CATEGORY")

    for br in bridge:
        if filter_it_only and not br.get("is_it"):
            continue

        t_key = br["tender_key"]
        s_key = br["supplier_key"]
        amt = br.get("allocated_amount") or 0.0

        if G.has_node(t_key) and G.has_node(s_key):
            G.add_edge(t_key, s_key, relationship="AWARDED_TO", amount=amt)

    return G


def build_projected_buyer_supplier_graph(
    pair_metrics: List[Dict[str, Any]], orgs: List[Dict[str, Any]]
) -> nx.Graph:
    G = nx.Graph()
    org_dict = {o["org_key"]: o for o in orgs}

    for pm in pair_metrics:
        b_key = pm["buyer_key"]
        s_key = pm["supplier_key"]
        val = pm["total_won_value"]
        cnt = pm["awards_won_count"]

        if not G.has_node(b_key):
            b_info = org_dict.get(b_key, {})
            G.add_node(
                b_key, node_type="buyer", name=b_info.get("canonical_name", b_key)
            )

        if not G.has_node(s_key):
            s_info = org_dict.get(s_key, {})
            G.add_node(
                s_key, node_type="supplier", name=s_info.get("canonical_name", s_key)
            )

        G.add_edge(b_key, s_key, weight=val, contract_count=cnt)

    return G


def run_graph_analytics(projected_graph: nx.Graph) -> Dict[str, Any]:
    if len(projected_graph) == 0:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "pagerank_top10": [],
            "degree_centrality_top10": [],
            "connected_components_count": 0,
        }

    deg_centrality = nx.degree_centrality(projected_graph)

    try:
        pagerank = nx.pagerank(projected_graph, weight="contract_count")
    except Exception:
        pagerank = deg_centrality

    components = list(nx.connected_components(projected_graph))

    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    sorted_deg = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)

    top10_pr = []
    for node_id, pr_val in sorted_pr[:10]:
        attrs = projected_graph.nodes[node_id]
        top10_pr.append(
            {
                "node_key": node_id,
                "name": attrs.get("name", node_id),
                "node_type": attrs.get("node_type", "unknown"),
                "pagerank": round(pr_val, 6),
            }
        )

    top10_deg = []
    for node_id, d_val in sorted_deg[:10]:
        attrs = projected_graph.nodes[node_id]
        top10_deg.append(
            {
                "node_key": node_id,
                "name": attrs.get("name", node_id),
                "node_type": attrs.get("node_type", "unknown"),
                "degree_centrality": round(d_val, 6),
            }
        )

    return {
        "total_nodes": projected_graph.number_of_nodes(),
        "total_edges": projected_graph.number_of_edges(),
        "pagerank_top10": top10_pr,
        "degree_centrality_top10": top10_deg,
        "connected_components_count": len(components),
    }


def export_graphml(graph: nx.Graph, filepath: Path) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, str(filepath))
    return filepath

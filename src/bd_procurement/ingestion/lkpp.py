import csv
from pathlib import Path
from typing import Any, Dict, List


def generate_sample_lkpp_aggregates(target_csv: Path) -> Path:
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    sample_data = [
        {
            "resource_id": "LKPP-AGG-2023-NASIONAL",
            "year": 2023,
            "category": "Pengadaan Barang",
            "total_tenders": 45120,
            "total_value_idr": 48200000000000.0,
            "source_label": "LKPP Open Data 2023",
        },
        {
            "resource_id": "LKPP-AGG-2023-NASIONAL",
            "year": 2023,
            "category": "Jasa Lainnya",
            "total_tenders": 28450,
            "total_value_idr": 31500000000000.0,
            "source_label": "LKPP Open Data 2023",
        },
        {
            "resource_id": "LKPP-AGG-2023-NASIONAL",
            "year": 2023,
            "category": "Jasa Konsultansi",
            "total_tenders": 14200,
            "total_value_idr": 9800000000000.0,
            "source_label": "LKPP Open Data 2023",
        },
        {
            "resource_id": "LKPP-AGG-2023-TI-ESTIMATE",
            "year": 2023,
            "category": "Teknologi Informasi (Perkiraan)",
            "total_tenders": 6500,
            "total_value_idr": 7200000000000.0,
            "source_label": "LKPP Open Data 2023 Subset TI",
        },
    ]

    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_data[0].keys()))
        writer.writeheader()
        writer.writerows(sample_data)

    return target_csv


def load_lkpp_aggregates(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    records = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "resource_id": row["resource_id"],
                    "year": int(row["year"]),
                    "category": row["category"],
                    "total_tenders": int(row["total_tenders"]),
                    "total_value_idr": float(row["total_value_idr"]),
                    "source_label": row["source_label"],
                }
            )
    return records

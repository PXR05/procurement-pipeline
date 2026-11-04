import csv
from pathlib import Path
from typing import Any, Dict, List


def generate_sample_djpk_fiscal_data(target_csv: Path) -> Path:
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    sample_records = [
        {
            "region_code": "31.00",
            "region_name": "Pemerintah Provinsi DKI Jakarta",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Barang dan Jasa",
            "budget_allocated": 28500000000000,
            "budget_realized": 26800000000000,
        },
        {
            "region_code": "31.00",
            "region_name": "Pemerintah Provinsi DKI Jakarta",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Modal Peralatan dan Mesin (TI)",
            "budget_allocated": 1250000000000,
            "budget_realized": 1180000000000,
        },
        {
            "region_code": "32.00",
            "region_name": "Pemerintah Provinsi Jawa Barat",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Barang dan Jasa",
            "budget_allocated": 15200000000000,
            "budget_realized": 14100000000000,
        },
        {
            "region_code": "32.00",
            "region_name": "Pemerintah Provinsi Jawa Barat",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Modal Peralatan dan Mesin (TI)",
            "budget_allocated": 640000000000,
            "budget_realized": 610000000000,
        },
        {
            "region_code": "33.00",
            "region_name": "Pemerintah Provinsi Jawa Tengah",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Barang dan Jasa",
            "budget_allocated": 11800000000000,
            "budget_realized": 10900000000000,
        },
        {
            "region_code": "33.00",
            "region_name": "Pemerintah Provinsi Jawa Tengah",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Modal Peralatan dan Mesin (TI)",
            "budget_allocated": 480000000000,
            "budget_realized": 450000000000,
        },
        {
            "region_code": "35.00",
            "region_name": "Pemerintah Provinsi Jawa Timur",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Barang dan Jasa",
            "budget_allocated": 14600000000000,
            "budget_realized": 13900000000000,
        },
        {
            "region_code": "35.00",
            "region_name": "Pemerintah Provinsi Jawa Timur",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Modal Peralatan dan Mesin (TI)",
            "budget_allocated": 590000000000,
            "budget_realized": 560000000000,
        },
        {
            "region_code": "32.73",
            "region_name": "Pemerintah Kota Bandung",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Barang dan Jasa",
            "budget_allocated": 3200000000000,
            "budget_realized": 3050000000000,
        },
        {
            "region_code": "32.73",
            "region_name": "Pemerintah Kota Bandung",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Modal Peralatan dan Mesin (TI)",
            "budget_allocated": 165000000000,
            "budget_realized": 158000000000,
        },
        {
            "region_code": "35.78",
            "region_name": "Pemerintah Kota Surabaya",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Barang dan Jasa",
            "budget_allocated": 4800000000000,
            "budget_realized": 4600000000000,
        },
        {
            "region_code": "35.78",
            "region_name": "Pemerintah Kota Surabaya",
            "fiscal_year": 2023,
            "apbd_version": "APBD-P",
            "account_name": "Belanja Modal Peralatan dan Mesin (TI)",
            "budget_allocated": 220000000000,
            "budget_realized": 212000000000,
        },
    ]

    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_records[0].keys()))
        writer.writeheader()
        writer.writerows(sample_records)

    return target_csv


def load_djpk_fiscal_data(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    records = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "region_code": row["region_code"],
                    "region_name": row["region_name"],
                    "fiscal_year": int(row["fiscal_year"]),
                    "apbd_version": row["apbd_version"],
                    "account_name": row["account_name"],
                    "budget_allocated": float(row["budget_allocated"]),
                    "budget_realized": float(row["budget_realized"]),
                }
            )
    return records

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdf

from bd_procurement.config import MANIFESTS_DIR, RAW_DIR
from bd_procurement.ingestion.manifest import create_manifest


def extract_findings_from_pdf(
    pdf_path: Path,
    report_id: str,
    audit_year: int,
    keywords: Optional[List[str]] = None,
    entity_name: Optional[str] = None,
    pages: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if keywords is None:
        keywords = [
            "pengadaan",
            "teknologi informasi",
            "komputer",
            "aplikasi",
            "jaringan",
            "kontrak",
            "hps",
            "penyedia",
            "kelebihan bayar",
            "denda keterlambatan",
            "spesifikasi",
            "aset tak berwujud",
            "lisensi",
        ]

    findings = []
    reader = pypdf.PdfReader(str(pdf_path))
    selected = (
        sorted(set(pages))
        if pages is not None
        else list(range(1, len(reader.pages) + 1))
    )
    if not selected or any(page < 1 or page > len(reader.pages) for page in selected):
        raise ValueError(
            f"PDF page numbers must be between 1 and {len(reader.pages)}: {pdf_path.name}"
        )
    text_pages = 0

    for page_idx in selected:
        text = reader.pages[page_idx - 1].extract_text() or ""
        text_pages += bool(text.strip())

        text_lower = text.lower()
        matched_kw = [kw for kw in keywords if kw in text_lower]

        if matched_kw:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            relevant_sentences = [
                s.strip()
                for s in sentences
                if any(kw in s.lower() for kw in matched_kw)
            ]
            quote = (
                " [...] ".join(relevant_sentences[:3])
                if relevant_sentences
                else text[:300]
            )

            issue_type = "Pengendalian Intern Pengadaan"
            if "kelebihan bayar" in text_lower or "denda" in text_lower:
                issue_type = "Ketidakpatuhan terhadap Kontrak/Spesifikasi"
            elif "hps" in text_lower or "evaluasi" in text_lower:
                issue_type = "Penyusunan HPS & Evaluasi Penawaran"
            elif "aset" in text_lower or "lisensi" in text_lower:
                issue_type = "Pengelolaan & Pencatatan Aset TI"

            findings.append(
                {
                    "report_id": report_id,
                    "audit_year": audit_year,
                    "filename": pdf_path.name,
                    "entity_name": entity_name,
                    "page_number": page_idx,
                    "matched_keywords": matched_kw,
                    "issue_type": issue_type,
                    "quote": quote[:500].strip(),
                    "needs_review": True,
                }
            )

    if not text_pages:
        raise ValueError(
            f"No extractable text in {pdf_path.name}. Apply OCR before ingestion."
        )
    return findings


def extract_bpk_sources(index_path: Path) -> List[Dict[str, Any]]:
    sources = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if not isinstance(sources, list) or not sources:
        raise ValueError("bpk_sources.json must be a non-empty array.")
    findings = []
    copied = {}
    for source in sources:
        for field in ("pdf", "report_id", "audit_year", "entity_name", "pages"):
            if not source.get(field):
                raise ValueError(f"BPK source is missing {field}.")
        pdf_path = (index_path.parent / source["pdf"]).resolve()
        if not pdf_path.is_relative_to(index_path.parent.resolve()):
            raise ValueError("BPK PDFs must be inside the context folder.")
        if pdf_path.name in copied and copied[pdf_path.name] != pdf_path:
            raise ValueError(f"Duplicate PDF filename: {pdf_path.name}")
        findings.extend(
            extract_findings_from_pdf(
                pdf_path,
                source["report_id"],
                int(source["audit_year"]),
                entity_name=source["entity_name"],
                pages=source["pages"],
            )
        )
        if pdf_path.name not in copied:
            destination = RAW_DIR / "bpk" / "pdf" / pdf_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if pdf_path != destination.resolve():
                shutil.copy2(pdf_path, destination)
            create_manifest(
                source_id="bpk_pdf",
                source_url="user-provided",
                filepath=destination,
                period=str(source["audit_year"]),
                format_type="PDF",
                license_type="User supplied",
                extra_metadata={"sample_data": False, "needs_review": True},
                output_dir=MANIFESTS_DIR,
            )
            copied[pdf_path.name] = pdf_path
    destination_index = RAW_DIR / "bpk" / "bpk_sources.json"
    saved_sources = [
        dict(source, pdf=f"pdf/{Path(source['pdf']).name}") for source in sources
    ]
    destination_index.write_text(
        json.dumps(saved_sources, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return findings


def generate_sample_bpk_audit_findings(target_json: Path) -> List[Dict[str, Any]]:
    target_json.parent.mkdir(parents=True, exist_ok=True)
    sample_data = [
        {
            "report_id": "IHPS-BPK-2023-II",
            "audit_year": 2023,
            "filename": "IHPS_II_Tahun_2023.pdf",
            "entity_name": "Kementerian Komunikasi dan Informatika",
            "page_number": 142,
            "issue_type": "Penyusunan HPS & Evaluasi Penawaran",
            "quote": "Penyusunan HPS paket pengadaan perangkat komputasi awan dan jaringan belum sepenuhnya mengacu pada survei harga pasar yang andal.",
            "matched_keywords": ["pengadaan", "hps", "jaringan"],
            "needs_review": True,
        },
        {
            "report_id": "IHPS-BPK-2023-II",
            "audit_year": 2023,
            "filename": "IHPS_II_Tahun_2023.pdf",
            "entity_name": "Pemerintah Provinsi Jawa Barat",
            "page_number": 218,
            "issue_type": "Ketidakpatuhan terhadap Kontrak/Spesifikasi",
            "quote": "Keterlambatan penyelesaian pekerjaan pemeliharaan infrastruktur pusat data belum dikenakan denda keterlambatan sesuai ketentuan kontrak.",
            "matched_keywords": ["pemeliharaan", "pusat data", "denda keterlambatan"],
            "needs_review": True,
        },
        {
            "report_id": "IHPS-BPK-2023-I",
            "audit_year": 2023,
            "filename": "IHPS_I_Tahun_2023.pdf",
            "entity_name": "Pemerintah Kota Bandung",
            "page_number": 89,
            "issue_type": "Pengelolaan & Pencatatan Aset TI",
            "quote": "Pengadaan lisensi perangkat lunak dan aplikasi layanan publik belum didukung oleh berita acara serah terima dan dokumentasi lisensi terpusat.",
            "matched_keywords": ["lisensi", "aplikasi", "perangkat lunak"],
            "needs_review": True,
        },
    ]
    import json

    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    return sample_data

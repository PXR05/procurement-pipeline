import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_manifest(
    source_id: str,
    source_url: str,
    filepath: Path,
    period: str,
    format_type: str,
    license_type: str,
    record_count: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if not filepath.exists():
        raise FileNotFoundError(f"File not found for manifest creation: {filepath}")

    file_bytes = filepath.stat().st_size
    sha256_hash = compute_sha256(filepath)
    now_iso = datetime.now(timezone.utc).isoformat()

    from bd_procurement.config import PROJECT_ROOT

    try:
        rel_path_str = filepath.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        rel_path_str = filepath.as_posix()

    manifest_data = {
        "source_id": source_id,
        "source_url": source_url,
        "filename": filepath.name,
        "relative_path": rel_path_str,
        "retrieved_at": now_iso,
        "period": period,
        "format": format_type,
        "bytes": file_bytes,
        "sha256": sha256_hash,
        "license": license_type,
        "record_count": record_count,
        "validation_status": "VALID",
        "metadata": extra_metadata or {},
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = (
            output_dir / f"manifest_{source_id}_{period}_{filepath.stem}.json"
        )
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        manifest_data["manifest_saved_to"] = str(manifest_file)

    return manifest_data


def verify_manifest(manifest_path: Path) -> bool:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_file = Path(data["relative_path"])
    if not target_file.is_absolute():
        from bd_procurement.config import PROJECT_ROOT

        target_file = PROJECT_ROOT / target_file
    if not target_file.exists():
        return False

    current_size = target_file.stat().st_size
    current_sha256 = compute_sha256(target_file)

    return current_size == data["bytes"] and current_sha256 == data["sha256"]

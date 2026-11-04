import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Run procurement analysis.")
    parser.add_argument("--raw-archive", type=Path, required=True)
    context = parser.add_mutually_exclusive_group(required=True)
    context.add_argument("--context-dir", type=Path)
    context.add_argument("--sample-context", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    archive = args.raw_archive.resolve()
    if not archive.is_file():
        parser.error(f"Archive not found: {archive}")
    if not archive.name.endswith(".jsonl.gz"):
        parser.error("Expected a 2023 OCDS .jsonl.gz archive.")
    java_home = os.environ.get("JAVA_HOME")
    java = (
        Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if java_home
        else shutil.which("java")
    )
    if not java or not Path(java).is_file():
        parser.error("Java not found. Install JDK 17+ and set JAVA_HOME.")
    java_version = subprocess.run(
        [str(java), "-version"], capture_output=True, text=True, check=True
    )
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    env = dict(
        os.environ,
        BD_DATA_DIR=str(data_dir),
        BD_OUTPUT_DIR=str(output_dir),
        PYTHONHASHSEED="0",
        PYTHONIOENCODING="utf-8",
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    command = [
        sys.executable,
        "-m",
        "bd_procurement.cli",
        "run-all",
        "--raw-archive",
        str(archive),
        "--recompute-silver",
    ]
    if args.context_dir is not None:
        command += ["--context-dir", str(args.context_dir.resolve())]
    else:
        command += ["--sample-context"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run.json"
    report = {
        "status": "running",
        "input": {"filename": archive.name, "sha256": sha256(archive)},
        "context": "sample" if args.sample_context else "supplied",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "java": (java_version.stderr or java_version.stdout).strip(),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "pyarrow",
                "matplotlib",
                "networkx",
                "scikit-learn",
                "scipy",
                "pyspark",
            )
        },
        "lock_sha256": sha256(ROOT / "uv.lock"),
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for directory in ("src", "scripts")
            for path in sorted((ROOT / directory).rglob("*.py"))
        },
        "k_values": [2, 3, 4, 5, 6],
        "seeds": [42, 7, 123, 2024, 999],
        "primary_seed": 42,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        for script in ("run_analytics.py", "integrate_multi_source.py"):
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script)],
                cwd=ROOT,
                env=env,
                check=True,
            )
        report["context_sha256"] = {
            path.relative_to(data_dir / "raw").as_posix(): sha256(path)
            for directory in ("djpk", "bpk", "lkpp")
            for path in sorted((data_dir / "raw" / directory).rglob("*"))
            if path.is_file()
        }
        report["status"] = "complete"
    except subprocess.CalledProcessError as exc:
        report["status"] = "failed"
        report["failed_command"] = exc.cmd
        report["exit_code"] = exc.returncode
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Run details: {report_path}")


if __name__ == "__main__":
    main()

import os
import subprocess
from pathlib import PurePosixPath
from urllib.parse import urlparse

from bd_procurement.config import (
    GOLD_DIR,
    MANIFESTS_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    SILVER_DIR,
)
from bd_procurement.ingestion.hdfs_client import HDFSConnector


def sync_to_hdfs():
    docker = [
        "docker",
        "compose",
        "-f",
        str(PROJECT_ROOT / "compose.yaml"),
        "exec",
        "-T",
        "namenode",
    ]
    root = PurePosixPath(os.environ.get("HDFS_ROOT", "/procurement"))
    endpoint = os.environ.get("HDFS_NAMENODE")
    connector = None
    if endpoint:
        url = urlparse(endpoint)
        connector = HDFSConnector(
            url.hostname, url.port or 9870, os.environ.get("HDFS_USER", "root")
        )
        if not connector.is_namenode_healthy():
            raise ConnectionError(f"NameNode unavailable: {endpoint}")
    uploaded = 0
    for source, layer in (
        (RAW_DIR, "raw"),
        (SILVER_DIR, "silver"),
        (GOLD_DIR, "gold"),
        (MANIFESTS_DIR, "manifests"),
    ):
        if not source.is_dir():
            raise FileNotFoundError(f"Layer not found: {source}")
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            target = root / layer / path.relative_to(source).as_posix()
            if connector:
                if not connector.create_directory(
                    str(target.parent)
                ) or not connector.upload_file(path, str(target)):
                    raise OSError(f"HDFS upload failed: {path} -> {target}")
            else:
                subprocess.run(
                    [
                        *docker,
                        "hdfs",
                        "dfs",
                        "-mkdir",
                        "-p",
                        str(target.parent),
                    ],
                    check=True,
                )
                with path.open("rb") as stream:
                    subprocess.run(
                        [
                            *docker,
                            "hdfs",
                            "dfs",
                            "-put",
                            "-f",
                            "-",
                            str(target),
                        ],
                        stdin=stream,
                        check=True,
                    )
            uploaded += 1
    print(f"Uploaded {uploaded} files to {root}.")
    return True


if __name__ == "__main__":
    sync_to_hdfs()

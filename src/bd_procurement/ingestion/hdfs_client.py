import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlencode


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HDFSConnector:
    def __init__(
        self,
        namenode_host: str = "localhost",
        namenode_port: int = 9870,
        user: str = "root",
    ):
        self.base_url = f"http://{namenode_host}:{namenode_port}/webhdfs/v1"
        self.user = user

    def is_namenode_healthy(self) -> bool:
        try:
            url = f"{self.base_url.split('/webhdfs')[0]}/jmx"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def create_directory(self, hdfs_path: str) -> bool:
        url = f"{self.base_url}{quote(hdfs_path, safe='/')}?{urlencode({'op': 'MKDIRS', 'user.name': self.user})}"
        req = urllib.request.Request(url, method="PUT")
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                return bool(data.get("boolean"))
        except Exception:
            return False

    def upload_file(self, local_path: Path, hdfs_path: str) -> bool:
        if not local_path.exists():
            return False

        init_url = f"{self.base_url}{quote(hdfs_path, safe='/')}?{urlencode({'op': 'CREATE', 'overwrite': 'true', 'user.name': self.user})}"
        req = urllib.request.Request(init_url, method="PUT", data=b"")
        try:
            try:
                with urllib.request.build_opener(NoRedirect).open(req, timeout=30):
                    return False
            except urllib.error.HTTPError as e:
                if e.code == 307:
                    datanode_url = e.headers.get("Location")
                    if not datanode_url:
                        return False

                    with open(local_path, "rb") as stream:
                        put_req = urllib.request.Request(
                            datanode_url,
                            data=stream,
                            method="PUT",
                            headers={
                                "Content-Type": "application/octet-stream",
                                "Content-Length": str(local_path.stat().st_size),
                            },
                        )
                        with urllib.request.urlopen(put_req, timeout=300) as dn_resp:
                            return dn_resp.status == 201
            return False
        except Exception:
            return False

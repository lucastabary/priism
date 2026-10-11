"""Key-value storage behind the job queue: S3-compatible (RunPod volume, R2...) or a local folder (tests).

Keys are relative paths (``queue/pending/<id>.json``). Every S3 call retries with backoff, then raises
``StoreError``: callers decide whether to wait (the agent keeps its job running and syncs later).
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse


class StoreError(RuntimeError):
    pass


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, key: str, data: bytes) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def get(self, key: str) -> bytes | None:
        p = self.root / key
        return p.read_bytes() if p.exists() else None

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)).replace("\\", "/") for p in base.rglob("*")
                      if p.is_file() and not p.name.endswith(".tmp"))

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)

    def upload_file(self, key: str, path: str | Path) -> None:
        self.put(key, Path(path).read_bytes())

    def download_file(self, key: str, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.get(key) or b"")


class S3Store:
    """``s3://bucket/prefix`` on any S3-compatible endpoint; credentials from an AWS profile or the
    standard AWS environment variables."""

    def __init__(self, url: str, endpoint: str | None = None, region: str | None = None,
                 profile: str | None = None, retries: int = 5):
        import boto3

        u = urlparse(url)
        self.bucket, self.prefix = u.netloc, u.path.strip("/")
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.s3 = session.client("s3", endpoint_url=endpoint, region_name=region)
        self.retries = retries

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _call(self, fn, *a, **kw):
        delay = 2.0
        for attempt in range(self.retries):
            try:
                return fn(*a, **kw)
            except Exception as e:  # network down, endpoint hiccup, throttling
                if getattr(e, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                    raise
                if attempt == self.retries - 1:
                    raise StoreError(str(e)) from e
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def put(self, key: str, data: bytes) -> None:
        self._call(self.s3.put_object, Bucket=self.bucket, Key=self._k(key), Body=data)

    def get(self, key: str) -> bytes | None:
        try:
            return self._call(self.s3.get_object, Bucket=self.bucket, Key=self._k(key))["Body"].read()
        except Exception as e:
            if getattr(e, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def list(self, prefix: str) -> list[str]:
        out, token, base = [], None, self._k(prefix)
        while True:
            kw = {"Bucket": self.bucket, "Prefix": base}
            if token:
                kw["ContinuationToken"] = token
            r = self._call(self.s3.list_objects_v2, **kw)
            out += [o["Key"][len(self.prefix) + 1 if self.prefix else 0:] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return sorted(out)
            token = r["NextContinuationToken"]

    def delete(self, key: str) -> None:
        self._call(self.s3.delete_object, Bucket=self.bucket, Key=self._k(key))

    def upload_file(self, key: str, path: str | Path) -> None:
        self._call(self.s3.upload_file, str(path), self.bucket, self._k(key))

    def download_file(self, key: str, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._call(self.s3.download_file, self.bucket, self._k(key), str(path))


def open_store(url: str, endpoint: str | None = None, region: str | None = None, profile: str | None = None):
    """``s3://bucket/prefix`` or a local folder (``file:///path`` or a plain path)."""
    if url.startswith("s3://"):
        return S3Store(url, endpoint, region, profile)
    return LocalStore(url[len("file://"):] if url.startswith("file://") else url)

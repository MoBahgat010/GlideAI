import shutil
from pathlib import Path


class ObjectStorage:
    """
    Idempotent file storage — uses S3 when an s3_uri is supplied, otherwise disk.
    """

    def __init__(self, s3_uri: str | None = None, local_dir: str = "uploads"):
        self.s3_uri = s3_uri
        self.local_dir = Path(local_dir)

        if s3_uri:
            import boto3
            self._s3 = boto3.client("s3")
            parts = s3_uri.replace("s3://", "").split("/", 1)
            self._bucket = parts[0]
            self._prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
        else:
            self.local_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    # ── public API ────────────────────────────────────────────────────────────

    def exists(self, key: str) -> bool:
        if self.s3_uri:
            try:
                self._s3.head_object(Bucket=self._bucket, Key=self._full_key(key))
                return True
            except Exception:
                return False
        return (self.local_dir / key).exists()

    def upload(self, file_obj, key: str) -> str:
        """Idempotent: skips upload if the key already exists. Returns key."""
        if self.exists(key):
            return key
        if self.s3_uri:
            self._s3.upload_fileobj(file_obj, self._bucket, self._full_key(key))
        else:
            dest = self.local_dir / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(file_obj, "read"):
                with open(dest, "wb") as fh:
                    shutil.copyfileobj(file_obj, fh)
            else:
                shutil.copy(str(file_obj), dest)
        return key

    def download(self, key: str, local_path: str | Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if self.s3_uri:
            self._s3.download_file(self._bucket, self._full_key(key), str(local_path))
        else:
            shutil.copy(self.local_dir / key, local_path)
        return local_path
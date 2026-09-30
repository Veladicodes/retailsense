"""Thin S3 wrapper. Works with AWS S3 and S3-compatible stores (MinIO) via an optional endpoint URL."""
from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError


class S3Store:
    def __init__(self, bucket: str, endpoint: str | None = None, client=None):
        self.bucket = bucket
        self.endpoint = endpoint
        self._client = client or boto3.client("s3", endpoint_url=endpoint)

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
            return
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket", "NotFound"):
                raise
        region = self._client.meta.region_name
        kwargs = {"Bucket": self.bucket}
        if region and region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        self._client.create_bucket(**kwargs)

    def put_file(self, path: str | Path, key: str) -> None:
        self._client.upload_file(str(path), self.bucket, key)

    def get_file(self, key: str, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(path))

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        for page in self._client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(keys)

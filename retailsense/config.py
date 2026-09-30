"""Central, environment-driven configuration. No hard-coded machine paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load .env when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    data_dir: Path
    db_url: str
    s3_bucket: str
    s3_endpoint: str | None
    top_n_skus: int
    holdout_weeks: int
    validation_weeks: int
    seed: int = 42


def load_config() -> Config:
    data_dir = Path(os.getenv("RETAILSENSE_DATA_DIR", "data"))
    return Config(
        data_dir=data_dir,
        db_url=os.getenv("RETAILSENSE_DB_URL", f"sqlite:///{data_dir.as_posix()}/retailsense.db"),
        s3_bucket=os.getenv("RETAILSENSE_S3_BUCKET", "retailsense"),
        s3_endpoint=os.getenv("RETAILSENSE_S3_ENDPOINT") or None,
        top_n_skus=int(os.getenv("RETAILSENSE_TOP_N_SKUS", "100")),
        holdout_weeks=int(os.getenv("RETAILSENSE_HOLDOUT_WEEKS", "13")),
        validation_weeks=int(os.getenv("RETAILSENSE_VALIDATION_WEEKS", "13")),
    )

"""Database access: engine factory and schema bootstrap (SQLite or PostgreSQL)."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, text

from retailsense.config import PROJECT_ROOT, load_config

SQL_DIR = PROJECT_ROOT / "sql"


def get_engine(url: str | None = None) -> Engine:
    url = url or load_config().db_url
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def read_sql_file(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


def init_schema(engine: Engine) -> None:
    statements = [s.strip() for s in read_sql_file("schema.sql").split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

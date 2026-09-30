"""CLI: run the full pipeline locally and (re)generate reports/experiments.md + reports/summary.json.

    python -m retailsense.pipeline.run_experiments --raw data/raw/online_retail_ii.csv
    python -m retailsense.pipeline.run_experiments --skip-ingest          # reuse the SQL database
    python -m retailsense.pipeline.run_experiments --raw ... --publish     # also upload artifacts to S3/MinIO
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from retailsense.config import load_config
from retailsense.pipeline import tasks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", help="raw transactions file (.csv / .xlsx)")
    p.add_argument("--skip-ingest", action="store_true", help="reuse raw_sales already loaded in the database")
    p.add_argument("--fast", action="store_true", help="tiny models, for smoke tests only")
    p.add_argument("--publish", action="store_true", help="upload artifacts to S3 / MinIO")
    p.add_argument("--reports-dir", default="reports")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config()
    if not args.skip_ingest:
        if not args.raw:
            p.error("--raw is required unless --skip-ingest is given")
        print("ingest:", tasks.ingest(cfg, args.raw))
    print("validate:", tasks.validate(cfg))
    print("features:", tasks.build_features_task(cfg))
    trained = tasks.train(cfg, fast=args.fast)
    print("forecast:", tasks.forecast_next_week(cfg, fast=args.fast))

    art = tasks.artifacts_dir(cfg)
    out = Path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("experiments.md", "summary.json", "next_week_forecast.csv"):
        shutil.copy(art / name, out / name)

    if args.publish:
        from retailsense.storage.s3 import S3Store

        print("published:", tasks.publish(cfg, S3Store(cfg.s3_bucket, cfg.s3_endpoint)))
    s = trained["summary"]
    print(f"\nholdout SKU-level WAPE {s['baseline_wape_sku']:.1f}% -> {s['best_wape_sku']:.1f}%  "
          f"(baseline {s['baseline_model']} -> {s['best_model']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# TDD evidence - RetailSense Enterprise build

Source: the approved 6-phase plan. Every phase has a RED commit (tests only, run and failing for the intended reason)
followed by a GREEN commit. `git log --oneline` preserves the sequence.

| Phase | RED commit | GREEN commit | What the tests guarantee |
|---|---|---|---|
| 0 Cleanup | d09d9db | e2c1a41 | config comes from env (no `F:\` paths anywhere in legacy code); the known legacy crashes are gone (undefined `safety_stock_factor`, duplicate `run_hybrid_forecast`, wrong kwarg, deprecated `fillna(method=)`) |
| 1 SQL | 8903b8e | 4695cb4 | schema creates 5 tables; returns flagged, junk SKUs dropped, weeks start on Monday; calendar excludes partial weeks; SQL panel is dense, zero-filled, top-N, matches pandas aggregates; split boundaries come from SQL |
| 2 Features | 395b45f | df96b01 | exactly 67 unique feature names; values match hand-computed lags/rolls; **perturbing week-t sales (or price) changes no week-t feature** (leak guard); the change does propagate to t+1 |
| 3 Experiments | 9ad9e9a, c2e9d5e | c013172, 8418418, 5b88542 | MAPE/WAPE/RMSE maths; >=15 configs across 4 families and raw/log/ratio/L1 variants; every family fits and predicts finite non-negative values; ratio target is scale-invariant; model selection uses validation only (corrupting the holdout actuals does not change the choice); top-3 ensemble uses validation ranking; runs and forecasts are logged to SQL with model identity; report contains the measured numbers |
| 4 Pipeline | c71ba04 | 39d6d0c (+ fixes in 8418418, 5b88542) | S3 round trip and idempotent bucket creation (moto); ingest -> validate -> features -> train -> forecast -> publish end to end; validation rejects empty/short data; next-week forecast has one row per SKU; ensemble can be the deployed model; CLI regenerates `reports/`; DAG task chain and schedule (Airflow stubbed) |
| 5 Deploy | c2e9d5e (RED) | 5074395 | Dockerfile/compose/Makefile/CI shape; compose wires Postgres + MinIO to Airflow and Streamlit; Streamlit renders measured metrics (AppTest) and handles an empty DB; **README numbers must equal `reports/summary.json`** |

## Commands actually run
- `python -m pytest tests -q` -> 95 passed (Windows, Python 3.11, pandas 3 / SQLAlchemy 2).
- Same suite in a second venv pinned like Airflow 2.9 (pandas 2.1.4, SQLAlchemy 1.4.52, numpy 1.26.4) -> 95 passed.
  This found and fixed a real incompatibility (`from sqlalchemy import Engine` does not exist in SQLAlchemy 1.4).
- `pytest --cov=retailsense` -> 97% line coverage (613 statements, 17 missed: error branches and CLI `--publish`).
- `python -m retailsense.pipeline.run_experiments --skip-ingest` on the real 1,055,289-row dataset -> `reports/`.

## Known gaps
- `docker compose up`, a live Airflow scheduler, real PostgreSQL and a real MinIO server were **not** executed
  (Docker is not installed in the authoring environment). Their configuration is covered only by structural tests.
- One 13-week holdout; differences of ~1 WAPE point between models are within noise.
- The 12.3% MAPE target from the resume is not reproduced (see README).

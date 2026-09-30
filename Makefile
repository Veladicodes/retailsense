PY ?= python

.PHONY: data test experiments up down demo

data:            ## download UCI Online Retail II and convert to data/raw/online_retail_ii.csv
	$(PY) scripts/download_data.py

test:            ## unit + integration tests with coverage
	$(PY) -m pytest --cov=retailsense --cov-report=term-missing

experiments:     ## ingest -> features -> 23 model runs -> reports/ (local SQLite)
	$(PY) -m retailsense.pipeline.run_experiments --raw data/raw/online_retail_ii.csv

up:              ## Airflow + Postgres + MinIO(S3) + Streamlit
	docker compose up -d --build
	@echo "Airflow http://localhost:8080 (admin/admin) | Streamlit http://localhost:8501 | MinIO http://localhost:9001"

down:
	docker compose down

demo: data experiments
	$(PY) -m streamlit run app/streamlit_app.py

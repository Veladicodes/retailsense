FROM apache/airflow:2.9.3-python3.11

# Airflow 2.9 pins SQLAlchemy 1.4 / pandas 2.1 through its constraints file; the project code is
# tested against exactly those pins (see README "Compatibility").
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"

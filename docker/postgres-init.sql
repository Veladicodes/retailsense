-- Runs once on first start: Airflow metadata gets its own database next to the application database.
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;

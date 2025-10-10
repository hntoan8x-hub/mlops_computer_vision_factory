# infra_deployment/dags/base_dag_config.py

from datetime import datetime, timedelta
import os

# --- Default Airflow Arguments ---
DEFAULT_DAG_ARGS = {
    'owner': 'ml_ops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- Shared Operational Paths (Critical for Execution) ---
# AIRFLOW_HOME is the base path where DAGs, logs, and scripts are accessible
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')

# Paths relative to AIRFLOW_HOME
SCRIPTS_DIR = os.path.join(AIRFLOW_HOME, 'scripts')
CONFIG_DIR = os.path.join(AIRFLOW_HOME, 'config')

# Base command to run a Python script from the scripts/ directory
PYTHON_EXEC_COMMAND = "python"
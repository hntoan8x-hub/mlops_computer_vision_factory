from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

def create_retraining_dag(dag_id: str, schedule_interval: str, retraining_script_path: str, **kwargs: Dict[str, Any]) -> DAG:
    """
    Creates an Airflow DAG for a retraining pipeline.

    Args:
        dag_id (str): The ID of the DAG.
        schedule_interval (str): The schedule interval (e.g., "0 0 * * *").
        retraining_script_path (str): The path to the script that runs the retraining orchestrator.
        **kwargs: Additional DAG parameters.

    Returns:
        DAG: The created Airflow DAG.
    """
    default_args = {
        'owner': 'ml_team',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    with DAG(
        dag_id=dag_id,
        default_args=default_args,
        schedule_interval=schedule_interval,
        catchup=False,
        **kwargs
    ) as dag:
        check_trigger_task = BashOperator(
            task_id='check_retraining_trigger',
            bash_command=f"python {retraining_script_path} --mode check"
        )
        run_retrain_task = BashOperator(
            task_id='run_retraining_job',
            bash_command=f"python {retraining_script_path} --mode run",
            trigger_rule='one_success' # Runs if check_trigger_task succeeds
        )
        check_trigger_task >> run_retrain_task
    
    return dag
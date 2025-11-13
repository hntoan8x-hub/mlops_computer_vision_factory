import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy_operator import DummyOperator

with DAG(
    dag_id='genai_model_retrain_pipeline',
    start_date=datetime.datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:

    start_task = DummyOperator(task_id='start')

    # Task to pull the latest trainer image
    pull_image = BashOperator(
        task_id='pull_trainer_image',
        bash_command='docker pull your-docker-registry/genai-trainer:latest'
    )

    # Task to run the training job
    run_training_job = BashOperator(
        task_id='run_training_job',
        bash_command='docker run --rm your-docker-registry/genai-trainer:latest python GenAI_Factory/domain_models/genai_assistant/services/assistant_trainer.py'
    )

    # Task to push the newly trained model to a registry
    push_model = BashOperator(
        task_id='push_model',
        bash_command='mlflow models push --model-name genai-assistant --version-tag {{ ds_nodash }}'
    )
    
    end_task = DummyOperator(task_id='end')

    start_task >> pull_image >> run_training_job >> push_model >> end_task

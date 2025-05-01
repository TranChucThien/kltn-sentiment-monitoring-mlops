from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from datetime import datetime

from src._01_dataset_preparation import main as dataset_main
from src._02_training import main as training_main
from src._03_evaluate import main as evaluate_main

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="A simple ML pipeline DAG",
    schedule=None,  # hoặc '@daily'
    catchup=False,
    tags=["ml", "pipeline"],
) as dag:

    def run_dataset():
        print("📦 Running Dataset Preparation...")
        dataset_main()

    def run_training(**kwargs):
        print("🧠 Running Training...")
        cv_version, tf_version = training_main()
        # Đưa version vào xcom để task sau dùng
        kwargs['ti'].xcom_push(key='cv_version', value=cv_version)
        kwargs['ti'].xcom_push(key='tf_version', value=tf_version)

    def run_evaluate(**kwargs):
        print("📊 Running Evaluation...")
        cv_version = kwargs['ti'].xcom_pull(key='cv_version', task_ids='train')
        tf_version = kwargs['ti'].xcom_pull(key='tf_version', task_ids='train')
        evaluate_main(name="CountVectorizer_Model", version=cv_version)
        evaluate_main(name="HashingTF_IDF_Model", version=tf_version)

    dataset = PythonOperator(
        task_id="dataset",
        python_callable=run_dataset,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=run_training,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=run_evaluate,
    )

    dataset >> train >> evaluate  # Thiết lập thứ tự thực thi

from datetime import datetime, timezone

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="procurement",
    start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
) as dag:
    reproduce = BashOperator(
        task_id="reproduce",
        bash_command='/app/.venv/bin/python /app/scripts/reproduce.py --raw-archive "$RAW_ARCHIVE" --context-dir "$CONTEXT_DIR"',
    )
    sync_hdfs = BashOperator(
        task_id="sync_hdfs",
        bash_command="/app/.venv/bin/python /app/scripts/sync_hdfs.py",
    )
    reproduce >> sync_hdfs

FROM python:3.12.9-slim-bookworm AS pipeline

RUN pip install --no-cache-dir uv==0.6.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless ca-certificates \
    && ln -s "$(dirname "$(dirname "$(readlink -f /usr/bin/java)")")" /opt/java \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/java \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    BD_PROJECT_ROOT=/app \
    BD_DATA_DIR=/data \
    BD_OUTPUT_DIR=/output \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --locked --no-dev --no-editable

CMD ["python", "scripts/reproduce.py", "--raw-archive", "/inputs/opentender-2023.jsonl.gz", "--context-dir", "/inputs/context"]

FROM pipeline AS airflow

RUN uv venv /opt/airflow-venv \
    && uv pip install --python /opt/airflow-venv/bin/python \
       "apache-airflow[postgres]==3.1.0" \
       --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.1.0/constraints-3.12.txt

ENV AIRFLOW_HOME=/opt/airflow \
    PATH="/opt/airflow-venv/bin:/app/.venv/bin:$PATH"

COPY infra/airflow/dags /opt/airflow/dags
CMD ["airflow", "standalone"]

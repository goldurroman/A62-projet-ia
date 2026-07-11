FROM python:3.10-slim
WORKDIR /app

COPY requirements-mlflow.txt /app/requirements-mlflow.txt

RUN pip install --no-cache-dir -r /app/requirements-mlflow.txt
#RUN pip install --no-cache-dir mlflow boto3 pysqlite3

ENV MLFLOW_ENABLE_DNS_REBINDING_PROTECTION="false"
ENV MLFLOW_ALLOWED_HOSTS='["*", "mlflow-service", "mlflow-service:5000", "10.106.119.189"]'


EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

# Use a standard Python base image
FROM python:3.10-slim

# Set up the environment
WORKDIR /app

# Copy exporter code and install dependencies
COPY requirements_monitoring.txt .
RUN pip install --no-cache-dir -r requirements_monitoring.txt

COPY exporters/ .

# Set entrypoint for the exporter script
CMD ["python", "model_metrics_exporter.py"]
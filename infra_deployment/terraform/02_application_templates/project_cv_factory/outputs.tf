# infra_deployment/terraform/02_application_templates/project_cv_factory/outputs.tf

# 1. MLOPS TRACKING OUTPUTS (Needed by Python Factory Code)
output "mlflow_db_connection_string" {
  description = "The SQLAlchemy connection string for MLflow tracking."
  # Format: postgresql://username:password@endpoint:port/dbname
  value = "postgresql://${module.mlflow_db.db_username}:${module.mlflow_db.db_password}@${module.mlflow_db.db_endpoint}:5432/${module.mlflow_db.db_name}"
  sensitive   = true
}

output "artifact_store_uri" {
  description = "Base URI for MLflow artifacts storage."
  value       = module.artifact_store.s3_uri
}

# 2. INFERENCE SERVING OUTPUT (Needed by CloudInferenceClient)
output "sagemaker_endpoint_name" {
  description = "The name of the deployed SageMaker Real-time Endpoint."
  value       = module.model_endpoint.endpoint_name
}
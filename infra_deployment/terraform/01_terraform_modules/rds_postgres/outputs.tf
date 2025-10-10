# infra_deployment/terraform/01_terraform_modules/rds_postgres/outputs.tf

output "db_endpoint" {
  description = "The endpoint address for the MLflow database."
  value       = aws_rds_cluster.mlflow_db_cluster.endpoint
}

output "db_username" {
  description = "The master username (sensitive)."
  value       = aws_rds_cluster.mlflow_db_cluster.master_username
  sensitive   = true
}

output "db_password" {
  description = "The master password (sensitive)."
  value       = var.db_password
  sensitive   = true
}

output "db_name" {
  description = "The name of the MLflow database."
  value       = var.db_name
}
# infra_deployment/terraform/02_application_templates/project_cv_factory/variables.tf

variable "project_name_prefix" {
  description = "A short, unique name for the project (e.g., cv-factory)."
  type        = string
  default     = "cv-factory-mlops"
}

variable "environment" {
  description = "The deployment environment (dev, staging, prod)."
  type        = string
}

variable "region" {
  description = "The AWS region to deploy resources into."
  type        = string
  default     = "us-east-1"
}

# CRITICAL: Security inputs loaded from environment/tfvars
variable "db_username" {
  description = "Master username for MLflow RDS database."
  type        = string
}

variable "db_password" {
  description = "Master password for MLflow RDS database."
  type        = string
  sensitive   = true
}

# CRITICAL: Inputs controlling the final compute layer
variable "sagemaker_instance_type" {
  description = "Instance type for the SageMaker Endpoint (e.g., ml.t2.medium for dev, ml.g4dn.xlarge for prod)."
  type        = string
}

variable "inference_image_uri" {
  description = "URI of the pre-built inference Docker image in ECR."
  type        = string
}
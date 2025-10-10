# infra_deployment/terraform/01_terraform_modules/iam_roles/variables.tf

variable "project_name" {
  description = "The prefix name for the project."
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
}

variable "artifact_bucket_arn" {
  description = "ARN of the S3 bucket where model artifacts are stored (output from s3_artifact_store module)."
  type        = string
}
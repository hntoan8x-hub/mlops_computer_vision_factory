# infra_deployment/terraform/01_terraform_modules/s3_artifact_store/variables.tf

variable "project_name" {
  description = "The prefix name for the S3 bucket (e.g., 'cv-factory')."
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod). Used for tagging and naming."
  type        = string
}

variable "acl_type" {
  description = "Canned ACL for the bucket. Private access is mandatory for artifacts."
  type        = string
  default     = "private"
}

variable "force_destroy" {
  description = "Set to true to allow Terraform to destroy the bucket when its not empty. ONLY set to true for ephemeral DEV environments."
  type        = bool
  default     = false
}
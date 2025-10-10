# infra_deployment/terraform/01_terraform_modules/rds_postgres/variables.tf

variable "project_name" {
  description = "The name/prefix for the project."
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
}

variable "vpc_security_group_ids" {
  description = "List of Security Group IDs (from the network module) that should be allowed to connect to the DB."
  type        = list(string)
}

variable "db_instance_class" {
  description = "The type of RDS instance (e.g., db.t3.micro, db.m5.large)."
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "The name of the database to create for MLflow."
  type        = string
  default     = "mlflowdb"
}

# CRITICAL: Admin credentials should be loaded securely, not hardcoded.
variable "db_username" {
  description = "Master username for the database."
  type        = string
}

variable "db_password" {
  description = "Master password for the database."
  type        = string
  sensitive   = true # CRITICAL: Marks password as sensitive
}
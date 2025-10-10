# infra_deployment/terraform/01_terraform_modules/network/variables.tf

variable "project_name" {
  description = "The prefix name for the project."
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
}

variable "vpc_cidr_block" {
  description = "The CIDR block for the VPC (e.g., 10.0.0.0/16)."
  type        = string
  default     = "10.0.0.0/16"
}

variable "azs" {
  description = "A list of Availability Zones to use (e.g., [us-east-1a, us-east-1b])."
  type        = list(string)
}
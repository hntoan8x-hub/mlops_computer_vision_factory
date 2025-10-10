# infra_deployment/terraform/02_application_templates/project_cv_factory/main.tf

# -----------------------------------------------------
# 1. PROVIDER SETUP
# -----------------------------------------------------
provider "aws" {
  region = var.region
}

# -----------------------------------------------------
# 2. NETWORK & SECURITY LAYER (Foundation)
# -----------------------------------------------------
module "network" {
  source = "../../01_terraform_modules/network"
  project_name = var.project_name_prefix
  environment  = var.environment
  vpc_cidr_block = "10.0.0.0/16"
  azs          = ["${var.region}a", "${var.region}b", "${var.region}c"] # Use 3 AZs for HA
}

# -----------------------------------------------------
# 3. STORAGE LAYER (Artifacts & Database)
# -----------------------------------------------------
module "artifact_store" {
  source = "../../01_terraform_modules/s3_artifact_store"
  project_name_prefix = var.project_name_prefix
  environment  = var.environment
  # Set force_destroy=true only for dev/staging environments
  force_destroy = var.environment != "prod" ? true : false 
}

module "mlflow_db" {
  source = "../../01_terraform_modules/rds_postgres"
  project_name = var.project_name_prefix
  environment  = var.environment
  db_username  = var.db_username
  db_password  = var.db_password # Sensitive input
  
  # CRITICAL: Injects the security boundary from the network module
  vpc_security_group_ids = [module.network.rds_security_group_id] 
}

# -----------------------------------------------------
# 4. IDENTITY LAYER (IAM Roles)
# -----------------------------------------------------
module "iam" {
  source = "../../01_terraform_modules/iam_roles"
  project_name = var.project_name_prefix
  environment  = var.environment
  
  # CRITICAL: Injects the ARN of the S3 bucket to create the execution policy
  artifact_bucket_arn = module.artifact_store.bucket_arn 
}

# -----------------------------------------------------
# 5. DEPLOYMENT LAYER (Model Serving)
# -----------------------------------------------------
# NOTE: We assume the sagemaker_endpoint module exists in 01_terraform_modules
module "model_endpoint" {
  source = "../../01_terraform_modules/sagemaker_endpoint"
  
  # CRITICAL: Injects the necessary dependencies
  endpoint_name = "${var.project_name_prefix}-prod-endpoint"
  instance_type = var.sagemaker_instance_type
  image_uri     = var.inference_image_uri
  
  # Injects security context (Execution Role and VPC network for the Endpoint)
  execution_role_arn = module.iam.sagemaker_execution_role_arn
  vpc_security_group_ids = [module.network.app_security_group_id]
}
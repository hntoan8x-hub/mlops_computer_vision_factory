# infra_deployment/terraform/02_application_templates/project_cv_factory/envs/dev.tfvars

# -----------------------------------------------------
# A. GENERAL SETTINGS
# -----------------------------------------------------
environment = "dev"
region      = "us-east-1"
project_name_prefix = "cv-factory-dev"

# -----------------------------------------------------
# B. SECURITY (CRITICAL - Load securely via CI/CD Secrets or Vault)
# NOTE: In reality, these would be loaded from a secret manager (AWS Secrets Manager/Vault)
db_username = "mlflow_dev_user"
db_password = "secureDevPassword123" 

# -----------------------------------------------------
# C. COMPUTE & RESOURCES (CHEAP & FAST)
# -----------------------------------------------------
# Small, CPU-based instance for quick testing and low cost
sagemaker_instance_type = "ml.t2.medium" 
# Example URI for the inference container built by the CI/CD pipeline
inference_image_uri     = "123456789012.dkr.ecr.us-east-1.amazonaws.com/cv-inference-dev:latest"
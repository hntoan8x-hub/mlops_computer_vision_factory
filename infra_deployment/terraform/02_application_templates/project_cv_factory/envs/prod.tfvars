# infra_deployment/terraform/02_application_templates/project_cv_factory/envs/prod.tfvars

# -----------------------------------------------------
# A. GENERAL SETTINGS
# -----------------------------------------------------
environment = "prod"
region      = "us-east-1"
project_name_prefix = "cv-factory-prod"

# -----------------------------------------------------
# B. SECURITY (CRITICAL - Load securely)
# NOTE: Use the highest level of security for these credentials
db_username = "mlflow_prod_master"
db_password = "TOP_SECRET_PROD_PASSWORD"

# -----------------------------------------------------
# C. COMPUTE & RESOURCES (HA & MAXIMUM RELIABILITY)
# -----------------------------------------------------
# High-end GPU for maximum throughput and low latency
sagemaker_instance_type = "ml.p3.2xlarge" 
# Image URI for the production deployment
inference_image_uri     = "123456789012.dkr.ecr.us-east-1.amazonaws.com/cv-inference-prod:latest"
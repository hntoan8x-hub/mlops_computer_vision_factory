# infra_deployment/terraform/02_application_templates/project_cv_factory/envs/staging.tfvars

# -----------------------------------------------------
# A. GENERAL SETTINGS
# -----------------------------------------------------
environment = "staging"
region      = "us-east-1"
project_name_prefix = "cv-factory-staging"

# -----------------------------------------------------
# B. SECURITY (CRITICAL - Load securely)
# NOTE: Use different credentials than dev
db_username = "mlflow_staging_user"
db_password = "secureStagingPasswordXYZ"

# -----------------------------------------------------
# C. COMPUTE & RESOURCES (PERFORMANCE TESTING)
# -----------------------------------------------------
# GPU instance for performance and load testing
sagemaker_instance_type = "ml.g4dn.xlarge" 
# Image URI built for the staging deployment
inference_image_uri     = "123456789012.dkr.ecr.us-east-1.amazonaws.com/cv-inference-staging:latest"
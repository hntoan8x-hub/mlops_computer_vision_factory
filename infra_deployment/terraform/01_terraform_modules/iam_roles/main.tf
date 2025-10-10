# infra_deployment/terraform/01_terraform_modules/iam_roles/main.tf

# This file creates the two most critical roles: one for SageMaker Execution (Compute/Training) 
# and one for the MLflow Backend (if applicable, but mainly focusing on the SageMaker role here as it's the primary compute identity)
# A. Execution Role for SageMaker/Compute
# This role is assumed by the SageMaker service when running your training or hosting your endpoint. 
# It needs permissions to read model artifacts and write logs.

# 1. IAM Role Definition (Trust Policy)
# This policy allows the SageMaker service principal to assume this role.
resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${var.project_name}-${var.environment}-sagemaker-exec-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Environment = var.environment
  }
}

# 2. IAM Policy (Permissions Policy)
# This policy defines WHAT the role can do.
resource "aws_iam_policy" "sagemaker_policy" {
  name        = "${var.project_name}-${var.environment}-sagemaker-policy"
  description = "Policy for SageMaker to access artifacts and write logs."
  
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      # CRITICAL: Permission to read/write from the Artifact Store (S3)
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ],
        Effect   = "Allow",
        Resource = ["${var.artifact_bucket_arn}/*"]
      },
      # Mandatory: Permission to write logs to CloudWatch
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Effect   = "Allow",
        Resource = "*" # Resource is often * for logs
      },
      # Add necessary permissions for ECR/other services here
    ]
  })
}

# 3. Attach Policy to Role
resource "aws_iam_role_policy_attachment" "sagemaker_policy_attach" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}

# 4. Attach Managed Policy for Full SageMaker Access (Optional, for simplicity/wider access)
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}
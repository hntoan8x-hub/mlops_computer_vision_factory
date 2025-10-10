# infra_deployment/terraform/01_terraform_modules/s3_artifact_store/main.tf

# 1. S3 Bucket Resource
resource "aws_s3_bucket" "artifact_bucket" {
  # Naming convention: project-env-mlops-artifacts
  bucket = "${var.project_name}-${var.environment}-mlops-artifacts" 
  acl    = var.acl_type
  
  # CRITICAL for DEV environments to allow easy teardown (use sparingly in PROD)
  force_destroy = var.force_destroy 

  tags = {
    Name        = "${var.project_name}-${var.environment}-mlops-artifact-store"
    Environment = var.environment
  }
}

# 2. VERSIONING (CRITICAL for Auditability/Reproducibility)
# Ensures every version of a model artifact or checkpoint is retained.
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.artifact_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 3. ENCRYPTION (Mandatory Security Best Practice)
# Enforces AES256 server-side encryption by default.
resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.artifact_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 4. BLOCK PUBLIC ACCESS (Mandatory Security Best Practice)
# Prevents accidental exposure of sensitive model weights and training data.
resource "aws_s3_bucket_public_access_block" "public_access_block" {
  bucket                  = aws_s3_bucket.artifact_bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
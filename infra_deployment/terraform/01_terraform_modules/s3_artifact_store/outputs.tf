# infra_deployment/terraform/01_terraform_modules/s3_artifact_store/outputs.tf

output "bucket_name" {
  description = "The unique name of the created S3 bucket."
  value       = aws_s3_bucket.artifact_bucket.bucket
}

output "bucket_arn" {
  description = "The ARN (Amazon Resource Name) of the S3 bucket (needed for IAM policy granting).."
  value       = aws_s3_bucket.artifact_bucket.arn
}

output "s3_uri" {
  description = "The base S3 URI for artifacts (s3://bucket-name)."
  value       = "s3://${aws_s3_bucket.artifact_bucket.bucket}"
}
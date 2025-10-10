# infra_deployment/terraform/01_terraform_modules/iam_roles/outputs.tf

output "sagemaker_execution_role_arn" {
  description = "The ARN of the IAM role that SageMaker assumes for execution."
  value       = aws_iam_role.sagemaker_execution_role.arn
}
# infra_deployment/terraform/01_terraform_modules/network/outputs.tf

output "vpc_id" {
  description = "The ID of the created VPC."
  value       = aws_vpc.mlops_vpc.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets (for RDS, EKS nodes)."
  value       = aws_subnet.private_subnets[*].id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets (for Load Balancers, NAT GW)."
  value       = aws_subnet.public_subnets[*].id
}

output "rds_security_group_id" {
  description = "ID of the Security Group for the RDS database."
  value       = aws_security_group.rds_security_group.id
}

output "app_security_group_id" {
  description = "ID of the Security Group for ML Application Nodes (EKS/SageMaker).."
  value       = aws_security_group.app_security_group.id
}
# infra_deployment/terraform/01_terraform_modules/rds_postgres/main.tf

# 1. DB Subnet Group (Required for placing RDS in private subnets)
resource "aws_db_subnet_group" "rds_subnet_group" {
  name       = "${var.project_name}-${var.environment}-mlflow-sng"
  subnet_ids = var.private_subnet_ids # Assuming this input comes from the network module (CRITICAL)

  tags = {
    Name = "${var.project_name}-${var.environment}-mlflow-sng"
  }
}

# 2. RDS Instance (PostgreSQL for MLflow Backend)
resource "aws_rds_cluster_instance" "mlflow_db_instance" {
  identifier          = "${var.project_name}-mlflow-db-${var.environment}"
  engine              = "postgres"
  engine_version      = "14.7"
  instance_class      = var.db_instance_class
  db_subnet_group_name = aws_db_subnet_group.rds_subnet_group.name
  skip_final_snapshot = true # CRITICAL for dev/staging teardown
  publicly_accessible = false # CRITICAL for security (accessible only via VPC)
  allocated_storage   = 20
  
  # Multi-AZ deployment (for PROD/HA)
  multi_az            = var.environment == "prod" ? true : false
  
  # Security Group (restricts access to application containers only)
  vpc_security_group_ids = var.vpc_security_group_ids
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-mlflow-db"
  }
}

# 3. RDS Database (The actual database creation)
resource "aws_rds_cluster" "mlflow_db_cluster" {
  cluster_identifier = "${var.project_name}-mlflow-cluster-${var.environment}"
  engine             = "postgres"
  database_name      = var.db_name
  master_username    = var.db_username
  master_password    = var.db_password
  db_subnet_group_name = aws_db_subnet_group.rds_subnet_group.name
  skip_final_snapshot = true
}
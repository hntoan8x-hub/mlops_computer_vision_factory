# infra_deployment/terraform/01_terraform_modules/network/main.tf

# 1. VIRTUAL PRIVATE CLOUD (VPC)
# Creates the isolated network boundary.
resource "aws_vpc" "mlops_vpc" {
  cidr_block           = var.vpc_cidr_block
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name        = "${var.project_name}-${var.environment}-vpc"
    Environment = var.environment
  }
}

# 2. INTERNET GATEWAY (IGW)
# Allows resources in Public Subnets to access the internet.
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.mlops_vpc.id
  tags = {
    Name = "${var.project_name}-${var.environment}-igw"
  }
}

# 3. PUBLIC SUBNETS (For Load Balancers, Bastion Hosts, Public Endpoints)
# Map subnets across Availability Zones (AZs) for high availability (HA).
resource "aws_subnet" "public_subnets" {
  count             = length(var.azs)
  vpc_id            = aws_vpc.mlops_vpc.id
  cidr_block        = cidrsubnet(var.vpc_cidr_block, 8, count.index) # Example: 10.0.0.0/24, 10.0.1.0/24
  availability_zone = var.azs[count.index]
  map_public_ip_on_launch = true # Instances get a public IP
  tags = {
    Name = "${var.project_name}-${var.environment}-public-subnet-${count.index}"
  }
}

# 4. PRIVATE SUBNETS (CRITICAL: For RDS, EKS Nodes, and ML Compute)
# Resources here cannot be accessed directly from the internet.
resource "aws_subnet" "private_subnets" {
  count             = length(var.azs)
  vpc_id            = aws_vpc.mlops_vpc.id
  cidr_block        = cidrsubnet(var.vpc_cidr_block, 8, count.index + length(var.azs)) # Example: 10.0.2.0/24, 10.0.3.0/24
  availability_zone = var.azs[count.index]
  tags = {
    Name = "${var.project_name}-${var.environment}-private-subnet-${count.index}"
  }
}

# 5. NAT GATEWAY (Allows Private Subnets to access the internet for updates/downloads)
# Requires an Elastic IP (EIP)
resource "aws_eip" "nat_eip" {
  vpc = true
}

resource "aws_nat_gateway" "nat_gw" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id     = aws_subnet.public_subnets[0].id # Place NAT GW in the first public subnet
  depends_on    = [aws_internet_gateway.igw]
}

# 6. ROUTING (Route Tables to link subnets to IGW/NAT GW)
# Public Route Table: Routes 0.0.0.0/0 traffic to the IGW
resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.mlops_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

# Private Route Table: Routes 0.0.0.0/0 traffic to the NAT GW
resource "aws_route_table" "private_route_table" {
  vpc_id = aws_vpc.mlops_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gw.id
  }
}

# Associate Public Subnets with Public Route Table
resource "aws_route_table_association" "public_rt_assoc" {
  count          = length(aws_subnet.public_subnets)
  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_route_table.id
}

# Associate Private Subnets with Private Route Table (CRITICAL for ML Compute)
resource "aws_route_table_association" "private_rt_assoc" {
  count          = length(aws_subnet.private_subnets)
  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_route_table.id
}

# 7. SECURITY GROUPS (CRITICAL for Access Control)
# Security Group for RDS/MLflow DB (Only accepts traffic from Application/EKS nodes)
resource "aws_security_group" "rds_security_group" {
  name        = "${var.project_name}-${var.environment}-rds-sg"
  description = "Allow inbound traffic from application subnets to RDS."
  vpc_id      = aws_vpc.mlops_vpc.id

  # Add specific ingress rules later when EKS/application SG IDs are known, 
  # or make it wide enough for the entire VPC CIDR block for internal communication.
  ingress {
    from_port   = 5432 # PostgreSQL port
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr_block] # Allow connection from within the entire VPC
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security Group for Application Nodes (EKS/SageMaker/Containers)
resource "aws_security_group" "app_security_group" {
  name        = "${var.project_name}-${var.environment}-app-sg"
  description = "General SG for application nodes."
  vpc_id      = aws_vpc.mlops_vpc.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
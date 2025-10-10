# 🚀 Production MLOps CI/CD Pipeline: End-to-End System Flow

This document outlines the complete production-grade DevOps/MLOps workflow for a Machine Learning Factory project (e.g., CV Factory). It illustrates the end-to-end flow from **local development** to **GitHub CI/CD**, **Terraform provisioning**, **Docker image deployment**, and **AWS SageMaker inference**.

---

## 🧑‍💻 1. Local Development (Git Workflow)

### 🎯 Goal: Develop, commit, and push code to GitHub.

```bash
git init                             # Initialize local Git repository
git add .                            # Stage all changes
git commit -m "Add CNNTrainer module"   # Save snapshot of changes
git remote add origin https://github.com/user/cv-factory.git
git push -u origin main               # Push code to remote 'main' branch
```

**Purpose:** Trigger GitHub Actions workflow upon code push.

---

## ⚙️ 2. GitHub Actions CI/CD Workflow

### 📄 File: `.github/workflows/prod_pipeline.yml`

Triggered automatically when code is pushed to `main` branch.

```yaml
name: Production MLOps CI/CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  build_and_test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest -q
```

✅ If all tests pass, the next job deploys infrastructure and Docker containers.

```yaml
  deploy_infra:
    needs: build_and_test
    runs-on: ubuntu-latest
    steps:
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-southeast-1

      - name: Initialize Terraform
        run: |
          cd infra_deployment/terraform/
          terraform init
          terraform plan
          terraform apply -auto-approve
```

**Terraform Actions:**
- Provision AWS S3, ECR, SageMaker, IAM roles.
- Create remote state and networking components.

---

## 🐳 3. Docker Build & Push to AWS ECR

```yaml
      - name: Build Docker image
        run: |
          docker build -t cv-factory-api:latest -f api_service/Dockerfile .
          docker tag cv-factory-api:latest ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.ap-southeast-1.amazonaws.com/cv-factory-api:latest

      - name: Login to ECR
        run: |
          aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.ap-southeast-1.amazonaws.com

      - name: Push image
        run: |
          docker push ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.ap-southeast-1.amazonaws.com/cv-factory-api:latest
```

**Result:** Docker image pushed to AWS ECR (used by SageMaker).

---

## ☁️ 4. Deploy Model to SageMaker

**Deployment Script:** `infra_deployment/deploy/deploy_to_sagemaker.py`

```python
import boto3

sagemaker = boto3.client("sagemaker", region_name="ap-southeast-1")

response = sagemaker.create_model(
    ModelName="cv-factory-prod-model",
    PrimaryContainer={
        "Image": "123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/cv-factory-api:latest",
        "ModelDataUrl": "s3://cv-factory-artifacts/mlflow/1/model.tar.gz",
    },
    ExecutionRoleArn="arn:aws:iam::123456789012:role/sagemaker-exec-role"
)

endpoint_config = sagemaker.create_endpoint_config(
    EndpointConfigName="cv-factory-prod-config",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": "cv-factory-prod-model",
        "InstanceType": "ml.m5.xlarge",
        "InitialInstanceCount": 1,
    }]
)

sagemaker.create_endpoint(
    EndpointName="cv-factory-prod",
    EndpointConfigName="cv-factory-prod-config"
)

print("✅ SageMaker endpoint deployed successfully!")
```

---

## 🌐 5. Access Model via Public Endpoint

**Client Code:**
```python
import boto3, json
runtime = boto3.client("sagemaker-runtime", region_name="ap-southeast-1")

response = runtime.invoke_endpoint(
    EndpointName="cv-factory-prod",
    ContentType="application/json",
    Body=json.dumps({"image_url": "https://.../sample.jpg"})
)
print(response["Body"].read().decode("utf-8"))
```

Output:
```json
{
  "predicted_label": "Pneumonia",
  "confidence": 0.97
}
```

---

## 🔁 6. Complete CI/CD Flow Diagram

```
🧑‍💻 Developer (Local)
  │
  ├── git add .
  ├── git commit -m "update CNNTrainer"
  └── git push origin main
        ↓
⚙️ GitHub Actions (CI/CD)
  │
  ├── Test code (pytest)
  ├── Deploy infra (Terraform)
  ├── Build + Push Docker (ECR)
  └── Deploy model (SageMaker)
        ↓
☁️ AWS Cloud
  │
  ├── S3 → store MLflow artifacts
  ├── ECR → store Docker images
  ├── SageMaker → serve model endpoint
  └── CloudWatch → log and monitor metrics
```

---

## 🧠 Summary of Roles

| Component | Role |
|------------|------|
| **Git** | Source control & trigger for CI/CD |
| **GitHub Actions** | CI/CD automation engine |
| **Terraform** | Infrastructure as Code (IaC) management |
| **Docker** | Application packaging & portability |
| **ECR** | Container image registry |
| **SageMaker** | Managed model serving platform |
| **S3** | Artifact storage for model and logs |
| **CloudWatch / Prometheus** | System observability and metrics |

---

✅ **Outcome:** End-to-end automated deployment pipeline — from local code to cloud model serving, ensuring reproducibility, scalability, and reliability across environments.


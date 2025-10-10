### ⚙️ CI/CD Pipeline Sequence – Handling YAMLs Across Environments

#### 1. Overview
This diagram illustrates how the four YAML configuration files (`training_config.yaml`, `inference_config.yaml`, `monitoring_config.yaml`, `retrain_config.yaml`) flow through the **CI/CD pipeline**. Each stage (GitHub Actions → Docker → Terraform → Airflow) handles a specific responsibility in deploying and synchronizing configurations across environments.

---

### 2. CI/CD Pipeline Flow
```
┌────────────────────────────────────────────────────────────┐
│                     GITHUB ACTIONS STAGE                   │
│  • Detects commit or tag push (e.g., 'release/v1.2.0')     │
│  • Runs linting, unit tests, and YAML schema validation     │
│  • Builds artifact bundles (Docker image + YAML configs)    │
│                                                            │
│  Example:                                                  │
│  - Validate YAML via Pydantic CLI                          │
│  - Run pytest --cov                                        │
│  - Build docker image tagged with commit SHA               │
│  - Upload configs → S3 or Artifact Registry                │
└──────────────┬─────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────┐
│                       DOCKER BUILD STAGE                   │
│  • Dockerfile copies environment-specific YAMLs             │
│  • ARG ENVIRONMENT flag controls which YAMLs to include     │
│  • Result: docker image with /configs/{env}/*.yaml          │
│                                                            │
│  Example:                                                  │
│  - COPY configs/prod/*.yaml /app/configs/                  │
│  - ENV ACTIVE_CONFIG=/app/configs/prod_training_config.yaml│
└──────────────┬─────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────┐
│                    TERRAFORM DEPLOYMENT STAGE              │
│  • Terraform defines infrastructure + environment mapping  │
│  • Injects YAML paths as variables into deployment modules  │
│  • Manages S3 buckets, ECR images, and Airflow triggers     │
│                                                            │
│  Example:                                                  │
│  variable "mlops_config_path" { default = "configs/prod/" }│
│  module "airflow" {                                        │
│     source = "../modules/airflow"                         │
│     config_path = var.mlops_config_path                    │
│  }                                                         │
└──────────────┬─────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────┐
│                        AIRFLOW EXECUTION STAGE             │
│  • DAGs (monitoring_dag.py, retraining_dag.py, etc.) load  │
│    YAML configs dynamically at runtime                     │
│  • Airflow Variables/Connections manage environment context │
│  • DAG tasks execute scripts using corresponding YAML files │
│                                                            │
│  Example:                                                  │
│  - BashOperator(cmd="python monitoring_main.py --config /app/configs/prod_monitoring_config.yaml") │
│  - BashOperator(cmd="python retraining_main.py --config /app/configs/prod_retrain_config.yaml")   │
└──────────────┬─────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT & FEEDBACK LOOP            │
│  • Trained model deployed via Deployer Adapter (AWS/GCP)   │
│  • Monitoring DAG resumes tracking using prod configs      │
│  • Retrain DAG triggers automatically upon drift detection │
│  • CI/CD ensures versioned YAML traceability               │
│                                                            │
│  Example:                                                  │
│  - MLflow logs model URI + YAML version hash               │
│  - Prometheus tracks alert thresholds from config          │
│  - GitHub Actions posts summary in release notes           │
└────────────────────────────────────────────────────────────┘
```

---

### 3. Key Benefits
- **Immutable Deployments:** Each CI/CD cycle deploys self-contained YAML + image version.
- **Environment Isolation:** Dev/Staging/Prod controlled through config directories.
- **Traceability:** Every model, DAG, and job can be traced back to YAML commit hash.
- **Automation:** Terraform + Airflow close the feedback loop, ensuring retraining and monitoring are triggered autonomously.

✅ **Result:** A fully automated, environment-aware CI/CD pipeline that manages both infrastructure and operational parameters through YAML-driven orchestration.


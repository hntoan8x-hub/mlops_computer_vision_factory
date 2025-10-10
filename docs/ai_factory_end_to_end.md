# 🧭 BIG PICTURE – END-TO-END AI FACTORY SYSTEM (Poster A4)

---

## I. DEV LAYER – CODEBASE & TRAINING PIPELINE

```
cv_factory/
│
├── shared_libs/
│   ├── ml_core/ .................... (Trainer, Evaluator, Feature Store)
│   ├── orchestrators/ ............. (Training, Inference Controllers)
│   └── inference/ ................. (Predictor API Contracts)
│
├── domain_models/
│   └── medical_imaging/ ........... (Domain Logic: Pre/Postprocessor)
│
└── configs/
    ├── training_pipeline_config.yaml
    └── inference_medical_config.yaml
```

**Flow:**
```
TrainingOrchestrator
 → validate_config()
 → CVDataset.load_data()
 → Trainer.fit()
 → MLflow.log_metrics()
 → MLflow.log_model()
 → MLflow.register_model()
```

🧠 *Output: Metadata → PostgreSQL | Artifacts → S3 | Registry → MLflow Model Registry*

---

## II. CLOUD INFRA LAYER – DEPLOYMENT & INFRASTRUCTURE (TERRAFORM)

```
infra_deployment/
│
├── terraform/
│   ├── main.tf .................... (AWS/GCP Providers)
│   ├── s3.tf ...................... (Artifact Store)
│   ├── rds.tf ..................... (Backend Metadata DB)
│   ├── iam.tf ..................... (Service Roles)
│   ├── sagemaker.tf ............... (Model Hosting Infra)
│   └── output.tf .................. (Expose endpoint info)
│
└── docker/
    ├── Dockerfile ................. (Build FastAPI inference service)
    └── docker-compose.yaml
```

**Terraform tạo:**
- S3 bucket → chứa model artifacts.
- RDS → backend metadata store.
- IAM Role → quyền đọc/ghi.
- SageMaker Endpoint → deploy mô hình.

---

## III. OPS LAYER – SERVING, MONITORING & RETRAINING

```
cv_factory/api_service/
│
├── endpoints/
│   └── prediction_router.py ....... (POST /predict)
│
├── clients/
│   └── cloud_inference_client.py .. (Call SageMaker endpoint)
│
└── schemas/
    └── service_schemas.py ......... (Pydantic validation)
```

**Flow:**
```
Client App → FastAPI /predict
   → Validate Input
   → call_sagemaker_endpoint()
   → Return JSON Response
```

🧱 *Containerized bằng Docker và triển khai qua CI/CD.*

---

### MONITORING & RETRAIN LOOP
```
Prometheus → metrics (latency, throughput, error_rate)
Grafana → dashboards & alerts
Airflow DAG → check drift
     ↓
 if drift_detected:
     retrain.py → MLflow new run → register new version → promote to Production
```

🔁 *System học liên tục từ dữ liệu mới.*

---

## IV. CI/CD LAYER – AUTOMATION

```
.github/workflows/mlops_pipeline.yml
│
├── on: push, pull_request
│
├── jobs:
│   ├── build_and_test ............ pytest + lint
│   ├── terraform_apply ........... IaC automation
│   ├── train_model ............... python train.py
│   ├── deploy_model .............. python aws_sagemaker_deploy.py
│   ├── smoke_test ................ pytest tests/inference/
│   └── notify .................... Slack/Webhook alerts
```

📦 *CI/CD tự động hóa toàn bộ vòng đời từ code đến cloud.*

---

## V. MONITORING + CLOUD ADAPTER

**Cloud Adapter:**
```
aws_sagemaker_deploy.py
  → model = sagemaker.Model(model_data=model_uri)
  → model.deploy(endpoint_name="cv-factory-prod")
```

**Monitoring:**
```
Prometheus + Grafana + MLflow Metrics
  → Quantify latency & accuracy drift
  → Trigger retraining workflow
```

---

## VI. END-TO-END TEXT FLOW

```
CONFIG (.yaml)
   ↓
ORCHESTRATOR (train → MLflow)
   ↓
MLFLOW DATABASE (metadata)
   ↓
S3 (model artifacts)
   ↓
REGISTRY (version control)
   ↓
TERRAFORM (infra provisioning)
   ↓
SAGEMAKER ENDPOINT (serving)
   ↓
FASTAPI (predict)
   ↓
PROMETHEUS / GRAFANA (monitor)
   ↓
AIRFLOW (drift detection)
   ↓
RETRAINING LOOP → back to orchestrator
```

---

## VII. SUMMARY TABLE

| Layer | Vai trò | Công cụ |
|-------|----------|---------|
| **Dev Layer** | Xây dựng logic model & pipeline | ML Core, Orchestrator |
| **Infra Layer** | Tự động hóa hạ tầng | Terraform / IaC |
| **Cloud Adapter** | Triển khai model | SageMaker / Vertex AI |
| **Ops Layer** | Giám sát & phục vụ | FastAPI / Prometheus |
| **CI/CD Layer** | Tự động hóa vòng đời | GitHub Actions |
| **Monitoring Layer** | Kiểm soát chất lượng | Airflow, Grafana |

---

📘 **Tổng quan:**
> Hệ thống AI Factory có khả năng huấn luyện, triển khai, giám sát và tái huấn luyện tự động — đảm bảo tính liên tục, có thể mở rộng, và dễ bảo trì.

> Mô hình kiến trúc này có thể mở rộng sang NLP, GenAI, hoặc Healthcare AI chỉ bằng cách thay domain model và adapter tương ứng.


# 🧠 **CV Factory – Surface Anomaly Detection System (SADS)**  
**Architecture & End-to-End MLOps Workflow (Production-Grade Design)**  

> **Principles:** *Decoupling (Separation of Concerns)* & *Dependency Injection (DI)*  
> **Goal:** Ensure *stability – scalability – reusability – lifecycle control* for AI vision systems.

---

## 🚀 I. **End-to-End MLOps Workflow**  

| Cycle | Description | Main Orchestrator | Entry Point |
|--------|--------------|------------------|--------------|
| **Training & Deployment (Offline)** | Train, evaluate, register & deploy models | `CVTrainingOrchestrator` + `CVDeploymentOrchestrator` | `scripts/run_training_job.py` |
| **Inference & Serving (Online)** | Real-time prediction & business decisions | `SADSInferenceOrchestrator` + `SADSPostprocessor` | `SADSPipeline.process_request()` |

---

### 🧱 **A. Training & Deployment Pipeline**

| Stage | Function | Core Components |
|--------|-----------|------------------|
| **1️⃣ Data Preparation** | Connect and load raw images & labels | `ConnectorFactory`, `ManualAnnotatorFactory`, `AutoAnnotatorFactory`, `CVDataset` |
| **2️⃣ Training & Evaluation** | Train CNN / Finetune / Contrastive models | `TrainerFactory`, `EvaluationOrchestrator`, `OutputAdapterFactory` |
| **3️⃣ Model Registration** | Save artifacts & register versions | `MLflowRegistry`, `MLflowLogger` |
| **4️⃣ Deployment Activation** | Trigger model deployment (standard / canary) | `CVDeploymentOrchestrator`, `DeployerFactory` |
| **5️⃣ Serving** | Deploy models to staging / production | `SageMaker`, `K8s`, `LocalDeployer` |

---

### ⚙️ **B. Inference & Serving Pipeline**

| Stage | Description | Related Components |
|--------|-------------|--------------------|
| **1️⃣ API Call** | Receive request from endpoint | `SADSInferenceService`, `SADSPipeline` |
| **2️⃣ Sequential Orchestration** | Execute Detection → Classification → Segmentation | `SADSInferenceOrchestrator`, `CVPredictor` |
| **3️⃣ Prediction Chain** | Run three models sequentially | `DetectionAdapter`, `ClassificationAdapter`, `SegmentationAdapter` |
| **4️⃣ Output Standardization** | Normalize model outputs (BBox, Class, Mask) | `OutputAdapterFactory` |
| **5️⃣ Business Decision** | Apply QA logic, compute defect area, PASS/FAIL | `SADSPostprocessor` |
| **6️⃣ Logging & Feedback** | Log results & send failed samples to retraining loop | `MonitoringService`, `MLflowLogger` |

---

## 🏛️ II. **Layer Responsibilities**

| Layer | Objective | Core Components |
|--------|------------|------------------|
| **Data Ingestion & Labeling** | Load reliable labeled data | `ConnectorFactory`, `BaseDataConnector`, `ManualAnnotatorFactory`, `AutoAnnotatorFactory`, `CVDataset` |
| **ML Core** | Training, evaluation & output normalization | `TrainerFactory`, `EvaluationOrchestrator`, `MetricFactory`, `OutputAdapterFactory`, `MLflowRegistry` |
| **Inference & Deployment** | Manage inference lifecycle & platform deployment | `CVPredictor`, `BaseCVPredictor`, `DeployerFactory`, `IstioTrafficController` |
| **Global Workflow (scripts & orchestrators)** | Automate model lifecycle | `CVTrainingOrchestrator`, `DeploymentOrchestrator`, `run_canary_rollout.py`, `rollback_deployment.py` |

---

## 🧩 III. **Domain Layer – Surface Anomaly Detection (QA Logic)**  

| File | Function | Integration Method |
|-------|-----------|--------------------|
| `sads_inference_orchestrator.py` | Domain orchestrator controlling the three-model pipeline | Inherits `BaseOrchestrator`, invoked by `SADSPipeline.run_inference()` |
| `sads_postprocessor.py` | QA logic: NMS, defect area computation, PASS/FAIL | Injected into `SADSInferenceOrchestrator` |
| `sads_pipeline.py` | Facade managing orchestrator lifecycle & entry point | Calls `SADSInferenceOrchestrator.run()` |
| `sads_inference_service.py` | Service layer handling caching & I/O | Initializes and caches `SADSPipeline` instance |

**Business Flow:**  
```
Image → Detection → Crop → Classification → Segmentation → Merge → Decision → Logging
```

---

## 🧮 IV. **Reusability & Multi-Usecase Expansion**  

| Use Case | Key Adjustments | Custom Layer |
|-----------|------------------|----------------|
| **Vehicle Tracking** | Frame sequence processing, object ID tracking | Domain Postprocessor (SORT / ByteTrack) |
| **Medical Imaging (X-Ray)** | DICOM parsing, Dice metric | Data Ingestion + Evaluator |
| **Retail Shelf Audit** | Detection → OCR → Classification pipeline | Domain Orchestrator |

**Reuse Advantages:**
- 100% shared modules (`shared_libs/`) are reusable.
- Only requires defining a new Domain Orchestrator + Postprocessor.

---

## 🏗️ V. **Blueprint Architecture (High-Level Overview)**  

```
┌────────────────────────────────────────────────────────────┐
│                        CV Factory                          │
│────────────────────────────────────────────────────────────│
│ shared_libs/                                               │
│   ├─ data_ingestion/ → Connector, Labeler, Dataset          │
│   ├─ ml_core/ → Trainer, Evaluator, Adapter, Metric         │
│   ├─ inference/ → CVPredictor, OutputAdapter                │
│   ├─ deployment/ → DeployerFactory, TrafficController       │
│   ├─ orchestrators/ → CVTraining, Deployment, Retraining    │
│                                                            │
│ domain_models/surface_anomaly_detection/                    │
│   ├─ sads_inference_orchestrator.py                         │
│   ├─ sads_postprocessor.py                                  │
│   ├─ sads_pipeline.py                                       │
│   ├─ sads_inference_service.py                              │
│                                                            │
│ infra/ (Monitoring, CI/CD, Prometheus, Airflow DAGs)        │
└────────────────────────────────────────────────────────────┘
```

---

## 🧱 VI. **System Maturity & Production Capabilities**

| Capability | Status |
|-------------|--------|
| Modular Design (Separation & Reusability) | ✅ |
| Config Schema Validation (Pydantic v2) | ✅ |
| Model Lifecycle: Train → Deploy → Serve → Retrain | ✅ |
| Multi-Usecase Orchestration | ✅ |
| Canary / Rollback / Blue-Green Deployment | ⚙️ In progress |
| Monitoring + Retraining Feedback Loop | 🔜 Next phase |

---

**📘 Summary:**  
> The CV Factory system is now a **full-lifecycle AI Vision platform**, not a demo project.  
> It supports training, deployment, and domain orchestration (SADS) at an enterprise-grade level,  
> comparable to modern MLOps platforms like **AWS SageMaker, Vertex AI, and Azure ML**.


📘 **CV Factory – Control Plane Operational Flow (Training Script Analysis)**

---

### 🚀 1. Execution Flow of `run_training_job.py`
This document analyzes how the `run_training_job.py` script orchestrates the full lifecycle: **Training → Evaluation → Registration → Deployment**, illustrating how dependencies are injected and executed in a hardened MLOps pipeline.

| Step | Core Action | Key Components | Result |
|------|--------------|----------------|---------|
| **1. Initialization (Composition Root)** | Calls `PipelineRunner.create_orchestrator()` | `PipelineRunner`, `CVPipelineFactory` | Returns an instance of `CVTrainingOrchestrator` fully dependency-injected. |
| **2. Execute Training** | Calls `training_orchestrator.run()` | `CVTrainingOrchestrator` | Starts MLflow Run and invokes `_prepare_data()`. |
| **3. Data Preparation** | `_prepare_data()` → Creates `CVDataset` | `CVDataset`, `ComponentFactory` | Dataset built (I/O, Preprocessing, Labeling). |
| **4. DataLoader Assembly** | `_prepare_data()` → `DataLoaderFactory.create()` | `DataLoaderFactory` | Returns train/val/test loaders (DDP-compatible). |
| **5. Model Training** | `trainer_factory.create()` → `trainer.fit()` | `TrainerFactory`, `BaseCVTrainer` | Model trained, metrics logged via injected `BaseTracker`. |
| **6. Evaluation** | `evaluation_orchestrator.evaluate()` | `EvaluationOrchestrator`, `OutputAdapter` | Final metrics computed (mAP, IoU...) and logged. |
| **7. Model Registration** | `registry.register_model()` and `tag_model_version()` | `BaseRegistry` (via `MLflowService` Façade) | Model registered and version-tagged. |
| **8. Continuous Deployment** | `deployment_orchestrator.run()` (if enabled) | `CVDeploymentOrchestrator`, `BaseDeployer` | Initiates Standard/Canary Deployment. |

---

### 🌳 2. Dependency Graph (Text Diagram)
This diagram shows the dependency relationships between modules and classes, focusing on initialization and execution flow from `run_training_job.py`.

#### A. Initialization Phase (Dependency Injection Assembly)
```
[run_training_job.py]
   │
   ├──▶ PipelineRunner.create_orchestrator()
   │         │
   │         └──▶ CVPipelineFactory.create()
   │                  ├── MLflowModelLoadingService.load
   │                  ├── TrackerFactory.create
   │                  ├── RegistryFactory.create
   │                  ├── DeployerFactory.create
   │                  ├── TrafficControllerFactory.create
   │                  ├── EvaluationOrchestrator
   │                  ├── TrainerFactory
   │                  └── ComponentFactory
   │
   │──▶ Creates CVDeploymentOrchestrator
   │──▶ Creates CVTrainingOrchestrator (Injected with D6–D8, D2–D3)
```
🧩 **Key Insight:** `PipelineRunner` → `CVPipelineFactory` acts as the Composition Root, ensuring all orchestrators are constructed with injected factories and services.

#### B. Execution Phase (Data Flow and Training)
```
CVTrainingOrchestrator.run()
   ├──▶ _prepare_data()
   │       ├──▶ CVDataset.__init__() → DataConnectorFactory
   │       ├──▶ LabelingFactory → Annotation + Label Schema
   │       ├──▶ CVPreprocessingOrchestrator → Image/Video/Depth pipelines
   │       ├──▶ MLComponentFactory → build preprocessor/augmenter chain
   │       └──▶ DataLoaderFactory.create() → train/val/test loaders
   │
   ├──▶ trainer.fit() → Model training loop
   ├──▶ evaluator.evaluate() → Metrics computation via OutputAdapter
   ├──▶ registry.register_model() → Tag + versioning via MLflowService
   └──▶ deployment_orchestrator.run() → Optional CD step
```

💡 **Summary of Hardening Logic:**
- **Dataset Construction** → `CVDataset` with schema validation and factory-based preprocessing.
- **Batching Layer** → `DataLoaderFactory` provides distributed-ready loaders.
- **Orchestration** → `CVTrainingOrchestrator` coordinates dataset, trainer, evaluator, and deployment.
- **Control Plane Integration** → `PipelineRunner` ensures unified lifecycle management.

---

✅ **Conclusion:**  
This Control Plane structure shows how each layer—from Dataset and DataLoader to Trainer and Deployment—is modularized and dependency-injected. The system is now fully **production-hardened**, ensuring that every pipeline (training, evaluation, and deployment) runs consistently and is easily extensible for new domains (Depth, Video, OCR, etc.).


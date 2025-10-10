# CV Factory: Production MLOps Reference Architecture

This document serves as the final technical specification for the CV Factory, a modular and highly scalable MLOps platform designed to ensure traceability, reliability, and architectural separation of concerns across Computer Vision workflows.

---

## I. Architectural Overview and Layering

The entire system is divided into clearly defined layers within the `shared_libs/` directory, adhering strictly to the **Single Responsibility Principle (SRP)** and **Dependency Injection (DI)** for maximum maintainability and reusability.

| Layer | Module | Responsibility | Key Contracts Enforced |
|--------|---------|----------------|-------------------------|
| 0. Utilities | core_utils/ | Centralized generic tools (Configuration Loading, File System I/O, Base Exceptions). | ConfigManager, FileSystemUtils |
| 1. Data I/O | data_ingestion/ | I/O Abstraction: Defines connectors for data movement (read/write, consume/produce). | BaseDataConnector, BaseStreamConnector |
| 2. Logic Execution | data_processing/ | Atomic Logic: Pure mathematical and algorithmic implementations (e.g., raw CV2 resize, PCA math). | Pure Logic/Functions |
| 3. Pipeline Interface | ml_core/pipeline_components_cv/ | Adapter Layer: Wraps Atomic Logic into the BaseComponent contract for pipeline execution. | BaseComponent (fit, transform, save, load) |
| 4. Core MLOps | ml_core/ | Engine Utilities: Trainer, Evaluator, Metrics, Checkpointing, and DDP tools. | BaseCVTrainer, BaseMetric |
| 5. Workflow | orchestrators/ | Master Control: Manages the end-to-end workflow lifecycle, DI, and Quality Gates. | BaseOrchestrator |
| 6. Serving Gateway | inference/ | Model Serving Contract: Defines the API for model execution. | BaseCVPredictor |

---

## II. Strategic Design Decisions (Why the Architecture is Strong)

The platform’s maturity is defined by its ability to solve critical production problems:

### A. Principle: Decoupling and Domain Separation (DI)

| Decision | Implementation Detail | Strategic Value |
|-----------|------------------------|-----------------|
| Postprocessor DI | BaseCVPredictor receives the postprocessor object in its `__init__` (Dependency Injection). | Reusability: The core CVPredictor is kept completely Cloud-Agnostic and Domain-Agnostic. It can be used for Medical or Retail domains without changing its source code. |
| I/O Abstraction | KafkaConnector implements BaseStreamConnector (connect, consume, produce). | Reliability: Enforces safe resource management (`__exit__` calls close()) for network and hardware resources (e.g., Kafka connection, camera handle). |

### B. Principle: State and Consistency (Adapter Pattern)

| Decision | Implementation Detail | Strategic Value |
|-----------|------------------------|-----------------|
| Adapter Pattern | CVResizer (Adapter) only manages the width/height parameters and delegates the actual resize call to a function in `data_processing/cleaners/` (Atomic Logic). | No Redundancy (SRP): Ensures that the mathematical logic is only written in one place (data_processing/), preventing consistency bugs. |
| Stateful Artifacts | CVDimReducer (Adapter) must implement save() and load() by delegating the serialization of the fitted model (PCA/UMAP object) to DimReducerAtomic. | Consistency: Guarantees that the exact learned PCA components from training are used during inference, ensuring feature consistency. |

### C. Principle: Reliability and Quality Gates

| Decision | Implementation Detail | Strategic Value |
|-----------|------------------------|-----------------|
| Configuration Validation | BaseOrchestrator.validate_config() calls Pydantic Schemas (e.g., TrainingOrchestratorConfig). | Error Prevention: Catches critical errors (missing URIs, invalid batch size) before provisioning Cloud resources or starting DDP training. |
| Monitoring Integration | `@measure_orchestrator_latency` decorator is applied to Orchestrator.run(). | Observability: Provides Prometheus metrics (latency, errors) automatically, crucial for detecting bottlenecks or model/data drift in production environments. |
| Distributed Integrity | CNNTrainer.save() uses `distributed_utils.synchronize_between_processes()`. | Data Integrity: Ensures that only Rank 0 saves the checkpoint and that all other ranks wait for the file to be fully written before continuing. |

---

## III. Final MLOps Workflow: The End-to-End Flow

The execution is orchestrated by the **CVTrainingOrchestrator**, which links all layers together:

```
START: training_pipeline_config.yaml is validated by Pydantic (Quality Gate)
   ↓
PREP: CVDataset uses ConnectorFactory to read metadata,
      then uses ComponentOrchestrator to execute the pipeline steps (Resizing, Normalizing).
   ↓
TRAIN: CNNTrainer is instantiated with services (DI) and automatically sets up DDP (Distributed Training).
   ↓
LOGGING: MLflowLogger records metrics and creates the model artifact.
   ↓
REGISTER: Rank 0 uses BaseRegistry to register_model and tag_model_version
          (e.g., attaching the Git SHA and config hash for auditability).
   ↓
DEPLOY (CD): The successful result triggers the CI/CD pipeline,
              which calls AWSSageMakerDeployer (Cloud Adapter)
              to provision the 24/7 Endpoint,
              linking the Inference Container (Docker)
              with the Model Artifact (MLflow S3 URI).
```

---

📘 **Summary:**
> The CV Factory architecture is a *reference implementation* of a fully modular, cloud-agnostic, production-grade MLOps framework for Computer Vision. It achieves total separation of concerns, reproducibility, and resilience — and serves as the foundation for expanding to NLP, GenAI, and Healthcare AI domains.


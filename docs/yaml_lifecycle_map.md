### 🧭 YAML Lifecycle Map – Configuration Flow in Factory-based MLOps Architecture

#### 1. Overview
This diagram illustrates how the four YAML configuration files (`training_config.yaml`, `inference_config.yaml`, `monitoring_config.yaml`, and `retrain_config.yaml`) are dynamically loaded and propagated through orchestrators across the full MLOps lifecycle. It demonstrates the **control flow** of configuration-driven behavior from initialization → orchestration → execution.

---

#### 2. Global Configuration Flow
```
┌──────────────────────────────────────────────────────────────┐
│                    CONFIGURATION SOURCE (YAML)               │
│  - training_config.yaml                                     │
│  - inference_config.yaml                                    │
│  - monitoring_config.yaml                                   │
│  - retrain_config.yaml                                      │
└───────────────┬──────────────────────────────────────────────┘
                │ (Validated by corresponding Pydantic Schemas)
                ▼
┌──────────────────────────────────────────────────────────────┐
│                  FACTORY INITIALIZATION LAYER                │
│ OrchestratorFactory loads proper orchestrator class          │
│ based on config_type (training, inference, monitoring, etc.) │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│               ORCHESTRATION EXECUTION LAYER                  │
│ Each orchestrator reads its config to determine runtime logic│
│ - TrainingOrchestrator: hyperparameters, epochs              │
│ - InferenceOrchestrator: model_uri, device                   │
│ - MonitoringOrchestrator: drift_threshold, reporters          │
│ - RetrainOrchestrator: trigger_conditions, scheduler          │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│                     MLOps EXECUTION LAYER                    │
│ Training, Inference, Monitoring, and Retraining pipelines    │
│ execute based on config-driven logic                         │
│ - Model training via CVTrainingOrchestrator                  │
│ - Real-time inference via CVInferenceOrchestrator            │
│ - Drift monitoring & alerting via MonitoringOrchestrator     │
│ - Job submission via RetrainOrchestrator                     │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│                     CI/CD & DEPLOYMENT LAYER                 │
│ CI/CD swaps environment-specific YAML files:                 │
│   - dev_training_config.yaml                                 │
│   - staging_inference_config.yaml                            │
│   - prod_monitoring_config.yaml                              │
│ YAMLs loaded automatically during pipeline deployment.       │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│                   ADAPTIVE AI FEEDBACK LOOP                  │
│ Monitoring detects drift → Retrain triggers job →            │
│ New model trained & deployed → Monitoring resumes tracking.  │
│ Each step governed by its own YAML config file.              │
└──────────────────────────────────────────────────────────────┘
```

---

#### 3. Key Takeaways
- **Declarative Control:** YAML files act as single sources of truth for each lifecycle stage.
- **Factory-Driven Loading:** The OrchestratorFactory dynamically loads orchestrators based on YAML type.
- **CI/CD Integration:** Environment switching occurs by swapping config YAMLs, not code changes.
- **Adaptive Lifecycle:** The system self-regulates using feedback from monitoring and retraining cycles.

✅ **Result:** A fully modular, configuration-driven MLOps system capable of evolving automatically across environments and lifecycle stages.


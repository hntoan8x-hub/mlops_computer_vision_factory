### 🧭 Automated MLOps Loop – DAGs Layer (Text-based Diagram)

#### 1. Overview
This flow represents the **Automated Feedback Loop** of the AI system, divided into two major segments:
- **Trigger Flow (Monitoring & Retraining)** — Detect issues and decide if retraining is needed.
- **Execution Flow (Training & Deployment)** — Execute training, register models, and deploy updates.

---

### I. The MLOps Loop (End-to-End Automation)
The **Monitoring** and **Retraining** layers form a continuous feedback loop that operates automatically without manual intervention.

---

#### A. Phase 1 – Monitoring & Triggering
```
[Airflow Scheduler – Hourly/Daily]
        │
        ▼
┌────────────────────────────┐
│ monitoring_dag.py          │
│ - BashOperator triggers    │
│   monitoring_main.py       │
└───────────────┬────────────┘
                ▼
┌────────────────────────────────────────────┐
│ MonitoringOrchestrator                    │
│ 1. Read MonitoringConfig.                 │
│ 2. Execute all monitors:                  │
│    - DriftMonitor.check()                 │
│    - FairnessMonitor.check()              │
│    - LatencyMonitor.check()               │
└───────────────┬────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│ Reporters (Base + Concrete)                │
│ - If no alert → PrometheusReporter.report()│
│ - If alert → AlertReporter.report()        │
│   (Slack/Email notification)               │
└───────────────┬────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│ retraining_dag.py                         │
│ - BashOperator triggers retraining_main.py │
└───────────────┬────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│ RetrainOrchestrator                       │
│ 1. Read RetrainConfig.                    │
│ 2. Execute triggers:                      │
│    - DriftTrigger.check()                 │
│    - PerformanceTrigger.check()           │
│    - TimeTrigger.check()                  │
│ 3. If any returns True →                  │
│    job_utils.submit_training_job()        │
│    (Submit job to K8s/Airflow)            │
└────────────────────────────────────────────┘
```

---

#### B. Phase 2 – Execution & Deployment
```
[Triggered by Retrain Job Submission]
        │
        ▼
┌──────────────────────────────┐
│ training_main.py             │
│ - Initiate CVTrainingOrchestrator │
└───────────────┬──────────────┘
                ▼
┌────────────────────────────────────────────┐
│ CVTrainingOrchestrator                    │
│ 1. Run Training → Evaluation.             │
│ 2. Use EvaluationOrchestrator for metrics │
│ 3. Register model to MLflow Registry      │
│    (tag: 'staging_ready')                 │
└───────────────┬────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│ deployment_dag.py                         │
│ - Trigger deployment_main.py after model   │
│   registration                            │
└───────────────┬────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│ Deployer Adapters                         │
│ - AWSSageMakerDeployer.deploy_model()     │
│   (Deploy to staging endpoint)            │
└───────────────┬────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│ Deployment DAG                            │
│ - Run smoke tests                         │
│ - If success → switch_to_production       │
│   (100% traffic to new model)             │
└────────────────────────────────────────────┘
```

---

### II. Operational Control Points

#### 1. Configuration Segregation
| Config File | Purpose | Read By |
|--------------|----------|----------|
| `retrain_config.yaml` | Trigger thresholds | `RetrainOrchestrator` |
| `monitoring_config.yaml` | Alert thresholds | `MonitoringOrchestrator` |
| `training_config.yaml` | Hyperparameters, data paths | `CVTrainingOrchestrator` |

📘 *Effect:* Configuration-driven control — changing retraining frequency or thresholds requires only YAML updates, not code modifications.

---

#### 2. Decision Logic Separation
- `RetrainOrchestrator` does **not** send notifications directly → delegates to `send_slack_alert`.
- `Deployment DAG` does **not** call AWS API directly → delegates to **Deployer Adapter**.

📘 *Effect:* Keeps orchestration clean, modular, and testable — pure coordination logic, no infrastructure coupling.


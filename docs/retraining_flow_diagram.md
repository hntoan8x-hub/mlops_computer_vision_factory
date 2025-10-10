### 🧭 Retraining Module Flow Diagram (Text-based)

#### 1. Overview
Objective: Automate model retraining based on predefined triggers such as drift detection, performance degradation, or time schedule. The module follows a **Trigger–Job architecture**, ensuring models stay up-to-date and adapt to data or environment changes.

---

#### 2. High-Level Architecture
```
                  ┌────────────────────────────┐
                  │ RetrainOrchestrator        │
                  │ (Central Retraining Control)│
                  └──────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────────────┐
          │                      │                              │
┌────────────────┐   ┌──────────────────────┐     ┌─────────────────────┐
│ DriftTrigger    │   │ PerformanceTrigger  │     │ TimeTrigger         │
│ (Monitoring)    │   │ (Evaluation)        │     │ (Scheduler-based)   │
└───────┬─────────┘   └──────────┬──────────┘     └──────────┬──────────┘
        │                        │                           │
        ▼                        ▼                           ▼
┌──────────────────────────────┐ ┌──────────────────────────┐ ┌──────────────────────────┐
│ BaseTrigger Contract         │ │ BaseTrigger Contract     │ │ BaseTrigger Contract     │
│ - check()                    │ │ - get_reason()           │ │ - check()                │
└──────────────────────────────┘ └──────────────────────────┘ └──────────────────────────┘
        │                        │                           │
        └──────────────┬──────────┴──────────────┬────────────┘
                       │                         │
                       ▼                         ▼
           ┌──────────────────────────────┐      ┌──────────────────────────────┐
           │ job_utils.py                 │      │ scheduler_airflow.py / cron  │
           │ - submit_training_job()      │      │ - run scheduled trigger check│
           │ - monitor_job_status()       │      │                              │
           └──────────────────────────────┘      └──────────────────────────────┘
```

---

#### 3. Detailed Operational Flow
```
[1] RetrainOrchestrator starts (either manually or via scheduler)
      ↓
[2] Loads configuration → dynamically imports Trigger classes (Drift, Performance, Time)
      ↓
[3] Iterates through all triggers and executes trigger.check()
      ↓
[4] If any trigger returns True:
        ├── Retrieve trigger.get_reason()
        ├── Log event via base_retrain_orchestrator.log_job_status()
        └── Call job_utils.submit_training_job() with job configuration
      ↓
[5] job_utils handles training execution → sends job to backend system (Kubernetes/Airflow)
      ↓
[6] Monitor job progress via job_utils.monitor_job_status()
      ↓
[7] Upon completion, update system registry and report via logging/monitoring.
```

---

#### 4. External Integration
- **Trigger Sources:**
  - Drift reports from monitoring/ layer.
  - Evaluation metrics from EvaluationOrchestrator.
  - Cron/Airflow schedule.

- **Execution Environment:**
  - Kubernetes jobs or Airflow DAGs.

- **Logging & Tracking:**
  - MLflow / Prometheus / Slack Alerts.

---

#### 5. Final Objective
- Maintain **model freshness and reliability** through automated retraining workflows.
- Enable **continuous learning** and system adaptability by linking monitoring feedback loops with model lifecycle management.


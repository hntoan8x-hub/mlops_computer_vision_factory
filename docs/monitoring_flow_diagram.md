### 🧭 Monitoring Module Flow Diagram (Text-based)

#### 1. Overview
Objective: Monitor the entire AI system at runtime, including **data drift**, **prediction drift**, **fairness**, and **latency**. The `monitoring/` module ensures that the system remains **reliable, transparent, and capable of 24/7 automated alerts**.

---

#### 2. High-Level Architecture
```
                 ┌────────────────────────────┐
                 │ MonitoringOrchestrator     │
                 │ (Central Monitoring Control)│
                 └──────────────┬─────────────┘
                                │
         ┌──────────────────────┼──────────────────────────┐
         │                      │                          │
┌────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│ FeatureDrift   │   │ PredictionDrift      │   │ FairnessMonitor     │
│ Monitor        │   │ Monitor              │   │ (Bias Checker)      │
└───────┬────────┘   └──────────┬───────────┘   └──────────┬──────────┘
        │                       │                          │
        │                       │                          │
        ▼                       ▼                          ▼
┌────────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
│ BaseMonitor Contract       │  │ BaseMonitor Contract     │  │ BaseMonitor Contract     │
│ - check()                  │  │ - get_alert_status()     │  │ - get_report_message()   │
└────────────────────────────┘  └──────────────────────────┘  └──────────────────────────┘
        │                       │                          │
        └──────────┬────────────┴──────────────┬────────────┘
                   │                           │
                   ▼                           ▼
           ┌──────────────────────────────┐    ┌───────────────────────────┐
           │ BaseReporter Contract        │    │ MonitoringConfigSchema     │
           │ - report()                   │    │ - monitors[]               │
           │                              │    │ - reporters[]              │
           └──────────────┬───────────────┘    │ - alert_thresholds         │
                          │                    │ - schedule / frequency     │
                          ▼                    └───────────────────────────┘
               ┌───────────────────────────────┐
               │ Concrete Reporters            │
               │ • PrometheusReporter          │
               │ • GrafanaReporter             │
               │ • AlertReporter (Slack/Email) │
               └───────────────────────────────┘
```

---

#### 3. Detailed Operational Flow
```
[1] MonitoringOrchestrator is triggered on a scheduled basis
      ↓
[2] Load configuration from MonitoringConfigSchema
      ↓
[3] Iterate through all configured monitors
      ├── FeatureDriftMonitor.check()
      ├── PredictionDriftMonitor.check()
      └── FairnessMonitor.check()
      ↓
[4] Each monitor returns results (status, message, metric)
      ↓
[5] If monitor.get_alert_status() == True
        ├── Send alert to AlertReporter (Slack/Email)
        └── Log metrics via PrometheusReporter/GrafanaReporter
      ↓
[6] MonitoringOrchestrator aggregates reports → stores them in logging/MLflow/Grafana
      ↓
[7] If alerts persist → trigger retraining DAG or escalation workflow.
```

---

#### 4. External System Integration
- **Trigger**: CronJob, Airflow DAG, or API event.
- **Input sources**: Data store (Parquet/S3/SQL), prediction logs.
- **Output sinks**: Prometheus metrics, Slack channel, Email alerts, MLflow.

---

#### 5. Final Objective
- Ensure the AI system can **continuously monitor, detect anomalies early, and react automatically.**
- When integrated with the **Retraining Pipeline**, this module enables an **Adaptive AI Lifecycle** — a system capable of self-adjustment and long-term reliability.


📘 **CV Factory – Control Plane Operational Flow (Standard Deployment Script Analysis)**

---

### 🚀 1. Execution Flow of `deploy_standard.py`
The script `deploy_standard.py` is responsible for triggering the **Standard Deployment** pipeline, ensuring that a validated and registered model is safely deployed into the serving environment. This is typically part of a Continuous Deployment (CD) process, initiated automatically by CI/CD systems like Jenkins or GitLab Runner after the model has passed Staging tests.

| Step | Core Action | Key Components | Result |
|------|--------------|----------------|---------|
| **1. Initialization (Composition Root)** | Call `PipelineRunner.create_orchestrator()` with `pipeline_type="deployment"` | `PipelineRunner`, `CVPipelineFactory` | Creates an instance of `CVDeploymentOrchestrator` with injected dependencies (`BaseDeployer`, `BaseTrafficController`, `BaseTracker`). |
| **2. Execute Deployment** | Call `deployment_orchestrator.run(mode="standard")` | `CVDeploymentOrchestrator` | Starts the Standard Deployment flow. |
| **3. Model Deployment** | `_standard_deployment()` → `deployer.async_update_endpoint()` | `BaseDeployer` (injected) | Deploys the new model version to the endpoint (e.g., Kubernetes, SageMaker) with **zero downtime**. |
| **4. Traffic Switching** | Check `self.traffic_controller` → `traffic_controller.async_set_traffic(100%)` | `BaseTrafficController` (injected) | Redirects **100% of traffic** to the new model version (if controller is available). |
| **5. Finalization & Logging** | Log deployment events and metrics | `BaseTracker`, `EventEmitter` | Logs the success status and returns the endpoint ID or deployment metadata. |

---

### 🌳 2. Dependency Graph (Text Diagram)
The following diagram illustrates the **dependency injection assembly** and **execution flow** of the standalone deployment pipeline.

#### A. Initialization Phase (Dependency Assembly)
```
[deploy_standard.py]
   │
   ├──▶ PipelineRunner.create_orchestrator()
   │         │
   │         └──▶ CVPipelineFactory.create()
   │                  ├── TrackerFactory.create
   │                  ├── RegistryFactory.create
   │                  ├── DeployerFactory.create
   │                  ├── TrafficControllerFactory.create
   │                  └── EventEmitter
   │
   │──▶ Creates CVDeploymentOrchestrator
   │──▶ Injects D3, D4, D1, D5 into the orchestrator
   │──▶ Returns a fully initialized CVDeploymentOrchestrator instance
```
🧩 **Key Insight:** `CVPipelineFactory` serves as the **Composition Root** for the deployment pipeline. It constructs and injects all required dependencies (Deployer, Traffic Controller, Tracker, EventEmitter) into the orchestrator, ensuring modular and testable deployment flows.

#### B. Execution Phase (Deployment and Traffic Flow)
```
CVDeploymentOrchestrator.run(mode="standard")
   ├──▶ _standard_deployment()
   │       ├──▶ BaseDeployer.async_update_endpoint()
   │       ├──▶ BaseTrafficController.async_set_traffic(100%)
   │       ├──▶ BaseTracker.log_metrics()
   │       └──▶ EventEmitter.emit_event()
```

💡 **Execution Logic Summary:**
- **Endpoint Deployment:** Ensures seamless model rollout to production infrastructure (EKS, SageMaker) with zero service interruption.
- **Traffic Control:** The Traffic Controller manages the gradual or complete traffic shift between old and new model versions.
- **Monitoring & Logging:** Tracker and EventEmitter record metrics and events for audit and CD monitoring.

---

✅ **Conclusion:**  
`deploy_standard.py` represents the final link in the MLOps chain, ensuring that a trained and validated model is **deployed safely, consistently, and automatically** under the Continuous Deployment paradigm.  
It exemplifies full coordination between **Orchestrator → Deployer → Traffic Controller → Tracker**, providing a robust, observable, and production-grade deployment lifecycle.


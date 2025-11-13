📘 **CV Factory – Control Plane Operational Flow (Canary Rollout & Rollback Deployment Analysis)**

---

### 🚀 3. Script Analysis: `run_canary_rollout.py`
This script triggers the **Canary Deployment** process, allowing gradual traffic shifting to a new model version. It acts as a safety mechanism for testing model stability before full rollout.

#### A. Execution Flow
| Step | Core Action | Related Components | Result |
|------|--------------|--------------------|---------|
| **1. Initialization** | Call `PipelineRunner.create_orchestrator()` with `pipeline_type="deployment"`. | `CVPipelineFactory`, `PipelineRunner` | Creates `CVDeploymentOrchestrator` with injected dependencies (`BaseDeployer`, `BaseTrafficController`). |
| **2. Execution** | `deployment_orchestrator.run(mode="canary")` | `CVDeploymentOrchestrator` | Starts Canary Deployment flow. |
| **3. New Deployment** | `_run_canary_deployment()` → `deployer.async_update_endpoint()` | `BaseDeployer` (Injected) | Deploys a new Canary version (not yet serving traffic). |
| **4. Traffic Shift** | `traffic_controller.async_set_traffic(canary_percent)` | `BaseTrafficController` (Injected) | Gradually redirects a small portion of traffic (e.g., 5%) to the Canary version. |
| **5. Failure Handling** | If traffic switch fails → `deployer.rollback(stable_version)` | `BaseDeployer` | Performs immediate rollback to stable version. |
| **6. Logging** | Log success/failure and events | `BaseTracker`, `EventEmitter` | Canary rollout completes and stabilizes at the specified traffic threshold. |

#### B. Dependency Graph
The dependency setup mirrors `deploy_standard.py`, but emphasizes `BaseTrafficController` for gradual rollout control.
```
[run_canary_rollout.py] → PipelineRunner.create_orchestrator()
     ↓
CVPipelineFactory.create()
     ├── BaseDeployer
     ├── BaseTrafficController
     └── BaseTracker

Execution Flow:
CVDeploymentOrchestrator.run(canary)
     ├── Deployer.async_update_endpoint()
     ├── TrafficController.async_set_traffic()
     │       ├── Success → Log Success
     │       ├── Failure → Deployer.rollback()
     │       └── EventEmitter.emit_event()
```
🟡 **Key Role:** `BaseTrafficController` ensures gradual rollout and safe rollback in case of deployment instability.

---

### 📉 4. Script Analysis: `rollback_deployment.py`
This script executes an **Emergency Rollback**, often triggered automatically by `check_model_health.py` after failure detection or manually by an operator.

#### A. Execution Flow
| Step | Core Action | Related Components | Result |
|------|--------------|--------------------|---------|
| **1. Initialization** | Call `PipelineRunner.create_orchestrator()` with `pipeline_type="deployment"`. | `CVPipelineFactory`, `PipelineRunner` | Creates `CVDeploymentOrchestrator` with injected dependencies (`BaseDeployer`, `BaseTrafficController`). |
| **2. Execution** | `deployment_orchestrator.run(mode="rollback")` | `CVDeploymentOrchestrator` | Starts rollback flow. |
| **3. Rollback Action** | `deployer.rollback(target_version)` | `BaseDeployer` (Injected) | Forces endpoint to shift **100% of traffic** to the stable version. |
| **4. Logging** | Log metrics and completion status | `BaseTracker`, `EventEmitter` | Rollback completes successfully; system returns to a stable state. |

#### B. Dependency Graph
Rollback flow is the simplest, focusing solely on invoking rollback through `BaseDeployer`.
```
[rollback_deployment.py] → PipelineRunner.create_orchestrator()
     ↓
CVPipelineFactory.create()
     ├── BaseDeployer
     ├── BaseTrafficController
     └── BaseTracker

Execution Flow:
CVDeploymentOrchestrator.run(rollback)
     └── Deployer.rollback()
             └── Log Success/Failure
```

🔴 **Key Insight:** `rollback_deployment.py` represents the final safeguard of the deployment system, ensuring immediate recovery to the last known stable state when a Canary or Standard Deployment fails.

---

✅ **Conclusion:**  
These two scripts complete the **Deployment Control Plane** in CV Factory, enabling a full end-to-end release strategy: gradual rollout (Canary), full rollout (Standard), and instant rollback (Emergency).  
Together, they ensure production reliability, observability, and zero-downtime resilience across model lifecycle operations.


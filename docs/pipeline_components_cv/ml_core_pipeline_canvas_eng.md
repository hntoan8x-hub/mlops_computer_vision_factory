## 🧠 README: shared_libs/ml_core/pipeline_components_cv/

### 🏗️ Structure and Operational Mechanism of the ML Core Pipeline
This layer is responsible for **building**, **executing**, and **managing the lifecycle** of preprocessing and feature engineering pipelines. It is where the principles of **Decoupling** and **Dependency Inversion** are strictly applied to create your core **MLOps Pipeline Engine**.

---

### 1. 🧩 Core Responsibility Hierarchy

| **Layer** | **File / Component** | **Primary Responsibility** |
|------------|----------------------|-----------------------------|
| **Execution Engine** | `ComponentOrchestrator` | **Orchestration:** Executes a sequence of components, manages the MLOps lifecycle (fit, save, load), and ensures integrated monitoring. |
| **Component Creation** | `ComponentFactory` | **Creation:** Centralizes logic for adapter component instantiation from configuration, ensuring correct and consistent construction. |
| **Input Validation** | `component_config_schema.py` | **Quality Gate:** Defines schema, enforces data types, and validates logical consistency of pipeline parameters. |
| **Atomic Logic** | 13 Adapter Components | **Interface / State Management:** Implements the `BaseComponent` contract, manages component state (save/load), and delegates actual operations (`transform`) to atomic logic (Adaptee). |

---

### 2. 🔁 Core MLOps Lifecycle Flow
The lifecycle of the pipeline is managed by the **ComponentOrchestrator**, consisting of four main phases:

| **Phase** | **Main Action** | **Related Component** | **Hardening Mechanism** |
|------------|-----------------|------------------------|--------------------------|
| **Initialization** | Call `ComponentOrchestrator.__init__(config)`.<br>The orchestrator invokes `PipelineStepConfig` for configuration validation, then calls `ComponentFactory.create()` to construct the adapter chain. | `ComponentOrchestrator`, `ComponentFactory`, `PipelineStepConfig` | **Pydantic Validation** prevents invalid configurations. |
| **Training / Fit** | Call `orchestrator.fit(X, y)`.<br>The orchestrator iterates through the pipeline, calling `component.fit(X, y)` for stateful adapters (e.g., `CVDimReducer`, `CVCNNEmbedder`). | `ComponentOrchestrator`, Adapter Components | `try/except` used in `fit()` to catch component fitting errors and raise detailed `RuntimeError`s. |
| **Execution / Transform** | Call `orchestrator.transform(X)`.<br>The orchestrator applies the `measure_latency` decorator to `transform()` and sequentially executes each component. | `ComponentOrchestrator`, Adapter Components | **Monitoring** tracks latency/errors; delegation logic ensures adapters focus solely on business functionality. |
| **Save / Load** | Call `orchestrator.save/load(path)`.<br>The orchestrator calls `component.save/load(path)`, and adapters use `io_utils.save/load_artifact` for serialization. | `ComponentOrchestrator`, `io_utils` | **I/O Abstraction** isolates storage logic from the engine, ensuring consistent artifact management. |

---

### 📊 Dependency Graph
This diagram illustrates the dependency relationships among the core components and utility modules within the ML Core Pipeline layer.

**Dependency Principle:** Arrows always point from **dependent components** to **the dependencies providing services**.

```
Execution Layer
│
└── A: ComponentOrchestrator

Abstraction Layer
│
├── B: ComponentFactory
└── C: BaseComponent

Validation Layer
│
└── D: PipelineStepConfig (Schema)

Utility & Atomic Layer
│
├── E: io_utils.py
├── F: monitoring_utils.py
└── G: Atomic Logic (Adaptee)

Application Layer (Clients)
│
├── H: CVTrainingOrchestrator
└── I: CVPredictor

Component Adapters
│
└── J: CVResizer / CVDimReducer / CVCNNEmbedder / etc.


1️⃣ Applications depend on the Engine:
    H → A
    I → A

2️⃣ The Engine depends on the Factory, Schema, and Monitoring:
    A → B, D, F

3️⃣ The Factory depends on the Schema and BaseComponent (for creation):
    B → D, C

4️⃣ Atomic Adapters depend on BaseComponent and I/O:
    C → E

5️⃣ Adapter Logic (Transformation) depends on Atomic Logic:
    C → G

6️⃣ Component Adapters depend on Utilities:
    J → C, E, G
```

---

### 🧭 Diagram Explanation
- **Dependency Inversion (DI):** High-level components (A, H, I) do **not directly depend** on low-level modules; instead, they depend on abstractions (B, C). For example, `ComponentOrchestrator` (A) does not directly know about `CVResizer` but calls it through `ComponentFactory` (B).
- **Validation:** `ComponentOrchestrator` (A) always uses `PipelineStepConfig` (D) for configuration validation before constructing the pipeline.
- **Utility Flow:** Core utility services (I/O, Monitoring) reside at the lowest level (E, F) and are imported by `BaseComponent` (C) or `ComponentOrchestrator` (A) to ensure high reusability.
- **Adapter Pattern:** Specific `ComponentAdapters` (J) use `BaseComponent` (C) as their interface and `Atomic Logic` (G) as their execution engine, fully decoupling CV logic from MLOps orchestration.


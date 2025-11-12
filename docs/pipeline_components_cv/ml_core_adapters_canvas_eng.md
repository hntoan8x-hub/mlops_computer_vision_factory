## 🧠 README: Layer ML Core Adapters (pipeline_components_cv/)

### 🧩 Overview
This layer serves as the **Adapter Layer**, acting as the central **Glue Point** that allows pure CV functionalities (Atomic Logic) to participate in higher-level MLOps workflows. It implements the **Adapter Pattern** to wrap the raw CV/ML logic from the Data Processing Layer and convert it into standardized Adapters that comply with the common MLOps contract.

---

### 🎯 I. Objectives and Responsibilities
The main goal of this layer is to enforce the **Dependency Inversion Principle (DIP)**, providing a unified MLOps interface regardless of the underlying logic.

| **Folder / Component** | **Core Responsibility** | **Role in the System** |
|------------------------|-------------------------|------------------------|
| `base/` Base Abstraction | Defines the `BaseComponent` contract (fit, transform, save, load). | Standardizes the interface for all adapters. |
| `configs/` Quality Gate | Provides `PipelineStepConfig` to validate adapter parameters. | Ensures configuration consistency and safety. |
| `factories/` Dependency Injection | `ComponentFactory` maps YAML config names to specific adapter classes. | Automatically instantiates adapters based on configuration. |
| `orchestrator/` Execution Engine | `ComponentOrchestrator` executes a sequence of adapters similar to a Scikit-learn Pipeline. | The execution engine for the entire pipeline. |
| `atomic/` Adapter Implementation | Contains concrete adapters (`CVResizer`, `CVDimReducer`, `CVCNNEmbedder`) wrapping atomic logic. | Implements the real CV logic. |

---

### 🔗 II. The "Glue" Relationship (Integration)
The ML Core Layer acts as the key **Glue Point** connecting two major layers of the Factory:

| **Component** | **Relationship** | **Role** |
|---------------|-----------------|-----------|
| **Input (Adaptee)** | Depends on Atomic Logic Classes from the Data Processing Layer (e.g., `DimReducerAtomic`, `ResizeCleaner`). | Provides the raw CV logic for adapters to wrap. |
| **Output (Adapter)** | Used by Master Orchestrators in the MLOps Workflow Layer (e.g., `CVTrainingOrchestrator`, `SADSInferenceOrchestrator`). | Exposes a standardized MLOps interface for upper layers. |

---

### 📐 III. Dependency Graph
This section illustrates how the ML Core Layer is built following **DIP (Dependency on Abstraction)** and how it connects to higher layers.

#### 1️⃣ Internal Layer Dependencies (The Adapter Framework)
```
Layer ML Core Adapters
│
├── Base Abstraction Layer
│   └── BaseComponent (BCC): Common interface for all adapters (fit, transform, save, load)
│
├── Configs Layer
│   └── ComponentConfigSchema (CSC): Validates adapter parameters
│
├── Factory Layer
│   └── ComponentFactory (CFC): Creates adapters from YAML configs
│        ├── Creates → CVDimReducer
│        ├── Creates → CVResizer
│        └── Creates → CVCNNEmbedder
│
├── Execution Layer
│   └── ComponentOrchestrator (COC): Executes adapters sequentially (like sklearn Pipeline)
│        ├── Uses → ComponentConfigSchema
│        ├── Uses → ComponentFactory
│        └── Executes Sequentially → Adapters
│
└── Adapters (atomic/)
    ├── CVDimReducer (Implements BaseComponent, Wraps Logic/State from Atomic Logic Layer)
    ├── CVResizer (Implements BaseComponent, Wraps Logic from Atomic Logic Layer)
    ├── CVCNNEmbedder (Implements BaseComponent, Wraps Model/Logic from Atomic Logic Layer)
    └── CVHOGExtractor (Implements BaseComponent)

External Dependency:
└── Atomic Logic Layer (Adaptee): Provides raw CV operations (ResizeCleaner, DimReducerAtomic...)
```

#### 2️⃣ External Glue Relationships
```
Layer 3: MLOps Workflow (User)
│
├── CVTrainingOrchestrator → Injects → ComponentFactory
└── SADSInferenceOrchestrator → Uses → ComponentOrchestrator (Execution Engine)

Layer 2: ML Core Adapters
│
├── ComponentFactory → Creates → CVResizer, CVDimReducer, CVCNNEmbedder
└── ComponentOrchestrator → Executes → Adapter Chain (fit → transform → output)

Layer 1: Data Processing Façade
└── CVPreprocessingOrchestrator → Delegates Execution → ComponentOrchestrator
```


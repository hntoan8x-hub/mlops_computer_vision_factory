🚀 **README: Data Processing & Feature Engineering Layer**

This is the **Decoupled Layer** at the core of the **CV Factory** project, responsible for all data preprocessing (Image, Video) and feature/embedding extraction tasks. It is designed as an **autonomous execution engine**, completely independent of the ML Core layer (Trainer, Evaluator).

---

### 🎯 I. Architectural Objectives (Production Hardening)

The main goal of this layer is to ensure **Data Integrity**, **Reproducibility**, and **Flexibility** through:

- **Decoupling:** Remove reverse dependencies from the ML Core layer.  
- **Configuration-Driven:** Entire flow is controlled by a single YAML file (`processing_master_config.yaml`).  
- **Policy-based Adaptive Execution:** Support dynamic logic execution (RandAugment, Conditional Cleaning).  
- **Abstraction & Composition:** Separate Atomic Logic (NumPy, OpenCV) from MLOps Logic (Orchestration).

---

### 📂 II. Module Structure and Responsibilities

This layer is organized into **four main functional zones**:

| Directory | Main Responsibility | Core Components |
|------------|----------------------|----------------|
| **_base/** | Define common abstractions (Contracts) for all components (BaseImageCleaner, BaseAugmenter, BaseFrameSampler). | BaseVideoCleaner, BaseFrameSampler |
| **configs/** | Quality Gate (Pydantic). Contains schema validation for processing configurations. | ProcessingConfig, AugmentationConfig, VideoProcessingConfig |
| **image_components/** | Execution Engine & Atomic Logic for 3D data (Image/Frame). | ImageCleanerOrchestrator, ImageAugmenterFactory, ViTEmbedder |
| **video_components/** | Atomic Logic & Bridge for 4D data (Video). | VideoProcessingOrchestrator, PolicySampler, VideoFrameResizer |

---

### 🔗 III. Global Data Flow (The Façade)

The entry point for the entire layer is **CVPreprocessingOrchestrator**, a façade class that manages all data pipelines:

| Data Flow | Orchestrator Chain (Composition) | Core Transformation Steps |
|------------|-----------------------------------|---------------------------|
| **Image Flow (Default)** | ImageCleanerOrchestrator → ImageAugmenterOrchestrator → ImageFeatureExtractorOrchestrator | Raw Image → Clean/Resize → Augment → Embed → Vector |
| **Video Flow (New)** | VideoProcessingOrchestrator → Image Flow | Video (4D) → Frame Sampler → List of Images (3D) → Image Pipeline |

---

### 📐 IV. Dependency Graph (Text Diagram)

The dependency graph below illustrates how components are **decoupled** and **injected** using the principles of **Composition over Inheritance** and **Dependency Inversion**.

#### Legend:
- `→` **Injection:** Initialization and usage (Composition/Delegation)
- `⇐` **Inherits:** Derived from abstraction
- `···>` **Dashed:** Dependent on Schema/Utils

#### Text Diagram:
```
Layer Orchestrators (Façade & Engine)
 ├─ CVPreprocessingOrchestrator
 │   ├─→ ImageCleanerOrchestrator
 │   ├─→ ImageAugmenterOrchestrator
 │   ├─→ ImageFeatureExtractorOrchestrator
 │   ├─→ VideoProcessingOrchestrator
 │   └···> DataTypeUtils
 │
 ├─ ImageCleanerOrchestrator
 │   ├─→ ImageCleanerFactory
 │   ├─→ CleanerPolicyController
 │   └···> ProcessingConfig
 │
 ├─ ImageAugmenterOrchestrator
 │   ├─→ ImageAugmenterFactory
 │   ├─→ AugmentPolicyController
 │   └···> ProcessingConfig
 │
 ├─ ImageFeatureExtractorOrchestrator
 │   ├─→ ImageFeatureExtractorFactory
 │   ├─→ FeaturePolicyController
 │   └···> ProcessingConfig
 │
 ├─ VideoProcessingOrchestrator
 │   ├─→ VideoCleanerFactory
 │   ├─→ FrameSamplerFactory
 │   └···> ProcessingConfig
 │
 └─ (All Orchestrators) ⇐ Base Image/Video Abstractions

Layer Factories & Atomic Components
 ├─ ImageCleanerFactory → AtomicCleaners ⇐ Base Abstraction
 ├─ ImageAugmenterFactory → AtomicAugmenters ⇐ Base Abstraction
 ├─ ImageFeatureExtractorFactory → AtomicFeature/Embedders ⇐ Base Abstraction
 ├─ VideoCleanerFactory → AtomicVideo/Samplers ⇐ Base Abstraction
 └─ FrameSamplerFactory → AtomicVideo/Samplers ⇐ Base Abstraction
```

---

### 🔍 V. Key Dependency Insights (Hardening Points)

| Principle | Description | Value |
|------------|--------------|--------|
| **Decoupling** | No arrow flows downward from ML Core to Orchestrators. | Ensures complete independence of Data Layer. |
| **Dependency Injection** | Orchestrators never reference Atomic classes directly — only through Factories. | Improves scalability and testability. |
| **Governance** | All Orchestrators depend on ProcessingConfig (Pydantic Schema). | Guarantees config integrity before runtime. |


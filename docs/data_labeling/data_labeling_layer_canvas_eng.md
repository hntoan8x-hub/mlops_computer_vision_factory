## 🏷️ README: Layer Data Labeling (Trusted Labels)

### 1. 📚 Overview and Objectives
The **Data Labeling Layer** is responsible for transforming raw label data (from CSV, JSON, XML, or model output) into **Trusted Labels**. These Trusted Labels are rigorously validated **Pydantic objects**, ensuring structural and semantic integrity before being used in model training.

**Core Goals:** Quality Control and Standardization.

---

### 2. 🧩 Module Architecture and Flow
This layer is divided into three main labeling modes — **Manual**, **Auto**, and **Semi** — coordinated by high-level Factories.

| **Module** | **Main Function** | **Output** |
|-------------|-------------------|-------------|
| `configs/` | Defines Pydantic schemas for configuration (`labeler_config_schema.py`) and label data (`label_schema.py`). | Trusted Labels (Pydantic Models) |
| `base_labeler.py` | **Facade:** Abstracts loading, config validation, and Data Connector setup for reading label files. | Contracts: `load_labels`, `validate_sample`, `convert_to_tensor` |
| `implementations/` | Concrete Labelers (`ClassificationLabeler`, `DetectionLabeler`, etc.) managing task-specific labeling workflows. | List[Dict] of standardized labels |
| `manual_annotation/` | **Parsers:** Converts manual raw annotations (CSV, COCO, VOC) into Trusted Labels. | List[Trusted Labels] |
| `auto_annotation/` | **Proposal Annotators:** Generates automatic annotations from model output (e.g., BBox, Mask) and validates results. | List[Trusted Labels] |
| `semi_annotation/` | **HITL / Active Learning:** Selects samples for labeling (`select_samples`) or refines proposed labels (`refine`). | List[Metadata] or List[Trusted Labels] |

---

### 3. 🛡️ Hardening Principles
- **Pydantic Strictness:** Every label (BBox, Text, Vector) must pass validation via `label_schema.py`, ensuring valid ranges (e.g., BBox ∈ [0,1]) and logical consistency (e.g., x_max > x_min).
- **Factory Decoupling:** `LabelingFactory` only instantiates `BaseLabeler`. Each BaseLabeler internally creates its child annotators/parsers (ManualAnnotatorFactory, AutoAnnotatorFactory).
- **No Direct I/O:** No module directly performs file/DB I/O. All label file operations are delegated to the **Data Connector** (from the Data Ingestion Layer).
- **Final Output Contract:** The `convert_to_tensor` method in each Concrete Labeler ensures outputs are always valid **PyTorch Tensors** ready for the DataLoader.

---

### 4. 🗂️ Key Schemas
| **Schema** | **File** | **Role** |
|-------------|-----------|-----------|
| `LabelerConfig` | `labeler_config_schema.py` | Controls configuration settings |
| `DetectionLabel` | `label_schema.py` | Enforces quality control on labels |
| `DetectionLabelerConfig` | `labeler_config_schema.py` | Controls detection-specific parameters |
| `DetectionParser` | `manual_annotation/` | Converts COCO/VOC raw labels into DetectionLabel |
| `DetectionProposalAnnotator` | `auto_annotation/` | Validates model-generated DetectionLabel outputs |

---

### 📐 Dependency Graph
This dependency diagram illustrates the data flow and relationships among the key modules in the Data Labeling Layer, emphasizing the central role of **Trusted Labels** and **Factories**.

```
Layer Data Ingestion
│
└── A: Data Connector Factory

Layer Data Labeling
│
├── Config & Schema
│   ├── C1: labeler_config_schema.py
│   └── C2: label_schema.py (Trusted Labels)
│
├── F1: Labeling Factory → Creates → B: BaseLabeler
│
├── B: BaseLabeler
│   ├── I1: ClassificationLabeler
│   ├── I2: DetectionLabeler
│   └── I3: SegmentationLabeler
│
├── I2: DetectionLabeler (Main Orchestrator)
│   ├── Calls (1) → F2: Manual Annotator Factory
│   ├── Calls (2) → F3: Auto Annotator Factory
│   └── Calls (3) → F4: Semi Annotator Factory
│
├── F2: Manual Annotator Factory → P1: Manual Parsers
├── F3: Auto Annotator Factory → P2: Auto Annotators
└── F4: Semi Annotator Factory → P3: Refinement / Active Learning

Data Flow & Validation
│
├── P1 → Parses/Validates → C2 (Trusted Labels)
├── P2 → Generates/Validates → C2 (Trusted Labels)
├── P3 → Refines/Validates → C2 (Trusted Labels)
│
├── C2 → Enforces Schema → I1, I2, I3
│
├── I2 → Reads File List → A (Data Connector)
│
└── F1 → Uses C1 → B (BaseLabeler)

Layer Training
│
└── D: CVDataset (Receives Final Labeled Data)
```

---

### 🧭 Diagram Description
- **F1 (Labeling Factory):** Entry point, uses **C1 (Config Schema)** to determine which Labeler to instantiate.
- **B (BaseLabeler):** Uses **A (Data Connector Factory)** from the Ingestion Layer to load raw label files.
- **I2 (DetectionLabeler):** Main orchestrator, coordinating sub-factories (F2, F3, F4) based on annotation mode.
- **P1, P2, P3:** All modules must conform to and return objects defined in **C2 (Trusted Labels)**.
- Final outputs from **I1, I2, I3** are passed to **D (CVDataset)** in the Training Layer.


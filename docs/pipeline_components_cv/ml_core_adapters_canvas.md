## 🧠 README: Layer ML Core Adapters (pipeline_components_cv/)

### 🧩 Tổng quan
Layer này đóng vai trò là **Adapter Layer (Lớp Chuyển đổi)** và là điểm **Glue trung tâm**, cho phép các chức năng CV thuần túy (Atomic Logic) tham gia vào chu trình MLOps cấp cao. Nó triển khai **Adapter Pattern** để bọc (wrap) các Logic CV/ML thuần túy từ Layer Data Processing, biến chúng thành các Adapter tuân thủ hợp đồng MLOps chung.

---

### 🎯 I. Mục tiêu và Trách nhiệm
Mục tiêu chính của Layer này là thực thi **Dependency Inversion Principle (DIP)**, tạo ra một giao diện MLOps đồng nhất, bất kể logic bên dưới là gì.

| **Thư mục / Component** | **Trách nhiệm chính** | **Vai trò trong Hệ thống** |
|---------------------------|------------------------|-----------------------------|
| `base/` Base Abstraction | Định nghĩa hợp đồng `BaseComponent` (fit, transform, save, load). | Chuẩn hóa interface cho mọi Adapter. |
| `configs/` Quality Gate | Cung cấp `PipelineStepConfig` để xác thực tham số của Adapter. | Đảm bảo tính nhất quán và an toàn cấu hình. |
| `factories/` Dependency Injection | `ComponentFactory` ánh xạ tên cấu hình (YAML) với lớp Adapter cụ thể. | Tự động khởi tạo Adapter dựa trên cấu hình. |
| `orchestrator/` Execution Engine | `ComponentOrchestrator` thực thi tuần tự chuỗi các Adapter theo kiểu Scikit-learn Pipeline. | Là động cơ thực thi của toàn bộ pipeline. |
| `atomic/` Adapter Implementation | Chứa các Adapter cụ thể (`CVResizer`, `CVDimReducer`, `CVCNNEmbedder`) bọc Logic Atomic. | Là nơi triển khai logic CV thực tế. |

---

### 🔗 II. Mối quan hệ "Glue" (Tích hợp)
Layer ML Core là điểm Glue quan trọng nhất, kết nối hai Layer lớn nhất của Factory:

| **Thành phần** | **Mối quan hệ** | **Vai trò** |
|----------------|----------------|-------------|
| **Input (Adaptee)** | Layer này phụ thuộc vào các Atomic Logic Classes (Layer Data Processing - ví dụ: `DimReducerAtomic`, `ResizeCleaner`). | Cung cấp logic thuần túy cho Adapter bọc lại. |
| **Output (Adapter)** | Layer này được sử dụng bởi các Master Orchestrator (Layer MLOps Workflow - ví dụ: `CVTrainingOrchestrator`, `SADSInferenceOrchestrator`). | Cung cấp giao diện thống nhất cho MLOps sử dụng. |

---

### 📐 III. Dependency Graph (Biểu đồ Phụ thuộc)
Biểu đồ này minh họa cách Layer ML Core được xây dựng theo **DIP (phụ thuộc vào Abstraction)** và cách nó được sử dụng bởi các Layer cấp cao.

#### 1️⃣ Phụ thuộc Nội bộ Layer (The Adapter Framework)

```
Layer ML Core Adapters
│
├── Base Abstraction Layer
│   └── BaseComponent (BCC): Giao diện chung cho mọi Adapter (fit, transform, save, load)
│
├── Configs Layer
│   └── ComponentConfigSchema (CSC): Xác thực tham số của các Adapter
│
├── Factory Layer
│   └── ComponentFactory (CFC): Tạo Adapter dựa trên YAML config
│        ├── Creates → CVDimReducer
│        ├── Creates → CVResizer
│        └── Creates → CVCNNEmbedder
│
├── Execution Layer
│   └── ComponentOrchestrator (COC): Thực thi tuần tự chuỗi Adapter (theo kiểu sklearn Pipeline)
│        ├── Uses → ComponentConfigSchema
│        ├── Uses → ComponentFactory
│        └── Executes Sequentially → Các Adapter
│
└── Adapters (atomic/)
    ├── CVDimReducer (Implements BaseComponent, Wraps Logic/State từ Atomic Logic Layer)
    ├── CVResizer (Implements BaseComponent, Wraps Logic từ Atomic Logic Layer)
    ├── CVCNNEmbedder (Implements BaseComponent, Wraps Model/Logic từ Atomic Logic Layer)
    └── CVHOGExtractor (Implements BaseComponent)

External Dependency:
└── Atomic Logic Layer (Adaptee): Cung cấp các hàm xử lý CV gốc (ResizeCleaner, DimReducerAtomic...)
```

#### 2️⃣ Mối quan hệ Glue (Phụ thuộc Bên ngoài)

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


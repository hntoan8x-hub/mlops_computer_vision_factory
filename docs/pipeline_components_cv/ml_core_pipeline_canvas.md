## 🧠 README: shared_libs/ml_core/pipeline_components_cv/

### 🏗️ Cấu trúc và Cơ chế Vận hành của ML Core Pipeline
Layer này chịu trách nhiệm cho việc **xây dựng**, **thực thi**, và **quản lý vòng đời** của pipeline tiền xử lý và đặc trưng hóa (preprocessing/feature engineering). Đây là nơi các nguyên tắc **Decoupling** và **Dependency Inversion** được áp dụng triệt để để tạo ra Engine MLOps Pipeline cốt lõi.

---

### 1. 🧩 Phân Cấp Trách Nhiệm Chính

| **Layer** | **Tệp / Thành phần** | **Vai trò Chính (Phụ trách)** |
|------------|----------------------|-------------------------------|
| **Execution Engine** | `ComponentOrchestrator` | **Điều phối (Orchestration):** Thực thi tuần tự chuỗi Component, quản lý vòng đời MLOps (fit, save, load), đảm bảo Monitoring được áp dụng. |
| **Component Creation** | `ComponentFactory` | **Sáng tạo (Creation):** Tập trung logic khởi tạo các Adapter Component từ cấu hình, đảm bảo tính đúng đắn của thể hiện. |
| **Input Validation** | `component_config_schema.py` | **Chất lượng (Quality Gate):** Định nghĩa cấu trúc, kiểm tra kiểu dữ liệu, và xác thực tính logic (logic validation) của tham số pipeline. |
| **Atomic Logic** | 13 Adapter Components | **Giao diện / Trạng thái (Interface/State Management):** Thực thi hợp đồng `BaseComponent`, quản lý trạng thái (save/load), và ủy quyền công việc thực tế (`transform`) cho Atomic Logic (Adaptee). |

---

### 2. 🔁 Vòng Đời MLOps Cơ Bản (Lifecycle Flow)
Vòng đời của pipeline được quản lý bởi **ComponentOrchestrator** qua 4 giai đoạn chính:

| **Pha** | **Hành động Chính** | **Component Liên quan** | **Cơ chế Hardening** |
|----------|--------------------|--------------------------|----------------------|
| **Khởi tạo** | Gọi `ComponentOrchestrator.__init__(config)`.<br>Orchestrator gọi `PipelineStepConfig` để xác thực cấu hình, sau đó gọi `ComponentFactory.create()` để xây dựng chuỗi Adapter. | `ComponentOrchestrator`, `ComponentFactory`, `PipelineStepConfig` | **Pydantic Validation** ngăn chặn cấu hình không hợp lệ. |
| **Huấn luyện / Fit** | Gọi `orchestrator.fit(X, y)`.<br>Orchestrator duyệt qua pipeline, gọi `component.fit(X, y)` cho các Adapter có trạng thái (VD: `CVDimReducer`, `CVCNNEmbedder`). | `ComponentOrchestrator`, Adapter Components | Sử dụng `try/except` trong `fit()` để bắt lỗi fitting và log `RuntimeError` có mô tả chi tiết. |
| **Thực thi / Transform** | Gọi `orchestrator.transform(X)`.<br>Orchestrator áp dụng decorator `measure_latency` cho `transform()` và gọi tuần tự các component. | `ComponentOrchestrator`, Adapter Components | **Monitoring** theo dõi độ trễ/lỗi; logic ủy quyền giúp Adapter chỉ tập trung vào nghiệp vụ. |
| **Lưu / Tải (Save/Load)** | Gọi `orchestrator.save/load(path)`.<br>Orchestrator gọi `component.save/load(path)` và Adapter sử dụng `io_utils.save/load_artifact` để xử lý serialization. | `ComponentOrchestrator`, `io_utils` | **I/O Abstraction** cô lập logic lưu trữ khỏi Engine, đảm bảo tính nhất quán I/O. |

---

### 📊 Biểu Đồ Phụ Thuộc (Dependency Graph)
Biểu đồ này minh họa mối quan hệ phụ thuộc giữa các thành phần cốt lõi (Core Components) và các thành phần tiện ích (Utility) trong lớp ML Core Pipeline.

**Nguyên tắc Phụ Thuộc:** Các mũi tên luôn chỉ từ **thành phần phụ thuộc (Dependent)** đến **thành phần được cung cấp dịch vụ (Dependency)**.

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


1️⃣ Ứng dụng phụ thuộc vào Engine:
    H → A
    I → A

2️⃣ Engine phụ thuộc vào Factory, Schema và Monitoring:
    A → B, D, F

3️⃣ Factory phụ thuộc vào Schema và BaseComponent (để tạo ra):
    B → D, C

4️⃣ Atomic Adapters phụ thuộc vào BaseComponent và I/O:
    C → E

5️⃣ Logic của Adapters (Transformation) phụ thuộc vào Logic Atomic:
    C → G

6️⃣ Component Adapters phụ thuộc vào Utility:
    J → C, E, G
```

---

### 🧭 Giải Thích Biểu Đồ
- **Dependency Inversion (DI):** Các thành phần cấp cao (A, H, I) **không phụ thuộc trực tiếp** vào các thành phần cấp thấp mà chỉ phụ thuộc vào các Abstraction (B, C). Ví dụ, `ComponentOrchestrator` (A) không biết chi tiết về `CVResizer` mà gọi thông qua `ComponentFactory` (B).
- **Validation:** `ComponentOrchestrator` (A) luôn sử dụng `PipelineStepConfig` (D) để xác thực đầu vào trước khi xây dựng pipeline.
- **Utility Flow:** Các dịch vụ cốt lõi (I/O, Monitoring) nằm ở tầng thấp nhất (E, F) và được sử dụng bởi `BaseComponent` (C) hoặc `ComponentOrchestrator` (A) để đảm bảo tái sử dụng cao.
- **Adapter Pattern:** Các `Component Adapter` cụ thể (J) sử dụng `BaseComponent` (C) làm giao diện và `Atomic Logic` (G) làm công cụ thực thi, tách biệt hoàn toàn nghiệp vụ CV khỏi giao diện MLOps.


## 🏷️ README: Layer Data Labeling (Trusted Labels)

### 1. 📚 Mục Tiêu và Tổng Quan
Layer **Data Labeling** chịu trách nhiệm chuyển đổi dữ liệu nhãn thô (từ file CSV, JSON, XML hoặc đầu ra mô hình) thành **Trusted Labels** (Nhãn đáng tin cậy). Các Trusted Labels này là các đối tượng **Pydantic** được xác thực nghiêm ngặt về cấu trúc và ngữ nghĩa, đảm bảo tính toàn vẹn trước khi được sử dụng trong huấn luyện mô hình.

**Mục tiêu chính:** Kiểm soát chất lượng (Quality Control) và Chuẩn hóa (Standardization).

---

### 2. 🧩 Kiến Trúc Module và Luồng Xử Lý
Layer này được chia thành ba luồng chính — **Manual**, **Auto**, và **Semi** — được điều phối bởi các Factory cấp cao.

| **Module** | **Chức năng chính** | **Đầu ra** |
|-------------|---------------------|-------------|
| `configs/` | Định nghĩa các Pydantic Schema cho cấu hình (`labeler_config_schema.py`) và dữ liệu nhãn (`label_schema.py`). | Trusted Labels (Pydantic Models) |
| `base_labeler.py` | **Facade:** Trừu tượng hóa việc tải nhãn, xác thực cấu hình và tạo Data Connector để đọc file nhãn. | Hợp đồng: `load_labels`, `validate_sample`, `convert_to_tensor` |
| `implementations/` | Concrete Labelers (`ClassificationLabeler`, `DetectionLabeler`, v.v.) điều phối toàn bộ flow cho từng loại task. | Danh sách nhãn đã chuẩn hóa (List[Dict]) |
| `manual_annotation/` | **Parsers:** Chuyển đổi nhãn thủ công thô (CSV, COCO, VOC) thành Trusted Labels. | List[Trusted Labels] |
| `auto_annotation/` | **Proposal Annotators:** Sinh nhãn tự động từ mô hình (BBox, Mask) và xác thực đầu ra. | List[Trusted Labels] |
| `semi_annotation/` | **HITL / Active Learning:** Chọn mẫu cần gán nhãn (`select_samples`) hoặc tinh chỉnh nhãn đề xuất (`refine`). | List[Metadata] hoặc List[Trusted Labels] |

---

### 3. 🛡️ Nguyên Tắc Hardening
- **Pydantic Strictness:** Mọi dữ liệu nhãn (BBox, Text, Vector) phải đi qua `label_schema.py` để xác thực phạm vi (ví dụ: BBox ∈ [0,1]) và tính hợp lý (ví dụ: x_max > x_min).
- **Factory Decoupling:** `LabelingFactory` chỉ khởi tạo `BaseLabeler`. Các `BaseLabeler` tự khởi tạo các Annotator/Parser con (ManualAnnotatorFactory, AutoAnnotatorFactory).
- **No Direct I/O:** Không có module nào trong Layer này trực tiếp đọc/ghi file hoặc DB. Mọi thao tác I/O được ủy quyền cho **Data Connector** (từ Layer Data Ingestion).
- **Final Output Contract:** Phương thức `convert_to_tensor` trong mỗi Concrete Labeler đảm bảo đầu ra cuối cùng luôn là **PyTorch Tensor** hợp lệ cho DataLoader.

---

### 4. 🗂️ Danh Sách Các Schema Quan Trọng
| **Schema** | **File** | **Vai Trò** |
|-------------|-----------|--------------|
| `LabelerConfig` | `labeler_config_schema.py` | Kiểm soát cấu hình (Config Control) |
| `DetectionLabel` | `label_schema.py` | Kiểm soát chất lượng nhãn (Quality Control) |
| `DetectionLabelerConfig` | `labeler_config_schema.py` | Kiểm soát tham số cho Detection Task |
| `DetectionParser` | `manual_annotation/` | Chuyển nhãn COCO/VOC thô thành DetectionLabel |
| `DetectionProposalAnnotator` | `auto_annotation/` | Xác thực đầu ra mô hình thành DetectionLabel |

---

### 📐 Sơ Đồ Phụ Thuộc (Dependency Graph)
Sơ đồ dưới đây minh họa luồng dữ liệu và mối quan hệ giữa các module trong Layer Data Labeling, nhấn mạnh vai trò trung tâm của **Trusted Labels** và **Factories**.

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
├── I2: DetectionLabeler (Orchestrator Chính)
│   ├── Gọi (1) → F2: Manual Annotator Factory
│   ├── Gọi (2) → F3: Auto Annotator Factory
│   └── Gọi (3) → F4: Semi Annotator Factory
│
├── F2: Manual Annotator Factory → P1: Manual Parsers
├── F3: Auto Annotator Factory → P2: Auto Annotators
└── F4: Semi Annotator Factory → P3: Refinement / Active Learning

Luồng Dữ Liệu & Xác Thực
│
├── P1 → Parse/Xác Thực → C2 (Trusted Labels)
├── P2 → Sinh/Xác Thực → C2 (Trusted Labels)
├── P3 → Tinh Chỉnh/Xác Thực → C2 (Trusted Labels)
│
├── C2 → Kiểm Soát Schema → I1, I2, I3
│
├── I2 → Đọc Danh Sách File → A (Data Connector)
│
└── F1 → Dùng C1 → B (BaseLabeler)

Layer Training
│
└── D: CVDataset (Nhận Dữ Liệu Gán Nhãn Cuối Cùng)
```

---

### 🧭 Mô Tả Sơ Đồ
- **F1 (Labeling Factory):** Điểm vào, sử dụng **C1 (Config Schema)** để xác định loại Labeler cần khởi tạo.
- **B (BaseLabeler):** Sử dụng **A (Data Connector Factory)** từ Layer Ingestion để tải file nhãn thô.
- **I2 (DetectionLabeler):** Là Orchestrator chính, gọi các Factory con (F2, F3, F4) tùy theo chế độ.
- **P1, P2, P3:** Mọi module đều phải tuân thủ và trả về đối tượng từ **C2 (Trusted Labels)**.
- **Đầu ra cuối cùng** của các Labeler (I1, I2, I3) được truyền sang **D (CVDataset)** ở Layer Training.
# 🧠 **CV Factory – Surface Anomaly Detection System (SADS)**  
**Kiến trúc & Luồng MLOps Toàn Diện (Production-Grade Design)**  

> **Nguyên lý:** *Decoupling (Tách rời trách nhiệm)* & *Dependency Injection (DI)*  
> **Mục tiêu:** Đảm bảo *ổn định – mở rộng – tái sử dụng – dễ kiểm soát vòng đời mô hình.*

---

## 🚀 I. **Luồng Hoạt Động End-to-End (MLOps Workflow)**  

| Chu trình | Mô tả | Orchestrator chính | Entry Point |
|------------|-------|--------------------|--------------|
| **Training & Deployment (Offline)** | Huấn luyện, đánh giá, đăng ký & triển khai mô hình | `CVTrainingOrchestrator` + `CVDeploymentOrchestrator` | `scripts/run_training_job.py` |
| **Inference & Serving (Online)** | Dự đoán và ra quyết định trên ảnh thực tế | `SADSInferenceOrchestrator` + `SADSPostprocessor` | `SADSPipeline.process_request()` |

---

### 🧱 **A. Training & Deployment Pipeline**

| Giai đoạn | Chức năng | Thành phần chính |
|------------|------------|------------------|
| **1️⃣ Data Preparation** | Kết nối dữ liệu, tải ảnh và nhãn | `ConnectorFactory`, `ManualAnnotatorFactory`, `AutoAnnotatorFactory`, `CVDataset` |
| **2️⃣ Training & Evaluation** | Huấn luyện mô hình CNN / Finetune / Contrastive | `TrainerFactory`, `EvaluationOrchestrator`, `OutputAdapterFactory` |
| **3️⃣ Model Registration** | Lưu Artifact + đăng ký model version | `MLflowRegistry`, `MLflowLogger` |
| **4️⃣ Deployment Activation** | Kích hoạt triển khai (standard / canary) | `CVDeploymentOrchestrator`, `DeployerFactory` |
| **5️⃣ Serving** | Triển khai mô hình ra Staging / Production | `SageMaker`, `K8s`, `LocalDeployer` |

---

### ⚙️ **B. Inference & Serving Pipeline**

| Giai đoạn | Mô tả | Thành phần liên quan |
|------------|--------|----------------------|
| **1️⃣ API Call** | Nhận yêu cầu từ endpoint | `SADSInferenceService`, `SADSPipeline` |
| **2️⃣ Sequential Orchestration** | Orchestrator tuần tự Detection → Classification → Segmentation | `SADSInferenceOrchestrator`, `CVPredictor` |
| **3️⃣ Prediction Chain** | Chạy chuỗi 3 mô hình | `DetectionAdapter`, `ClassificationAdapter`, `SegmentationAdapter` |
| **4️⃣ Output Standardization** | Chuẩn hóa đầu ra (BBox, Class, Mask) | `OutputAdapterFactory` |
| **5️⃣ Business Decision** | Áp dụng logic QA, tính diện tích lỗi, PASS/FAIL | `SADSPostprocessor` |
| **6️⃣ Logging & Feedback** | Lưu kết quả, đẩy dữ liệu FAIL về retraining loop | `MonitoringService`, `MLflowLogger` |

---

## 🏩 II. **Chức Năng Từng Layer**

| Layer | Mục tiêu | Thành phần chính |
|--------|-----------|------------------|
| **Data Ingestion & Labeling** | Tải dữ liệu và nhãn tin cậy | `ConnectorFactory`, `BaseDataConnector`, `ManualAnnotatorFactory`, `AutoAnnotatorFactory`, `CVDataset` |
| **ML Core** | Huấn luyện, đánh giá, chuẩn hóa đầu ra | `TrainerFactory`, `EvaluationOrchestrator`, `MetricFactory`, `OutputAdapterFactory`, `MLflowRegistry` |
| **Inference & Deployment** | Quản lý inference & triển khai đa nền tảng | `CVPredictor`, `BaseCVPredictor`, `DeployerFactory`, `IstioTrafficController` |
| **Global Workflow (scripts & orchestrators)** | Tự động hóa vòng đời MLOps | `CVTrainingOrchestrator`, `DeploymentOrchestrator`, `run_canary_rollout.py`, `rollback_deployment.py` |

---

## 🧩 III. **Layer Domain – Surface Anomaly Detection (QA Logic)**  

| File | Vai trò | Cách tích hợp |
|-------|----------|----------------|
| `sads_inference_orchestrator.py` | Orchestrator domain điều phối 3 mô hình | Kế thừa `BaseOrchestrator`, gọi trong `SADSPipeline.run_inference()` |
| `sads_postprocessor.py` | Logic QA: NMS, diện tích lỗi, PASS/FAIL | Inject vào `SADSInferenceOrchestrator` |
| `sads_pipeline.py` | Facade quản lý vòng đời Orchestrator, entry point | Gọi `SADSInferenceOrchestrator.run()` |
| `sads_inference_service.py` | Service layer, caching & giao thức I/O | Khởi tạo và cache `SADSPipeline` instance |

**Luồng nghiệp vụ cụ thể:**  
```
Ảnh → Detection → Crop → Classification → Segmentation → Merge → Decision → Logging
```

---

## 🧦 IV. **Mở Rộng & Tái Sử Dụng (Reusability & Multi-Usecase)**  

| Use Case | Điều chỉnh chính | Layer tùy chỉnh |
|-----------|------------------|----------------|
| **Vehicle Tracking** | Xử lý chuỗi frames, ID tracking | Domain Postprocessor (SORT / ByteTrack) |
| **Medical Imaging (X-Ray)** | Đọc DICOM, metric Dice coefficient | Data Ingestion + Evaluator |
| **Retail Shelf Audit** | Detection → OCR → Classification | Domain Orchestrator |

**Lợi thế tái sử dụng:**
- 100% module shared (`shared_libs/`) có thể dùng lại.
- Chỉ cần viết thêm Domain Orchestrator + Postprocessor.

---

## 🌇 V. **Blueprint Kiến Trúc (High-Level Overview)**  

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


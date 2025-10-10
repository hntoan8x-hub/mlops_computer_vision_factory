# CV Factory: An MLOps Framework for Computer Vision

## Giới thiệu

CV Factory là một framework MLOps toàn diện và có khả năng mở rộng, được xây dựng để đơn giản hóa quá trình phát triển, huấn luyện, triển khai và giám sát các mô hình Computer Vision trong môi trường sản xuất.

Dự án được thiết kế theo kiến trúc module hóa, với các thư viện dùng chung (`shared_libs`) và các mô hình cụ thể cho từng domain (`domain_models`), đảm bảo tính linh hoạt và khả năng tái sử dụng cao.

## Kiến trúc

Kiến trúc của dự án bao gồm các thành phần chính sau:

- **`shared_libs/`**: Thư viện dùng chung cho các tác vụ như nạp dữ liệu, tiền xử lý, huấn luyện và đánh giá.
- **`domain_models/`**: Nơi chứa các dự án cụ thể, tùy chỉnh các thư viện dùng chung cho một domain nhất định (ví dụ: `medical_imaging`).
- **`infra/`**: Hạ tầng triển khai, bao gồm Docker, Kubernetes, CI/CD, và Terraform.

## Yêu cầu

- Python 3.10+
- `pip install -r requirements.txt`
- Docker
- Kubernetes (minikube, Kind, hoặc một cluster đám mây)

## Hướng dẫn sử dụng

### 1. Cấu hình pipeline

Các pipeline được điều khiển hoàn toàn thông qua các file YAML. Bạn có thể tùy chỉnh các bước của pipeline huấn luyện, suy luận, hoặc tái huấn luyện bằng cách chỉnh sửa các file trong `domain_models/medical_imaging/configs/`.

### 2. Chạy pipeline

Bạn có thể chạy các pipeline từ dòng lệnh bằng cách trỏ đến file cấu hình tương ứng:

```bash
# Chạy pipeline huấn luyện
python main_app.py --config cv_factory/domain_models/medical_imaging/configs/pipeline_config.yaml

# Chạy pipeline suy luận (sử dụng một config khác)
python main_app.py --config cv_factory/domain_models/retail_analytics/configs/inference_config.yaml
```

.
├── cv_factory/
│ ├── infra/
│ │ ├── cloud/
│ │ │ ├── aws_sagemaker_deploy.py
│ │ │ ├── azure_ml_deploy.py
│ │ │ └── gcp_vertex_deploy.py
│ │ ├── cicd/
│ │ │ ├── github_actions.yaml
│ │ │ └── gitlab_ci.yaml
│ │ ├── dags/
│ │ │ ├── inference_batch_dag.py
│ │ │ ├── retrain_dag.py
│ │ │ └── training_dag.py
│ │ ├── docker/
│ │ │ ├── inference.Dockerfile
│ │ │ ├── monitoring.Dockerfile
│ │ │ └── trainer.Dockerfile
│ │ ├── k8s/
│ │ │ ├── cv-batch-inference.yaml
│ │ │ ├── cv-inference-deployment.yaml
│ │ │ ├── cv-retrain-cronjob.yaml
│ │ │ ├── cv-trainer-job.yaml
│ │ │ └── service.yaml
│ │ ├── monitoring/
│ │ │ ├── exporters/
│ │ │ │ ├── data_drift_exporter.py
│ │ │ │ └── model_metrics_exporter.py
│ │ │ ├── grafana_dashboard.json
│ │ │ └── prometheus_config.yaml
│ │ └── terraform/
│ │ ├── main.tf
│ │ ├── outputs.tf
│ │ └── variables.tf
│ ├── shared_libs/
│ │ ├── data_ingestion/
│ │ │ ├── atomic/
│ │ │ │ ├── api_loader.py
│ │ │ │ ├── camera_stream_consumer.py
│ │ │ │ ├── dicom_loader.py
│ │ │ │ ├── image_loader.py
│ │ │ │ ├── kafka_consumer.py
│ │ │ │ └── video_loader.py
│ │ │ ├── base/
│ │ │ │ ├── base_loader.py
│ │ │ │ └── base_stream_consumer.py
│ │ │ ├── configs/
│ │ │ │ ├── ingestion_config.yaml
│ │ │ │ └── ingestion_config_schema.py
│ │ │ ├── factories/
│ │ │ │ ├── loader_factory.py
│ │ │ │ └── stream_factory.py
│ │ │ ├── orchestrator/
│ │ │ │ └── ingestion_orchestrator.py
│ │ │ └── utils/
│ │ │ ├── api_utils.py
│ │ │ ├── dicom_utils.py
│ │ │ ├── file_utils.py
│ │ │ └── validation_utils.py
│ │ ├── data_processing/
│ │ │ ├── \_base/
│ │ │ │ ├── base_augmenter.py
│ │ │ │ ├── base_embedder.py
│ │ │ │ ├── base_feature_extractor.py
│ │ │ │ └── base_image_cleaner.py
│ │ │ ├── augmenters/
│ │ │ │ ├── atomic/
│ │ │ │ │ ├── cutmix.py
│ │ │ │ │ ├── flip_rotate.py
│ │ │ │ │ ├── mixup.py
│ │ │ │ │ └── noise_injection.py
│ │ │ │ ├── augmenter_factory.py
│ │ │ │ └── augmenter_orchestrator.py
│ │ │ ├── cleaners/
│ │ │ │ ├── atomic/
│ │ │ │ │ ├── color_space_cleaner.py
│ │ │ │ │ ├── normalize_cleaner.py
│ │ │ │ │ └── resize_cleaner.py
│ │ │ │ ├── cleaner_factory.py
│ │ │ │ └── cleaner_orchestrator.py
│ │ │ ├── configs/
│ │ │ │ ├── augmentation_config.yaml
│ │ │ │ ├── config_schema.py
│ │ │ │ ├── feature_config.yaml
│ │ │ │ └── preprocessing_config.yaml
│ │ │ ├── embedders/
│ │ │ │ ├── atomic/
│ │ │ │ │ ├── cnn_embedder.py
│ │ │ │ │ └── vit_embedder.py
│ │ │ │ ├── embedder_factory.py
│ │ │ │ └── embedder_orchestrator.py
│ │ │ ├── feature_extractors/
│ │ │ │ ├── atomic/
│ │ │ │ │ ├── hog_extractor.py
│ │ │ │ │ ├── orb_extractor.py
│ │ │ │ │ └── sift_extractor.py
│ │ │ │ ├── feature_extractor_factory.py
│ │ │ │ └── feature_extractor_orchestrator.py
│ │ │ └── orchestrators/
│ │ │ ├── feature_pipeline_orchestrator.py
│ │ │ └── image_preprocessing_orchestrator.py
│ │ ├── feature_store/
│ │ │ ├── base/
│ │ │ │ ├── base_retriever.py
│ │ │ │ └── base_vector_store.py
│ │ │ ├── configs/
│ │ │ │ ├── feature_store_config.yaml
│ │ │ │ └── feature_store_config_schema.py
│ │ │ ├── connectors/
│ │ │ │ ├── chromadb_connector.py
│ │ │ │ ├── faiss_connector.py
│ │ │ │ ├── milvus_connector.py
│ │ │ │ ├── pinecone_connector.py
│ │ │ │ └── weaviate_connector.py
│ │ │ ├── factories/
│ │ │ │ ├── retriever_factory.py
│ │ │ │ └── vector_store_factory.py
│ │ │ ├── orchestrator/
│ │ │ │ └── feature_store_orchestrator.py
│ │ │ ├── retrievers/
│ │ │ │ ├── dense_retriever.py
│ │ │ │ ├── hybrid_retriever.py
│ │ │ │ └── reranker.py
│ │ │ └── utils/
│ │ │ ├── embedding_utils.py
│ │ │ ├── index_utils.py
│ │ │ └── metadata_utils.py
│ │ └── ml_core/
│ │ ├── configs/
│ │ │ ├── config_utils.py
│ │ │ ├── evaluator_config_schema.py
│ │ │ ├── pipeline_config.yaml
│ │ │ ├── pipeline_config_schema.py
│ │ │ └── trainer_config_schema.py
│ │ ├── evaluator/
│ │ │ ├── base/
│ │ │ │ ├── base_evaluator.py
│ │ │ │ └── base_explainer.py
│ │ │ ├── factories/
│ │ │ │ ├── evaluator_factory.py
│ │ │ │ ├── explainer_factory.py
│ │ │ │ └── metric_factory.py
│ │ │ ├── explainability/
│ │ │ │ ├── gradcam_explainer.py
│ │ │ │ ├── ig_explainer.py
│ │ │ │ ├── lime_explainer.py
│ │ │ │ └── shap_explainer.py
│ │ │ ├── metrics/
│ │ │ │ ├── classification_metrics.py
│ │ │ │ ├── detection_metrics.py
│ │ │ │ └── segmentation_metrics.py
│ │ │ ├── orchestrator/
│ │ │ │ └── evaluation_orchestrator.py
│ │ │ └── utils/
│ │ │ ├── report_utils.py
│ │ │ ├── threshold_utils.py
│ │ │ └── visualization_utils.py
│ │ ├── mlflow_service/
│ │ │ ├── base/
│ │ │ │ ├── base_registry.py
│ │ │ │ └── base_tracker.py
│ │ │ ├── configs/
│ │ │ │ ├── mlflow_config_schema.py
│ │ │ │ └── mlflow_default.yaml
│ │ │ ├── factories/
│ │ │ │ ├── registry_factory.py
│ │ │ │ └── tracker_factory.py
│ │ │ ├── implementations/
│ │ │ │ ├── mlflow_client_wrapper.py
│ │ │ │ ├── mlflow_logger.py
│ │ │ │ └── mlflow_registry.py
│ │ │ └── utils/
│ │ │ ├── mlflow_exceptions.py
│ │ │ └── retry_utils.py
│ │ ├── orchestrators/
│ │ │ ├── base/
│ │ │ │ └── base_orchestrator.py
│ │ │ ├── configs/
│ │ │ │ ├── default_training.yaml
│ │ │ │ └── orchestrator_config_schema.py
│ │ │ ├── cv_inference_orchestrator.py
│ │ │ ├── cv_pipeline_factory.py
│ │ │ └── cv_training_orchestrator.py
│ │ ├── pipeline_components_cv/
│ │ │ ├── atomic/
│ │ │ │ ├── cv_augmenter.py
│ │ │ │ ├── cv_dim_reducer.py
│ │ │ │ ├── cv_embedder.py
│ │ │ │ ├── cv_normalizer.py
│ │ │ │ └── cv_resizer.py
│ │ │ ├── base/
│ │ │ │ ├── base_component.py
│ │ │ │ └── base_validator.py
│ │ │ ├── configs/
│ │ │ │ ├── cv_component_config_schema.py
│ │ │ │ └── default_pipeline.yaml
│ │ │ ├── factories/
│ │ │ │ └── component_factory.py
│ │ │ ├── orchestrator/
│ │ │ │ └── component_orchestrator.py
│ │ │ └── utils/
│ │ │ ├── io_utils.py
│ │ │ ├── logging_utils.py
│ │ │ └── monitoring_utils.py
│ │ ├── retraining/
│ │ │ ├── base/
│ │ │ │ ├── base_retrain_orchestrator.py
│ │ │ │ └── base_trigger.py
│ │ │ ├── configs/
│ │ │ │ ├── retrain_config.yaml
│ │ │ │ └── retrain_config_schema.py
│ │ │ ├── orchestrator/
│ │ │ │ └── retrain_orchestrator.py
│ │ │ ├── scheduler/
│ │ │ │ ├── scheduler_airflow.py
│ │ │ │ └── scheduler_cron.py
│ │ │ ├── triggers/
│ │ │ │ ├── drift_trigger.py
│ │ │ │ ├── performance_trigger.py
│ │ │ │ └── time_trigger.py
│ │ │ └── utils/
│ │ │ ├── job_utils.py
│ │ │ └── notification_utils.py
│ │ ├── selector/
│ │ │ ├── base/
│ │ │ │ └── base_selector.py
│ │ │ ├── configs/
│ │ │ │ ├── selector_config_schema.py
│ │ │ │ └── selector_default.yaml
│ │ │ ├── factories/
│ │ │ │ └── selector_factory.py
│ │ │ ├── implementations/
│ │ │ │ ├── ensemble_selector.py
│ │ │ │ ├── metric_based_selector.py
│ │ │ │ └── rule_based_selector.py
│ │ │ └── utils/
│ │ │ ├── selection_exceptions.py
│ │ │ └── selection_logging.py
│ │ └── trainer/
│ │ ├── base/
│ │ │ ├── base_cv_trainer.py
│ │ │ ├── base_distributed_trainer.py
│ │ │ └── base_trainer.py
│ │ ├── factories/
│ │ │ └── trainer_factory.py
│ │ ├── implementations/
│ │ │ ├── automl_cv_trainer.py
│ │ │ ├── cnn_trainer.py
│ │ │ ├── contrastive_trainer.py
│ │ │ ├── finetune_trainer.py
│ │ │ ├── semi_supervised_trainer.py
│ │ │ └── transformer_trainer.py
│ │ └── utils/
│ │ ├── checkpoint_utils.py
│ │ ├── distributed_utils.py
│ │ ├── early_stopping.py
│ │ ├── gradient_clip.py
│ │ └── optimizer_utils.py
│ └── domain_models/
│ ├── medical_imaging/
│ │ ├── configs/
│ │ │ ├── evaluation_config.yaml
│ │ │ ├── model_config.yaml
│ │ │ ├── preprocessing_config.yaml
│ │ │ ├── service_config.yaml
│ │ │ │ └── training_config.yaml
│ │ ├── evaluators/
│ │ │ ├── domain_eval_adapter.py
│ │ │ └── domain_explainability_adapter.py
│ │ ├── pipelines/
│ │ │ ├── evaluation_pipeline.py
│ │ │ ├── inference_pipeline.py
│ │ │ ├── retraining_pipeline.py
│ │ │ └── training_pipeline.py
│ │ ├── schemas/
│ │ │ ├── evaluation_schema.py
│ │ │ ├── input_schema.py
│ │ │ ├── output_schema.py
│ │ │ └── processed_schema.py
│ │ ├── services/
│ │ │ ├── imaging_evaluator.py
│ │ │ ├── imaging_predictor.py
│ │ │ ├── imaging_service_orchestrator.py
│ │ │ └── imaging_trainer.py
│ │ ├── tests/
│ │ │ ├── test_inference_pipeline.py
│ │ │ ├── test_schemas.py
│ │ │ ├── test_service_orchestrator.py
│ │ │ └── test_training_pipeline.py
│ │ └── utils/
│ │ ├── config_utils.py
│ │ ├── medical_rules_utils.py
│ │ ├── postprocessing_utils.py
│ │ └── visualization_utils.py
│ └── main_app.py
│ └── README.md
│ └── requirements.txt
│ └── setup.py
│ └── .gitignore

#

.
├── cv_factory/
│   ├── infra/
│   │   ├── cloud/ .................... (Cloud Deployment Scripts: SageMaker, Azure ML, Vertex AI)
│   │   ├── cicd/ ..................... (CI/CD Pipeline: GitHub Actions/GitLab CI with Quality Gates)
│   │   ├── docker/ ................... (Container Definitions: trainer, inference, monitoring)
│   │   ├── k8s/ ...................... (Kubernetes Deployment/Scheduling: Jobs, Deployments, CronJobs)
│   │   ├── monitoring/ ............... (Prometheus/Grafana Configs and Exporters)
│   │   └── terraform/ ................ (IaC: Cloud Provisioning)
│
│   ├── shared_libs/ .................. (MLOps Platform Core - Reusable Components)
│   │
│   │   ├── core_utils/ ............... (Utilities: Config/File System/Exceptions - Centralized)
│   │   │   ├── config_manager.py ...... (Loads YAML/JSON and handles Pydantic validation via utilities)
│   │   │   ├── exceptions.py .......... (Base Factory Exception Hierarchy)
│   │   │   ├── file_system_utils.py ... (Cloud/Local Path checking, directory creation)
│   │   │   └── validation_utils.py .... (NumPy/Data integrity checks - e.g., check_numpy_dimension)
│   │
│   │   ├── data_ingestion/ ........... (I/O Connectors - Abstraction Layer)
│   │   │   ├── base/ ................. (Contracts)
│   │   │   │   ├── base_data_connector.py . (Contract: read(), write(), connect())
│   │   │   │   └── base_stream_connector.py (Contract: consume(), produce(), close())
│   │   │   ├── configs/ .............. (Local Configuration Schema - Ensures I/O params are correct)
│   │   │   │   └── ingestion_config_schema.py (Pydantic Schema: Validates Kafka, DICOM, API configs)
│   │   │   ├── connectors/ ........... (Concrete I/O Implementations - Adapters for I/O)
│   │   │   │   ├── image_connector.py
│   │   │   │   ├── kafka_connector.py
│   │   │   │   └── dicom_connector.py
│   │   │   └── factories/ ............ (Dependency Injection Tooling)
│   │   │       ├── connector_factory.py . (Creates BaseDataConnector instances)
│   │   │       └── stream_connector_factory.py (Creates BaseStreamConnector instances)
│   │
│   │   ├── data_processing/ .......... (Atomic Logic - Pure Math/Algorithms)
│   │   │   ├── augmenters/ ........... (Pure Augmentation Logic)
│   │   │   ├── cleaners/ ............. (Pure Cleaning/Normalization Logic)
│   │   │   └── feature_extractors/ .... (Pure Feature/Dim Reduction Logic - SIFT, PCA math)
│   │
├── feature_store/ ............ (FEATURE & VECTOR MANAGEMENT)
│   │   │   ├── base/ ................. (Contracts)
│   │   │   │   ├── base_vector_store.py . (Contract: connect, add, search, delete, update, close)
│   │   │   │   └── base_retriever.py .. (Contract: retrieve with filters)
│   │   │   ├── connectors/ ........... (Vector DB Implementations)
│   │   │   │   ├── pinecone_connector.py
│   │   │   │   ├── milvus_connector.py
│   │   │   │   └── faiss_connector.py
│   │   │   ├── factories/ ............ (VectorStoreFactory, RetrieverFactory)
│   │   │   ├── orchestrator/ ......... (FeatureStoreOrchestrator - CRUD Façade)
│   │   │   └── retrievers/ ........... (Retrieval Logic)
│   │   │       ├── dense_retriever.py
│   │   │       ├── hybrid_retriever.py
│   │   │       └── reranker.py
│   │   ├── inference/ ................ (Serving Contracts - Final API Gateway)
│   │   │   ├── base_cv_predictor.py ... (Contract: load_model, preprocess, predict, postprocess)
│   │   │   └── cv_predictor.py ........ (Implementation: Uses MLflow client & Preprocessing Orchestrator)
│   │
│   │   ├── ml_core/ .................. (Core ML Logic - The Engine)
│   │   │   ├── data/ ................. (Dataset Abstraction)
│   │   │   │   ├── base_cv_dataset.py . (Contract: **len**, **getitem**, prepare)
│   │   │   │   └── cv_dataset.py ...... (Implementation: Uses Connectors and Preprocessing Orchestrator)
│   │   │   ├── configs/ .............. (Strategic Config Schemas - Controls MLOps Workflow)
│   │   │   │   ├── evaluator_config_schema.py
│   │   │   │   ├── model_config_schema.py
│   │   │   │   └── orchestrator_config_schema.py (Master Config)
│   │   │   ├── pipeline_components_cv/ (Execution Engine Adapters)
│   │   │   │   ├── factories/ ........ (ComponentFactory)
│   │   │   │   ├── base/ ............. (BaseComponent, BaseValidator)
│   │   │   │   └── atomic/ ........... (CVResizer, CVCNNEmbedder, CVDimReducer - All are ADAPTERS)
│   │   │   ├── trainer/ .............. (Training Logic)
│   │   │   │   ├── base/ ............. (BaseCVTrainer, BaseTrainer)
│   │   │   │   └── utils/ ............ (distributed_utils, optimizer_utils, gradient_clip, checkpoint_utils)
│   │   │   └── mlflow_service/ ....... (MLOps Tracking & Registry)
│   │   │       ├── base/ ............. (BaseTracker, BaseRegistry)
│   │   │       └── implementations/ .. (MLflowClientWrapper, MLflowLogger - Uses the new tag/transition methods)
│   │
│   │   └── orchestrators/ ............ (Workflow Management - Top Level Controller)
│   │       ├── base/ ................. (BaseOrchestrator - Enforces DI, Structured Logging, Pydantic Check)
│   │       ├── cv_training_orchestrator.py (Master Workflow: Data Flow, DDP Execution, Model Tagging)
│   │       └── cv_inference_orchestrator.py (Workflow: Batch and Stream Inference)
│
│   └── domain_models/ ................ (Domain-Specific Logic - Adapter Layer)
│       └── medical_imaging/ .......... (Medical domain-specific config and rules)
│           ├── configs/ .............. (training_config.yaml, service_config.yaml)
│           ├── pipelines/ ............ (MedicalTrainingPipeline - Thin Façade Adapter)
│           └── services/ ............. (ImagingServiceOrchestrator - The actual external API entry point)
cv_factory/
└── api_service/
├── **init**.py
├── api_config.py .................... (Service settings: host, port, model_endpoint_name)
├── endpoints/
│ └── prediction_router.py ......... (Defines the /predict route)
├── schemas/ ......................... (Pydantic Input/Output Validation)
│ ├── service_schemas.py ........... (Input, Output, and Error Schemas)
└── clients/
└── cloud_inference_client.py .... (Calls the SageMaker/Vertex Endpoint)

#

/my_mlops_cv_project/
├── .gitignore
├── README.md
├── requirements.txt .................. (Dependencies for the entire project)
│
├── configs/ .......................... (CONFIGURATION MANAGEMENT - SỞ HỮU CỦA DỰ ÁN)
│   ├── training_pipeline_config.yaml . (File YAML điều khiển CV Factory: Bắt buộc Validation Pydantic)
│   └── inference_medical_config.yaml
│
├── data/ ............................. (Local data or data access scripts)
│   └── sample_xray_images/
│
├── scripts/ .......................... (Execution scripts for CI/CD)
│   ├── run_smoke_test.py ............. (Calls Factory with specific config for Model Quality Gate)
│   └── main_service_launcher.py ...... (The application entry point)
│
├── cv_factory/ ....................... (MLOPS FACTORY FRAMEWORK - CODE TÁI SỬ DỤNG)
│   ├── shared_libs/ .................. (The entire architecture we built)
│   │   └── ... (data_ingestion/, ml_core/, orchestrators/, inference/, etc.)
│   │
│   └── domain_models/ ................ (Domain Adapters for Medical Imaging)
│       └── medical_imaging/
│           └── ... (Logic for Postprocessor DI, Schemas nghiệp vụ)
│
└── infra_deployment/ ................. (IaC ASSETS - TRIỂN KHAI HẠ TẦNG)
    └── terraform/
        └── main.tf ................... (Provision EKS/Vertex/SageMaker and MLflow Tracking Server)

#

/my_mlops_cv_project/
├── configs/ .................... (Cấu hình YAML)
├── cv_factory/ ................. (MLOPS FACTORY FRAMEWORK - CHỈ CÓ CODE PYTHON/LOGIC)
│   └── shared_libs/ ............ (Code có thể tái sử dụng, không có files infra/cloud/k8s)
│
└── infra_deployment/ ........... (INFRASTRUCTURE & OPS ASSETS - VẬN HÀNH)
├── cicd/ ................... (GitHub Actions, GitLab CI)
├── cloud/ .................. (SageMakerDeployer.py, VertexDeployer.py)
├── dags/ ................... (Airflow DAGs)
├── k8s/ .................... (K8s Deployments, Jobs)
└── terraform/ .............. (main.tf)

# DATA FLOW MAP – TRAINING WORKFLOW

1. CỔNG VÀO (Gate)
   └── training_pipeline_config.yaml
   [Input] YAML thô
   [Quality Gate] validate bằng TrainingOrchestratorConfig (Pydantic Schema)

2. ĐIỀU PHỐI (Control Layer)
   └── CVTrainingOrchestrator.run() - validate_config() - logger.start_run() - \_prepare_data()
   [Lifecycle] Quản lý toàn bộ vòng đời Training

3. DỮ LIỆU (Data Prep Layer)
   └── CVDataset - ConnectorFactory (I/O) → load dữ liệu từ S3/local - CVPreprocessingOrchestrator → tiền xử lý
   [Output] DataLoader (có DistributedSampler nếu DDP)

4. LOGIC LÕI (Core ML Layer)
   └── BaseCVTrainer / CNNTrainer - setup DDP (distributed_utils) - optimizer_utils.init() - trainer.fit()
   [Reliability] Gradient Clipping + All-Reduce Sync

5. KẾT THÚC (Finalization Layer)
   └── CVTrainingOrchestrator - log final_metrics → MLflow - registry.register_model() (Rank 0) - registry.tag_model_version(git_sha)
   [Auditability] Kết quả huấn luyện trở thành Asset traceable

# MODULE CONTRACTS MAP – shared_libs/

A. CVTrainingOrchestrator
├── Contract: BaseTracker, BaseRegistry
├── Service: MLflowLogger / Registry
└── Principle: Dependency Injection (không tự tạo services)

B. CVDataset
├── Contract: BaseDataConnector
├── Service: ImageConnector
└── Principle: Separation of I/O (không quan tâm S3 hay local)

C. CVDataset (Preprocessing)
├── Contract: BaseComponent
├── Service: ComponentOrchestrator
└── Principle: Adapter Pattern (Resizer, Normalizer wrap logic toán học)

D. CVInferenceOrchestrator
├── Contract: BaseCVPredictor
├── Service: CVPredictor
└── Principle: Workflow Separation (chỉ gọi predict_pipeline)

E. CVPredictor
├── Contract: Domain Postprocessor (Inject)
├── Service: MedicalPostprocessor
└── Principle: Domain Isolation (core ML không dính nghiệp vụ)

# Training Workflow (Lưu trữ Đặc trưng và Trạng thái)

1. BẮT ĐẦU TRAINING (CVTrainingOrchestrator.run())
   |
   V
2. DATA PREP (CVDataset)
   |
   V
3. PREPROCESSING/FEATURE ENGINEERING (ComponentOrchestrator)
   |
   +-- A. Xử lý Trạng thái (Stateful Components) --+
   | |
   | CVDimReducer.fit(Data) <--- HỌC THAM SỐ PCA |
   | |
   +----------------------------------------------+
   |
   V
4. LƯU TRỮ VÀ GHI VECTOR (Feature Store Integration)
   |
   |-- a. LƯU PARAMETER TRẠNG THÁI (Artifacts)
   | |
   | |-- CVDimReducer.save(path)
   | └── FeatureStoreOrchestrator.persist(index_path)
   |
   |-- b. LƯU EMBEDDING DỮ LIỆU
   | |
   | └── FeatureStoreOrchestrator.index_embeddings(Embeddings, Metadata)
   | |
   | └── VectorStoreConnector.add_embeddings() (e.g., Pinecone/Milvus)
   |
   V
5. KẾT THÚC (MLflow Registry Tagging)

# Inference Workflow (Truy vấn Nâng cao)

1. BẮT ĐẦU INFERENCE (CVInferenceOrchestrator.run())
   |
   V
2. PREDICT CHUẨN BỊ (CVPredictor.predict_pipeline())
   |
   |-- a. LOAD TRẠNG THÁI: Tải lại tham số PCA đã lưu từ Feature Store
   | └── CVDimReducer.load(path) <--- Đảm bảo tính nhất quán giữa Train & Serve
   |
   |-- b. TRÍCH XUẤT QUERY VECTOR (Embedding)
   | └── CVPredictor.preprocess() → CVMaskedImageEmbedder → [Query Vector Q]
   |
   V
3. TRUY VẤN NÂNG CAO (Advanced Retrieval)
   |
   |-- FeatureStoreOrchestrator.search_embeddings(Q, top_k, filters)
   |
   |-- [Logic Truy vấn Phức tạp] ---------------------------------------------
   | |
   | |-- DenseRetriever.retrieve()
   | |-- HybridRetriever.retrieve() (Nếu cần kết hợp Sparse Search)
   | |-- Reranker.retrieve() (Sử dụng Cross-Encoder để tinh chỉnh)
   | |
   | └── VectorStoreConnector.search() (Thực hiện tìm kiếm trên Cluster)
   |
   V
4. POSTPROCESSING & QUYẾT ĐỊNH
   |
   └── Kết quả Retrieval được đưa vào Postprocessor Domain (Injector)
   └── Quyết định cuối cùng (ví dụ: Chẩn đoán bằng cách tham chiếu đến 5 bệnh nhân tương tự đã được tìm thấy).

# 🚀 A. TRAINING FLOW (End-to-End Huấn luyện)

1. CONFIG GATE
   └── training_pipeline_config.yaml
   [Quality Gate: Pydantic Schema Validation]

2. ORCHESTRATION
   └── CVTrainingOrchestrator.run() - validate_config() - logger.start_run() - \_prepare_data()

3. DATA PREP
   └── CVDataset
   └── ConnectorFactory (I/O: S3, Local, DB)
   └── CVPreprocessingOrchestrator

4. FEATURE ENGINEERING
   └── ComponentOrchestrator
   └── Stateful Components
   └── CVDimReducer.fit(Data)
   [Learn PCA Parameters]

5. FEATURE STORE INTEGRATION
   ├── Save State/Params
   │ ├── CVDimReducer.save(path)
   │ └── FeatureStoreOrchestrator.persist(index_path)
   │
   └── Save Embeddings
   └── FeatureStoreOrchestrator.index_embeddings(Embeddings, Metadata)
   └── VectorStoreConnector.add_embeddings()
   [Pinecone/Milvus/Weaviate]

6. CORE ML TRAINING
   └── BaseCVTrainer / CNNTrainer - setup DDP (distributed_utils) - optimizer_utils.init() - trainer.fit()
   [Reliability: gradient clipping + all-reduce sync]

7. FINALIZATION
   └── CVTrainingOrchestrator - final_metrics → MLflowLogger - registry.register_model() (Rank 0) - registry.tag_model_version(git_sha)
   [Auditability: model traceable + deployable]

# 🔎 B. INFERENCE FLOW (Serving & Advanced Retrieval)

1. START INFERENCE
   └── CVInferenceOrchestrator.run()

2. PREDICT PIPELINE
   ├── Load State
   │ └── CVDimReducer.load(path)
   │ [Consistency: train == serve]
   │
   └── Extract Query Vector
   └── CVPredictor.preprocess()
   └── CVMaskedImageEmbedder → Query Vector Q

3. ADVANCED RETRIEVAL
   └── FeatureStoreOrchestrator.search_embeddings(Q, top_k, filters)
   └── Retrieval Logic
   ├── DenseRetriever.retrieve()
   ├── HybridRetriever.retrieve()
   ├── Reranker.retrieve() [Cross-Encoder]
   └── VectorStoreConnector.search() [Vector DB Cluster]

4. POSTPROCESSING & DOMAIN DECISION
   └── Domain Postprocessor (Injected)
   └── Business Decision
   (VD: y tế → tham chiếu 5 bệnh nhân tương tự,
   tài chính → tìm khách hàng gian lận gần nhất)

# 🗂️ C. SHARED CONTRACTS (Trái tim – shared_libs/)

- CVTrainingOrchestrator → BaseTracker, BaseRegistry → MLflowLogger, Registry
  [Principle: Dependency Injection]

- CVDataset → BaseDataConnector → ImageConnector
  [Principle: Separation of I/O]

- CVDataset Preprocessing → BaseComponent → ComponentOrchestrator
  [Principle: Adapter Pattern]

- CVInferenceOrchestrator → BaseCVPredictor → CVPredictor
  [Principle: Workflow Separation]

- CVPredictor → Domain Postprocessor (Injected) → MedicalPostprocessor
  [Principle: Domain Isolation]

  # Flow End-to-End

  [Terraform/K8s] Tạo cluster → N nodes (mỗi node có M GPU)
  |
  v
  [K8s Job] Scheduler khởi tạo pods cho training
  |
  v
  [PyTorch DDP Init]

  - Assign rank/global_rank/local_rank
  - NCCL backend khởi động
    |
    v
    [Training Step]
  - Mỗi GPU xử lý 1 mini-batch
  - Gradient all-reduce (network sync)
    |
    v
    [Checkpoint Save]
  - Rank 0 lưu → S3/NFS (accessible bởi tất cả nodes)

  # Training Flow (Save Checkpoint)

  [GPU0 (Rank 0)] ----------------------------
  | Compute batch, backward
  | All-reduce gradient (sync với GPU1..N)
  | Update weights
  | ---> SAVE checkpoint.pt ---------------> [Shared Storage (S3/NFS/PVC)]
  |
  [GPU1 (Rank 1)] -----------\
  [GPU2 (Rank 2)] ------------> All-reduce gradient → Sync → Update weights
  [GPU3 (Rank 3)] -----------/
  | Không save checkpoint
  | (Tránh trùng file)

# Resume Flow (Load Checkpoint)

[Shared Storage (S3/NFS/PVC)] ---> checkpoint.pt
|
| READ BY ALL RANKS
v

---

[GPU0 (Rank 0)] load_state_dict(checkpoint)
[GPU1 (Rank 1)] load_state_dict(checkpoint)
[GPU2 (Rank 2)] load_state_dict(checkpoint)
[GPU3 (Rank 3)] load_state_dict(checkpoint)

---

|
---> Training tiếp tục từ epoch+1

# 🧭 MLflow Tracking System – End-to-End Data Flow

### 🧱 1. TRAINING STAGE (Logging Phase)

[Training Script / Orchestrator]
|
|--- log_param("lr", 0.001)
|--- log_metric("val_acc", 0.92)
|--- log_artifact("confusion_matrix.png")
|--- mlflow.sklearn.log_model(model, "model")
|
v
┌──────────────────────────────────────┐
│ MLflow Tracking Server │
│ (Receives logs from run) │
└──────────────────────────────────────┘

### 🗄️ 2. BACKEND STORE (Metadata Database)

PostgreSQL / MySQL / SQLite
│
├── experiments
│ └── id, name, artifact_location
│
├── runs
│ └── run_uuid, status, start_time, end_time, artifact_uri
│
├── metrics
│ └── key, value, step, timestamp
│
├── params
│ └── key, value
│
└── model_versions
└── name, version, source_artifact_uri, stage

### ☁️ 3. ARTIFACT STORE (File Storage)

S3 / MinIO / GCS / Local FS
│
└── <experiment_name>/<run_id>/
├── metrics/
│ ├── train_loss.csv
│ └── val_accuracy.csv
│
├── params/
│ ├── lr.txt
│ └── batch_size.txt
│
├── artifacts/
│ ├── model/
│ │ ├── MLmodel
│ │ ├── conda.yaml
│ │ └── model.pkl
│ ├── confusion_matrix.png
│ └── roc_curve.png
│
└── meta.yaml

### 🧩 4. MODEL REGISTRY

MLflow Model Registry
│
├── name: ResNetModel
├── versions:
│ ├── v1 → s3://mlflow-artifacts/ResNetModel/run_5af1234a/model
│ ├── v2 → s3://mlflow-artifacts/ResNetModel/run_7be8f9e2/model
│
└── stages:
├── "Staging"
└── "Production"

### 🚀 5. DEPLOYMENT / SERVING

mlflow models serve -m "models:/ResNetModel/Production" -p 5000
|
v
[Local REST API]
├── POST /invocations
└── JSON → Prediction

Hoặc:

[SageMaker Adapter / Vertex AI Adapter]
|
└── Upload model.tar.gz + MLmodel
Deploy endpoint → Scalable Inference

### 🔄 6. MONITORING / RE-TRAINING LOOP

[Prometheus + MLflow UI]
|
├── Monitor metrics drift, model latency
└── Trigger retraining → MLflow new run

#

infra_library/
│
├── terraform_modules/ # Cấp thấp: các module cơ bản (atomic infrastructure)
│ ├── network/ # VPC, Subnet, Security Group
│ ├── s3_bucket/ # Lưu model, artifact, tfstate, logs
│ ├── rds_postgres/ # Metadata Store (MLflow, backend)
│ ├── eks_cluster/ # Kubernetes cluster cho training/inference jobs
│ ├── mlflow_stack/ # MLflow tracking + MinIO + Prometheus + Grafana
│ └── sagemaker_endpoint/ # Model deployment hạ tầng (cloud adapter)
│
└── templates/ # Cấp cao: tổ hợp module cho từng dự án
├── project_cv_factory/ # Gọi module network + s3 + rds + sagemaker
├── project_nlp_factory/ # Gọi network + s3 + rds + eks + vertex adapter
└── project_genai_platform/ # Full stack + API Gateway + Bedrock integration

#

terraform_modules/s3_bucket/
│
├── main.tf ................. (Logic tạo S3 + IAM policy)
├── variables.tf ............ (Input config: bucket_name, versioning, acl,...)
├── outputs.tf .............. (Output: bucket_name, arn, url)
└── README.md ............... (Giải thích input/output)

#

templates/project_cv_factory/
│
├── main.tf
│ ├── module "network" { source = "../../terraform_modules/network" }
│ ├── module "s3_bucket" { source = "../../terraform_modules/s3_bucket" }
│ ├── module "rds" { source = "../../terraform_modules/rds_postgres" }
│ └── module "sagemaker" {
│ source = "../../terraform_modules/sagemaker_endpoint"
│ vpc_id = module.network.vpc_id
│ bucket_arn = module.s3_bucket.bucket_arn
│ }
│
├── variables.tf
├── outputs.tf
└── envs/
├── dev.tfvars
├── staging.tfvars
└── prod.tfvars

#

cd templates/project_cv_factory/
terraform init
terraform workspace new dev
terraform plan -var-file=envs/dev.tfvars
terraform apply -var-file=envs/dev.tfvars

# Dependency Injection + Inversion of Control.

             ┌──────────────────────────────┐
             │ domain_models/               │
             │   └── medical_imaging/       │
             │       └── postprocessor.py   │
             └──────────────┬───────────────┘
                            │  (import bởi Orchestrator)
                            ▼

┌───────────────────────────┴───────────────────────────┐
│ orchestrators/cv_inference_orchestrator.py │
│ - tạo instance: MedicalPostprocessor(), Model(), ... │
│ - inject vào CVPredictor(...) │
└───────────────┬───────────────────────────────────────┘
│ (dependency injection)
▼
┌───────────────────────────────────────────────────────┐
│ shared_libs/inference/predictor.py │
│ - nhận preprocessor, model, postprocessor │
│ - chạy predict_pipeline() │
└───────────────────────────────────────────────────────┘

#

cv_factory/
├── domain_models/
│ └── medical_imaging/ # (Logic Nghiệp vụ/Domain-Specific)
│ ├── configs/
│ │ └── domain_config.yaml # Config độc lập cho domain (e.g., ngưỡng chẩn đoán)
│ ├── factory/
│ │ └── domain_factory.py # Tạo và cấu hình các đối tượng domain (Postprocessor, Entities)
│ ├── model/
│ │ └── medical_entity.py # Định nghĩa cấu trúc dữ liệu kết quả cuối cùng
│ ├── postprocessor/
│ │ └── medical_postprocessor.py # Logic nghiệp vụ (e.g., áp dụng ngưỡng chẩn đoán)
│ └── utils/
│ └── dicom_parser.py # Utilities chuyên biệt (e.g., phân tích cú pháp DICOM)
│
└── shared_libs/ # (Các thành phần dùng chung, Agnostic)
├── core_utils/ # (Utilities nền tảng và Exception)
│ ├── config_manager.py # Tải và Validate config (YAML/JSON + Pydantic)
│ ├── exceptions.py # Định nghĩa các Exception tùy chỉnh (DataIntegrityError, v.v.)
│ ├── file_system_utils.py # Tiện ích I/O (kiểm tra cloud URI, tạo thư mục local)
│ └── validation_utils.py # Hàm kiểm tra NumPy array (dimension, shape)
│
├── data_ingestion/ # (I/O Connectors - Đọc/Ghi dữ liệu thô)
│ ├── base/
│ │ ├── base_data_connector.py
│ │ └── base_stream_connector.py
│ ├── connectors/ # (Các Connector Cụ thể)
│ │ ├── api_connector.py
│ │ ├── camera_stream_connector.py
│ │ ├── dicom_connector.py
│ │ ├── image_connector.py
│ │ ├── kafka_connector.py
│ │ └── video_connector.py
│ └── factories/
│ └── connector_factory.py
│
├── data_processing/ # (Pipeline Tiền xử lý - CLEAN -> AUGMENT -> FEATURE)
│ ├── \_base/
│ │ ├── base_augmenter.py
│ │ ├── base_image_cleaner.py
│ │ └── base_feature_extractor.py
│ ├── augmenters/
│ │ └── augmenter_orchestrator.py # Điều phối pipeline Augmentation
│ ├── cleaners/
│ │ └── cleaner_orchestrator.py # Điều phối pipeline Cleaning
│ ├── configs/
│ │ └── preprocessing_config_schema.py # Master schema cho Preprocessing
│ ├── embedders/
│ │ └── embedder_orchestrator.py # Điều phối pipeline Embedding
│ ├── feature_extractors/
│ │ └── feature_extractor_orchestrator.py # Điều phối pipeline Feature Extraction
│ └── orchestrators/
│ └── cv_preprocessing_orchestrator.py # Façade/Master Orchestrator cho Data Processing
│
├── inference/ # (Logic Thực thi Mô hình)
│ ├── base_cv_predictor.py # Contract/Interface cho mọi Predictor
│ └── cv_predictor.py # Predictor cụ thể (Load MLflow, PyTorch/Adapter Router)
│
├── ml_core/ # (ML Framework/MLOps Cốt lõi)
│ ├── data/
│ │ ├── base_cv_dataset.py # Base Dataset cho PyTorch/TF
│ │ └── cv_dataset.py # Dataset cụ thể (kết hợp Connector + Preprocessor)
│ ├── mlflow_service/
│ │ └── implementations/
│ │ └── mlflow_client_wrapper.py # (Giả định)
│ └── pipeline_components_cv/
│ └── orchestrator/
│ └── component_orchestrator.py # Execution Engine cho chuỗi Component (fit/transform)
│
└── orchestrators/ # (Workflow Controllers - Tầng Cao nhất)
├── base/
│ └── base_orchestrator.py
├── cv_inference_orchestrator.py # Điều phối Batch/Single Inference
├── cv_stream_inference_orchestrator.py # Điều phối Real-time Stream Inference
├── cv_training_orchestrator.py # Điều phối Training (Data -> Train -> Eval -> Register)
└── cv_pipeline_factory.py # Factory Cấp Cao (Thực hiện Dependency Injection)

#

domain_models/
└── medical_imaging/
├── configs/
│ └── domain_config.yaml # (Placeholder)
├── factory/
│ └── domain_factory.py # (Logic lắp ráp Domain)
├── model/
│ └── medical_entity.py # (Final Diagnosis Entity)
├── schema/
│ ├── input_schema.py # Input Validation (e.g., Image bytes)
│ ├── processed_schema.py # Processed Data Validation
│ ├── output_schema.py # Model Raw Output
│ └── evaluation_schema.py # Evaluation Report Structure
├── postprocessor/
│ └── medical_postprocessor.py # CHỨA LOGIC CỦA postprocessing_utils.py
├── evaluator/ # (MLOps Adapters)
│ ├── medical_eval_adapter.py # CHỨA LOGIC CỦA domain_eval_adapter.py
│ └── medical_explain_adapter.py # CHỨA LOGIC CỦA domain_explainability_adapter.py
└── utils/ # (Domain Utilities)
├── **init**.py # (Đã cập nhật)
├── visualization_utils.py # (Giữ nguyên logic plot/heatmap)
└── validation_utils.py # CHỨA LOGIC CỦA medical_rules_utils.py

# multi-factory architecture

project_root/
├── infra_deployment/ # IaC & Cloud setup (Terraform, Docker, CI/CD)
│ ├── terraform/
│ ├── cloud/
│ └── cicd_factory/
│
├── configs/ # Global + domain configs
│ ├── global_config.yaml
│ ├── deployment.yaml
│ ├── cv_config.yaml
│ ├── nlp_config.yaml
│ └── genai_config.yaml
│
├── shared_libs/ # Shared utilities (logging, ML core, DI, adapters)
│ ├── ml_core/
│ ├── adapters/
│ ├── monitoring/
│ └── utils/
│
├── cv_factory/ # Application Factory for CV
│ ├── orchestrators/
│ ├── domain_models/
│ ├── api_service/
│ └── registry/
│
├── nlp_factory/ # Application Factory for NLP
│ ├── orchestrators/
│ ├── domain_models/
│ ├── api_service/
│ └── registry/
│
├── genai_factory/ # Application Factory for GenAI
│ ├── orchestrators/
│ ├── domain_models/
│ ├── api_service/
│ └── registry/
│
└── ml_factory/ # Core reusable ML components
├── feature_engineering/
├── model_selection/
├── pipeline_builder/
└── training_orchestrator/

#

shared_libs/
├── ml_core/
│ ├── base/ # Base Contracts & Interfaces (Rất chung)
│ ├── monitoring/ # Logic Drift/Alert/Reporter (Có thể dùng cho NLP)
│ └── retraining/ # Logic Trigger/Scheduler (Có thể dùng cho NLP)
│
└── ml_cv_core/ # THƯ MỤC MỚI (CHUYÊN BIỆT CV)
├── data/
│ └── cv_dataset.py # (Dùng chung cho CV)
├── inference/
│ └── cv_predictor.py
└── pipeline_components_cv/
└── orchestrator/
└── component_orchestrator.py

#

shared_libs/
├── core_utils/ # (GENERAL)
├── data_ingestion/ # (GENERAL - Connectors)
├── ml_core/ # (GENERAL MLOPS - Monitoring, Retraining)
│ └── monitoring/
│
├── data_processing/
│ ├── cv_pipelines/ # (CV-SPECIFIC)
│ └── nlp_pipelines/ # (NLP-SPECIFIC)
│
└── ml_cv_core/ # (CV-SPECIFIC CORE)
├── data/
├── inference/
└── orchestrators/

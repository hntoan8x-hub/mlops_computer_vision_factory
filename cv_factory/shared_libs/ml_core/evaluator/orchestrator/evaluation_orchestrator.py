# cv_factory/shared_libs/ml_core/evaluator/evaluation_orchestrator.py

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union

from shared_libs.ml_core.evaluator.base.base_evaluator import BaseEvaluator
from shared_libs.ml_core.evaluator.base.base_metric import BaseMetric 
from shared_libs.ml_core.evaluator.factories.metric_factory import MetricFactory
from shared_libs.ml_core.evaluator.factories.explainer_factory import ExplainerFactory
from shared_libs.ml_core.evaluator.utils.visualization_utils import visualize_metrics, visualize_explanations

# IMPORT CẦN THIẾT CHO CHUẨN HÓA ĐẦU RA
from shared_libs.ml_core.output_adapter.output_adapter_factory import OutputAdapterFactory
from shared_libs.ml_core.output_adapter.base_output_adapter import BaseOutputAdapter, StandardizedOutput

logger = logging.getLogger(__name__)

class EvaluationOrchestrator(BaseEvaluator):
    """
    Orchestrates the end-to-end model evaluation process.
    
    This class integrates the OutputAdapter to standardize model predictions 
    before feeding them into Stateful Metrics.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_to_compute: Dict[str, BaseMetric] = {}
        self.explainers_to_run: List[Any] = []
        self.task_type = config.get("task_type", "classification")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric_config = config.get("metrics_config", {})
        
        # 1. KHỞI TẠO OUTPUT ADAPTER
        self.output_adapter: BaseOutputAdapter = OutputAdapterFactory.get_adapter(
            task_type=self.task_type,
            config=config.get("adapter_config", {})
        )
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initializes the metric instances and explainer configurations from the config.
        """
        # 2. KHỞI TẠO STATEFUL METRICS (qua MetricFactory)
        metrics_list = self.config.get("metrics", [])
        for metric_name in metrics_list:
            metric_config = self.metric_config.get(metric_name, {})
            metric_instance = MetricFactory.get_metric(metric_name, metric_config)
            
            # Reset trạng thái metric trước khi sử dụng
            metric_instance.reset() 
            self.metrics_to_compute[metric_name] = metric_instance

        # Initialize Explainers (giữ nguyên logic)
        explainers_config = self.config.get("explainers", [])
        for explainer_config in explainers_config:
            self.explainers_to_run.append(explainer_config)

        logger.info(f"EvaluationOrchestrator initialized for task: {self.task_type}.")


    def evaluate(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                 **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a full evaluation, including metric computation and explanation generation.
        """
        model.eval()
        
        # 1. Thu thập dữ liệu và Cập nhật Metrics
        self._collect_and_update_metrics(model, data_loader)

        # 2. Tính toán Metric cuối cùng
        results: Dict[str, Any] = {"metrics": {}, "detailed_metrics": {}}
        for metric_name, metric_instance in self.metrics_to_compute.items():
            computed_value = metric_instance.compute()
            
            # Xử lý kết quả phức tạp (Dict cho mAP, IoU per class)
            if isinstance(computed_value, dict):
                 results["metrics"][metric_name] = computed_value.get(metric_name) # Lưu giá trị tổng quan
                 results["detailed_metrics"].update(computed_value) # Lưu chi tiết
            else:
                 results["metrics"][metric_name] = computed_value
                 
            # Reset Metric sau khi tính toán xong
            metric_instance.reset()
        
        logger.info(f"All metrics computed: {results['metrics']}")

        # 3. Generate Explanations (Logic giữ nguyên)
        explainability_config = self.config.get("explainers", [])
        if explainability_config:
            results["explanations"] = self._run_explainers(model, data_loader, explainability_config)
            logger.info("Explanations generated.")
        
        # 4. Visualization (optional)
        if kwargs.get("visualize_explanations", False) and "explanations" in results:
            visualize_explanations(results["explanations"])
            logger.info("Explanations visualized.")
            
        return results

    def _collect_and_update_metrics(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> None:
        """
        Runs inference, collects raw outputs, passes them through the OutputAdapter,
        và updates all configured metrics.
        """
        logger.info("Starting inference and metric accumulation...")
        
        # Giả định DataLoader trả về (inputs, targets_tensor)
        with torch.no_grad():
            for inputs, targets_tensor in data_loader:
                inputs = inputs.to(self.device)
                
                # 1. Core Prediction
                raw_output = model(inputs)
                
                # 2. ADAPTER: Chuẩn hóa đầu ra thô của mô hình
                predictions_standardized: StandardizedOutput = self.output_adapter.adapt(
                    raw_output, 
                    image_size=inputs.shape[2:] 
                )
                
                # 3. Targets Standardization (Chuẩn hóa Ground Truth)
                # Chuyển đổi Target Tensor sang Input Data chuẩn (NumPy)
                targets_standardized = self.output_adapter._to_numpy(targets_tensor) 


                # 4. Tích lũy Metrics
                for metric_instance in self.metrics_to_compute.values():
                    metric_instance.update(
                        predictions=predictions_standardized, 
                        targets=targets_standardized
                    )
        
        logger.info("Inference and metric accumulation completed.")


    def _run_explainers(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                        explainer_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Khởi tạo và chạy các Explainer đã cấu hình trên một tập hợp con dữ liệu (đã triển khai trước đó).
        """
        if not explainer_configs:
            return {}

        results: Dict[str, Any] = {}
        model_device = next(model.parameters()).device
        
        num_samples_to_explain = self.config.get("max_explainer_samples", 5)
        
        # Sử dụng DataLoader để lấy mẫu
        for explainer_config in explainer_configs:
            explainer_type = explainer_config.get("type")
            
            try:
                explainer = ExplainerFactory.get_explainer(
                    explainer_type=explainer_type, 
                    model=model,
                    config=explainer_config.get("params", {})
                )
            except Exception as e:
                logger.error(f"Failed to initialize explainer '{explainer_type}': {e}")
                continue
            
            explanations_list: List[Dict[str, Any]] = []
            
            # Lặp qua các mẫu trong DataLoader (chỉ lấy N mẫu đầu tiên)
            for i, (inputs, targets) in enumerate(data_loader):
                if i >= num_samples_to_explain:
                    break
                    
                inputs = inputs.to(model_device)
                target_class = targets[0].item()

                explanation_output = explainer.explain(
                    model=model, 
                    image=inputs[0].unsqueeze(0), 
                    target_class=target_class
                )
                
                # Trực quan hóa (tùy chọn)
                if explainer_config.get("visualize_sample", False):
                    visualized_img = explainer.visualize(
                        explanation=explanation_output, 
                        image=inputs[0].cpu().numpy()
                    )
                    explanations_list.append({
                        "original_image_tensor": inputs[0].cpu(),
                        "visualized_image": visualized_img,
                        "explanation_data": explanation_output
                    })
                else:
                     explanations_list.append({"explanation_data": explanation_output})
            
            results[explainer_type] = explanations_list
            
        return results


    def log_metrics(self, metrics: Dict[str, Any], logger_instance: Any) -> None:
        """Logs metrics to the provided logger instance."""
        if not metrics:
            logger.warning("No metrics to log.")
            return
            
        for key, value in metrics.items():
            if not isinstance(value, dict):
                 logger_instance.log_metric(key, value)
            
        logger.info("Metrics successfully logged.")
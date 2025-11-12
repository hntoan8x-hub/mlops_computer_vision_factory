# shared_libs/ml_core/selector/selector_service.py

import logging
from typing import Dict, Any, List, Optional
from shared_libs.ml_core.mlflow_service.mlflow_service import MLflowService # Façade MLOps
from shared_libs.ml_core.selector.factories.selector_factory import SelectorFactory # Factory đã có

logger = logging.getLogger(__name__)

class ModelSelectorService:
    """
    Service chịu trách nhiệm tìm kiếm các Run và sử dụng Selector Engine 
    để chọn ra mô hình tốt nhất.
    """
    def __init__(self, mlflow_service: MLflowService, config: Dict[str, Any]):
        self.mlflow_service = mlflow_service
        self.config = config
        self.selector_config = config.get("selector", {})

    def select_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm và chọn mô hình dựa trên tiêu chí.
        """
        selector_type = self.selector_config.get("type", "metric_based")
        search_config = self.selector_config.get("search", {})
        
        # 1. SEARCH: Tìm kiếm các Run từ MLflow
        candidates = self.mlflow_service.search_runs(
            experiment_ids=search_config.get("experiment_ids"),
            filter_string=search_config.get("filter_string", ""),
            order_by=search_config.get("order_by", ["metrics.accuracy DESC"])
        )
        
        if not candidates:
            logger.warning("No candidate runs found for model selection.")
            return None

        # 2. SELECT: Khởi tạo và chạy Selector Engine
        selector_instance = SelectorFactory.create(
            selector_type=selector_type, 
            config=self.selector_config.get("params")
        )
        
        # Giả định BaseSelector.select() nhận List[Candidate Dict]
        best_candidate = selector_instance.select(candidates) 
        
        if best_candidate:
            logger.info(f"Best model selected: Run ID {best_candidate['run_id']} with metrics: {best_candidate['metrics']}")
            return best_candidate
            
        return None
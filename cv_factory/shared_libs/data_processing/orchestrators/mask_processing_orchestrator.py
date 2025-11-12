# shared_libs/data_processing/mask_components/mask_processing_orchestrator.py
import logging
import time
from typing import Dict, Any, Optional
import numpy as np

# Import Factory
from .mask_processor_factory import MaskProcessorFactory 

# Import Atomic Components (để sử dụng trong type hint)
from .mask_loader import MaskLoader 
from .mask_normalizer import MaskNormalizer
from .mask_augmenter import MaskAugmenter
from .mask_validator import MaskValidator

logger = logging.getLogger(__name__)

class MaskProcessingOrchestrator:
    """
    Orchestrates the full Mask preprocessing pipeline: Load -> Normalize -> Augment -> Validate.
    
    Quản lý luồng xử lý đồng bộ giữa RGB Image và Mask/Label Map/Bounding Box.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo Orchestrator bằng cách xây dựng tất cả các thành phần từ cấu hình Pydantic đã được validate.
        """
        self.config = config
        logger.info("Initializing MaskProcessingOrchestrator...")
        
        # Load components (Giả định config là dictionary từ Pydantic MaskProcessingConfig)
        # Các key là tên component (loader, normalizer...) và giá trị là ComponentStepConfig
        
        # NOTE: Truyền params từ ComponentStepConfig.params vào Factory
        self.loader: MaskLoader = MaskProcessorFactory.create("loader", config.get("loader", {}))
        self.normalizer: MaskNormalizer = MaskProcessorFactory.create("normalizer", config.get("normalizer", {}))
        self.augmenter: MaskAugmenter = MaskProcessorFactory.create("augmenter", config.get("augmenter", {}))
        self.validator: MaskValidator = MaskProcessorFactory.create("validator", config.get("validator", {}))
        
        # Thứ tự thực thi tuần tự
        self._pipeline = [
            ("loader", self.loader), 
            ("normalizer", self.normalizer),
            ("augmenter", self.augmenter),
            ("validator", self.validator)
        ]
        logger.info("Mask processing pipeline constructed successfully.")
        
    def run(self, rgb_path: str, mask_path: str) -> Dict[str, np.ndarray]:
        """
        Thực thi toàn bộ pipeline xử lý Mask.
        """
        start_time = time.time()
        
        # 1. Load (Bước đặc biệt, nhận paths)
        # Loader xử lý tải và căn chỉnh, trả về mảng np.ndarray (RGB) và MaskData
        load_result = self.loader.process(rgb_path, mask_path) 
        current_rgb, current_mask = load_result["rgb"], load_result["mask"]
        
        # 2. Sequential Processing
        for name, component in self._pipeline[1:]: 
            
            component_config = self.config.get(name, {})
            # Kiểm tra enabled flag trong cấu hình Pydantic
            if not component_config.get('enabled', True):
                 continue

            try:
                step_start = time.time()
                # Tất cả component đều nhận current_rgb và current_mask (đồng bộ)
                result = component.process(current_rgb, current_mask) 
                
                # Cập nhật kết quả
                current_rgb = result.get("rgb", current_rgb)
                current_mask = result.get("mask", current_mask)
                
                logger.debug(f"Component '{name}' finished in {time.time() - step_start:.4f}s.")
            except Exception as e:
                logger.error(f"Component '{name}' failed with error: {e}")
                raise

        total_time = time.time() - start_time
        logger.info(f"Mask processing completed successfully in {total_time:.4f}s.")
        
        return {"rgb": current_rgb, "mask": current_mask}

    def fit(self, X: Any, y: Optional[Any] = None) -> 'MaskProcessingOrchestrator':
        logger.info("Mask pipeline fit called (Delegating to sub-components).")
        # Logic fit cụ thể nếu có stateful component (ví dụ: MaskPCA)
        return self

    def save(self, directory_path: str) -> None:
        logger.info("Mask pipeline state saved.")
        pass
        
    def load(self, directory_path: str) -> None:
        logger.info("Mask pipeline state loaded.")
        pass
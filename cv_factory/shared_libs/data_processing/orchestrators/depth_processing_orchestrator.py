# shared_libs/data_processing/depth_components/depth_processing_orchestrator.py
import logging
import time
from typing import Dict, Any, Tuple
import numpy as np
from .depth_processor_factory import DepthProcessorFactory

logger = logging.getLogger(__name__)

class DepthProcessingOrchestrator:
    """
    Orchestrates the full Depth preprocessing pipeline: Load -> Normalize -> Augment -> Validate.
    
    Đây là đầu vào cho các bài toán Depth Estimation, chịu trách nhiệm chuẩn hóa 
    cặp RGB/Depth từ file path thành tensor đã sẵn sàng cho mô hình.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo Orchestrator bằng cách xây dựng tất cả các thành phần.

        Args:
            config (Dict[str, Any]): Cấu hình cho toàn bộ pipeline DepthProcessing.
        """
        # Hardening: Log the configuration upon initialization
        self.config = config
        logger.info("Initializing DepthProcessingOrchestrator...")
        
        # Load components using the Factory (tên ngắn được dùng làm key)
        self.loader = DepthProcessorFactory.create("loader", config.get("loader", {}))
        self.normalizer = DepthProcessorFactory.create("normalizer", config.get("normalizer", {}))
        self.augmenter = DepthProcessorFactory.create("augmenter", config.get("augmenter", {}))
        self.validator = DepthProcessorFactory.create("validator", config.get("validator", {}))
        # Converter là tùy chọn, có thể không cần thiết trong mọi config
        self.converter = DepthProcessorFactory.create("converter", config.get("converter", {}))
        
        self._pipeline = [
            ("loader", self.loader), 
            ("converter", self.converter), 
            ("normalizer", self.normalizer),
            ("augmenter", self.augmenter),
            ("validator", self.validator)
        ]
        logger.info("Depth processing pipeline constructed successfully.")

    def run(self, rgb_path: str, depth_path: str) -> Dict[str, np.ndarray]:
        """
        Thực thi toàn bộ pipeline xử lý Depth.

        Args:
            rgb_path (str): Đường dẫn đến ảnh RGB.
            depth_path (str): Đường dẫn đến Depth Map.

        Returns:
            Dict[str, np.ndarray]: Kết quả chứa 'rgb' và 'depth' tensors đã được chuẩn hóa.
        """
        start_time = time.time()
        
        # 1. Load (Bước đặc biệt vì cần 2 đường dẫn đầu vào)
        try:
            load_result = self.loader.process(rgb_path, depth_path)
            rgb_img, depth_map = load_result["rgb"], load_result["depth"]
            logger.debug(f"Loader finished. RGB: {rgb_img.shape}, Depth: {depth_map.shape}")
        except Exception as e:
            logger.error(f"DepthLoader failed for paths '{rgb_path}', '{depth_path}': {e}")
            raise
        
        # 2. Sequential Processing (Converter -> Normalizer -> Augmenter -> Validator)
        # Bắt đầu với kết quả từ Loader
        current_rgb = rgb_img
        current_depth = depth_map
        
        # Các bước tiếp theo chỉ xử lý trên (RGB, Depth)
        for name, component in self._pipeline[1:]: # Bỏ qua Loader đã chạy
            
            # Nếu component không được bật (ví dụ: converter) hoặc không phải là converter
            # nhưng process trả về một flag để skip. Ở đây ta chỉ kiểm tra enabled.
            if hasattr(component, 'enabled') and not component.enabled:
                continue

            try:
                step_start = time.time()
                result = component.process(current_rgb, current_depth)
                
                # Cập nhật kết quả cho vòng lặp tiếp theo
                current_rgb = result.get("rgb", current_rgb)
                current_depth = result.get("depth", current_depth)
                
                # HARDENING: Log timing
                logger.debug(f"Component '{name}' finished in {time.time() - step_start:.4f}s.")
            except Exception as e:
                # HARDENING: Log failure per component
                logger.error(f"Component '{name}' failed with error: {e}")
                # Quyết định: Tiếp tục với dữ liệu cũ hay dừng lại? Ta chọn dừng.
                raise

        total_time = time.time() - start_time
        logger.info(f"Depth processing completed successfully in {total_time:.4f}s.")
        
        return {"rgb": current_rgb, "depth": current_depth}
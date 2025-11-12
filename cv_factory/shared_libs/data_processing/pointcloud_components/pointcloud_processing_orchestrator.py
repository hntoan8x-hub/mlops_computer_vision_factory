# shared_libs/data_processing/pointcloud_components/pointcloud_processing_orchestrator.py
import logging
import time
from typing import Dict, Any, Optional
import numpy as np
from .pointcloud_processor_factory import PointcloudProcessorFactory
from .base_pointcloud_processor import PointcloudData

logger = logging.getLogger(__name__)

class PointcloudProcessingOrchestrator:
    """
    Orchestrates the full Point Cloud preprocessing pipeline: 
    Load -> Normalize -> Augment -> Voxelize.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo Orchestrator bằng cách xây dựng tất cả các thành phần.
        """
        self.config = config
        logger.info("Initializing PointcloudProcessingOrchestrator...")
        
        # Load components (Tên được chuẩn hóa: pc_loader, pc_normalizer, etc.)
        self.loader = PointcloudProcessorFactory.create("pc_loader", config.get("loader", {}))
        self.normalizer = PointcloudProcessorFactory.create("pc_normalizer", config.get("normalizer", {}))
        self.augmenter = PointcloudProcessorFactory.create("pc_augmenter", config.get("augmenter", {}))
        self.voxelizer = PointcloudProcessorFactory.create("pc_voxelizer", config.get("voxelizer", {}))
        
        # Thứ tự thực thi tuần tự (Lưu ý: Voxelization thường là bước cuối)
        self._pipeline = [
            ("loader", self.loader), 
            ("normalizer", self.normalizer),
            ("augmenter", self.augmenter),
            ("voxelizer", self.voxelizer) 
        ]
        logger.info("Point Cloud processing pipeline constructed successfully.")
        
    def run(self, pc_path: str) -> PointcloudData:
        """
        Thực thi toàn bộ pipeline xử lý Point Cloud.

        Args:
            pc_path (str): Đường dẫn đến file Point Cloud (.pcd, .bin, ...).

        Returns:
            PointcloudData: Dữ liệu Point Cloud (N, M) hoặc Voxel Grid (L, W, H, C) tensor.
        """
        start_time = time.time()
        
        # 1. Load (Bước đặc biệt, nhận path)
        current_pc = self.loader.process(pc_path) 
        logger.debug(f"Loader finished. PC Shape: {current_pc.shape}")
        
        # 2. Sequential Processing
        for name, component in self._pipeline[1:]: 
            
            component_config = self.config.get(name, {})
            if not component_config.get('enabled', True):
                 continue

            try:
                step_start = time.time()
                current_pc = component.process(current_pc) 
                
                logger.debug(f"Component '{name}' finished in {time.time() - step_start:.4f}s. New Shape: {current_pc.shape}")
            except Exception as e:
                logger.error(f"Component '{name}' failed with error: {e}")
                raise

        total_time = time.time() - start_time
        logger.info(f"Point Cloud processing completed successfully in {total_time:.4f}s.")
        
        return current_pc

    def fit(self, X: Any, y: Optional[Any] = None) -> 'PointcloudProcessingOrchestrator':
        logger.info("PC pipeline fit called (No stateful components implemented).")
        return self

    def save(self, directory_path: str) -> None:
        logger.info("PC pipeline state saved.")
        pass
        
    def load(self, directory_path: str) -> None:
        logger.info("PC pipeline state loaded.")
        pass
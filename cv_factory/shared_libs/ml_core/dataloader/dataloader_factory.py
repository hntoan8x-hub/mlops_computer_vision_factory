# shared_libs/ml_core/dataloader/dataloader_factory.py

import logging
from typing import Dict, Any, Union
from torch.utils.data import DataLoader, DistributedSampler
from shared_libs.ml_core.dataset.cv_dataset import CVDataset
from shared_libs.ml_core.trainer.utils import distributed_utils # Tiện ích DDP

logger = logging.getLogger(__name__)

class DataLoaderFactory:
    """
    Factory class chịu trách nhiệm tạo ra các DataLoader instances (Train/Validation/Production).

    Lớp này đóng gói tất cả logic liên quan đến sampling, batching, và num_workers,
    tách biệt khỏi CVDataset và các Orchestrator.
    """

    @staticmethod
    def create(dataset: CVDataset, config: Dict[str, Any], context: str) -> DataLoader:
        """
        Tạo và trả về một DataLoader instance dựa trên ngữ cảnh (training/evaluation/production).

        Args:
            dataset (CVDataset): Dataset instance đã được khởi tạo.
            config (Dict[str, Any]): Configuration dict chứa các tham số batching/worker.
            context (str): Ngữ cảnh sử dụng ("training", "evaluation", "production").

        Returns:
            DataLoader: DataLoader instance đã cấu hình.
        """
        is_distributed = distributed_utils.get_world_size() > 1
        
        # 1. Lấy tham số chung (Giả định nằm trong config['trainer'] hoặc config['evaluator'])
        if context == "training":
             params = config.get('trainer', {})
             shuffle = True
        elif context in ["evaluation", "production"]:
             # Sử dụng config chung cho evaluator/production nếu không có config chuyên biệt
             params = config.get('evaluator', {}) 
             shuffle = False
        else:
             raise ValueError(f"Unsupported DataLoader context: {context}")

        batch_size = params.get('batch_size', 32)
        num_workers = params.get('num_workers', 4)
        
        # 2. Xử lý Sampling (CRITICAL for Production-Grade ML)
        sampler = None
        if is_distributed and (context == "training"):
             # DDP Training: Dùng DistributedSampler với shuffle=True
             sampler = DistributedSampler(dataset, shuffle=True)
             shuffle = False # Bắt buộc phải tắt shuffle nếu dùng sampler

        elif is_distributed and (context in ["evaluation", "production"]):
             # DDP Evaluation: Dùng DistributedSampler với shuffle=False
             sampler = DistributedSampler(dataset, shuffle=False)
             shuffle = False

        # 3. Tạo DataLoader
        logger.info(f"Creating DataLoader for context '{context}'. DDP Active: {is_distributed}")
        
        try:
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler, 
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True # Tăng tốc độ truyền dữ liệu CPU <-> GPU
            )
            return data_loader
        except Exception as e:
            logger.error(f"Failed to create DataLoader for context {context}: {e}")
            raise RuntimeError(f"DataLoader creation failed: {e}")
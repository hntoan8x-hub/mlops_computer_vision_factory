# shared_libs/data_labeling/auto_annotation/embedding_proposal.py

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from torch import Tensor
import os

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
from ...configs.label_schema import EmbeddingLabel

logger = logging.getLogger(__name__)

# Định nghĩa kiểu cho Feature Vector (NumPy array hoặc List[float])
FeatureVector = Union[np.ndarray, List[float]]

class EmbeddingProposalAnnotator(BaseAutoAnnotator):
    """
    Annotator chuyên biệt cho Embedding Learning (ví dụ: Face Recognition, Image Retrieval).
    Sinh vector đặc trưng (vector) hoặc Target ID (ID) để tạo nhãn.
    """
    
    # Kích thước vector đặc trưng
    EMBEDDING_DIM = 512 

    def _run_inference(self, image_data: np.ndarray) -> Tuple[FeatureVector, str]:
        """
        Mô phỏng chạy mô hình Embedding (ví dụ: ResNet, CLIP) và trả về: 
        (vector đặc trưng, ID thực thể/lớp gần nhất)
        
        Args:
            image_data (np.ndarray): Ảnh đầu vào (H, W, C).

        Returns:
            Tuple[FeatureVector, str]: (Vector đặc trưng 512D, ID thực thể gần nhất).
        """
        # 1. Giả lập vector đặc trưng (Feature Vector)
        # Vector này thường là đầu ra của lớp cuối cùng (trước lớp phân loại)
        feature_vector = np.random.rand(self.EMBEDDING_DIM).astype(np.float32)
        
        # 2. Giả lập tìm kiếm ID gần nhất (ví dụ: từ một cơ sở dữ liệu khuôn mặt)
        # ID này có thể được dùng làm nhãn target_id
        if np.mean(image_data) > 120:
            target_id = "person_A_v1"
        else:
            target_id = "unknown"
            
        return feature_vector.tolist(), target_id

    def _normalize_output(self, raw_prediction: Tuple[FeatureVector, str], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Chuẩn hóa kết quả dự đoán (vector/ID) thành đối tượng EmbeddingLabel (Pydantic).
        """
        feature_vector, target_id = raw_prediction
        image_path: str = metadata.get("image_path", "unknown")
        
        # NOTE: Trong các tác vụ Retrieval/Triplet, target_id là nhãn quan trọng nhất
        # Vector có thể được trả về nếu cần lưu trữ vector đặc trưng ban đầu.
        
        # 1. Tạo đối tượng EmbeddingLabel
        try:
            label_obj = EmbeddingLabel(
                image_path=image_path,
                target_id=target_id,
                # Lưu vector dự đoán vào trường vector (optional)
                vector=feature_vector 
            )
            
            # 2. Trả về dưới dạng List[StandardLabel]
            return [label_obj]
            
        except Exception as e:
            logger.error(f"Failed to create valid EmbeddingLabel for {image_path}: {e}")
            return []
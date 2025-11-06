# shared_libs/data_labeling/auto_annotation/segmentation_proposal.py

import logging
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from torch import Tensor
from PIL import Image

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
from ...configs.label_schema import SegmentationLabel

logger = logging.getLogger(__name__)

class SegmentationProposalAnnotator(BaseAutoAnnotator):
    """
    Annotator chuyên biệt cho Image Segmentation: Tạo các Mask nhãn (pixel-wise).
    """
    
    def _run_inference(self, image_data: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """
        Mô phỏng chạy mô hình Segmentation (ví dụ: SAM) và trả về danh sách: 
        [(mask_array, class_name, confidence), ...]
        
        Args:
            image_data (np.ndarray): Ảnh đầu vào (H, W, C).

        Returns:
            List[Tuple[np.ndarray, str, float]]: Danh sách các mask dự đoán (binary mask), tên lớp, và độ tin cậy.
        """
        H, W, _ = image_data.shape
        
        # 1. Giả lập Mask (ví dụ: một vùng vuông ở trung tâm)
        mask1 = np.zeros((H, W), dtype=bool)
        mask1[int(H*0.2):int(H*0.8), int(W*0.2):int(W*0.8)] = True
        
        # 2. Giả lập một Mask khác (ví dụ: góc trên bên phải)
        mask2 = np.zeros((H, W), dtype=bool)
        mask2[:int(H*0.4), int(W*0.6):] = True
        
        # Giả lập kết quả dự đoán
        predictions = [
            (mask1, "main_object", 0.98),
            (mask2, "background_area", 0.70), # Vẫn trên ngưỡng
        ]
        return predictions

    def _normalize_output(self, raw_prediction: List[Tuple[np.ndarray, str, float]], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Chuẩn hóa binary mask (NumPy array) thành cấu trúc SegmentationLabel (đường dẫn mask).
        
        Trong thực tế, chúng ta sẽ lưu mask array vào một file tạm thời (ví dụ: PNG) 
        để Dataset có thể đọc lại sau.
        """
        suggested_labels: List[SegmentationLabel] = []
        image_path: str = metadata.get("image_path", "unknown")
        
        # Giả định: Nơi lưu trữ mask tạm thời (nên là một temp directory)
        temp_mask_dir = self.config.get("temp_mask_dir", "/tmp/cvf_masks")
        os.makedirs(temp_mask_dir, exist_ok=True)
        
        for idx, (mask_array, class_name, confidence) in enumerate(raw_prediction):
            if confidence >= self.min_confidence:
                
                # 1. Chuẩn hóa Mask: Chuyển binary mask sang 8-bit grayscale
                mask_8bit = (mask_array * 255).astype(np.uint8)
                
                # 2. Lưu Mask vào file tạm thời
                # Tên file: hash_ảnh_index_lớp.png
                file_hash = hash(image_path) % 100000 
                mask_filename = f"{file_hash}_{idx}_{class_name}.png"
                mask_save_path = os.path.join(temp_mask_dir, mask_filename)
                
                try:
                    Image.fromarray(mask_8bit, 'L').save(mask_save_path)
                    
                    # 3. Tạo đối tượng nhãn đã được kiểm tra (Pydantic)
                    label_obj = SegmentationLabel(
                        image_path=image_path,
                        mask_path=mask_save_path,
                        class_name=class_name # Giả định thêm class_name vào schema SegmentationLabel
                    )
                    suggested_labels.append(label_obj)
                
                except Exception as e:
                    logger.error(f"Failed to save or create SegmentationLabel for {image_path}: {e}")
            
        return suggested_labels
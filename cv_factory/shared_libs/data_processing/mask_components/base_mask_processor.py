# shared_libs/data_processing/mask_components/base_mask_processor.py
import abc
import numpy as np
from typing import Any, Dict, Optional, Union

# Type hint cho Mask Data: có thể là Binary Mask (H, W) hoặc Label Map (H, W)
MaskData = Union[np.ndarray, Dict[str, np.ndarray]]

class BaseMaskProcessor(abc.ABC):
    """
    Abstract Base Class cho tất cả các Mask Processor.

    Định nghĩa giao diện chuẩn để xử lý cặp dữ liệu RGB và Mask/Label Map 
    cho các tác vụ Segmentation/Detection/Tracking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Khởi tạo với cấu hình."""
        self.config = config if config is not None else {}

    @abc.abstractmethod
    def process(self, rgb_image: np.ndarray, mask_data: MaskData) -> Dict[str, Any]:
        """
        Áp dụng quá trình xử lý đồng bộ cho ảnh RGB và Mask/Label Map.

        Args:
            rgb_image (np.ndarray): Ảnh màu RGB (H, W, 3).
            mask_data (MaskData): Dữ liệu Mask/Label Map.

        Returns:
            Dict[str, Any]: Dictionary chứa các kết quả đã được chuyển đổi. 
                            Tối thiểu phải có key 'rgb' và 'mask'.
        
        Raises:
            NotImplementedError: Nếu phương thức chưa được triển khai.
        """
        raise NotImplementedError
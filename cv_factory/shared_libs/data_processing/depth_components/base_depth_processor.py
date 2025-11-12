# shared_libs/data_processing/depth_components/base_depth_processor.py
import abc
import numpy as np
from typing import Any, Dict, Optional, Tuple

class BaseDepthProcessor(abc.ABC):
    """
    Abstract Base Class cho tất cả các Depth Processor.

    Định nghĩa giao diện chuẩn để xử lý cặp dữ liệu RGB và Depth Map, 
    đảm bảo tính nhất quán trong pipeline DepthProcessingOrchestrator.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo với cấu hình.
        Args:
            config (Optional[Dict[str, Any]]): Các tham số cấu hình riêng của component.
        """
        self.config = config if config is not None else {}

    @abc.abstractmethod
    def process(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        """
        Áp dụng quá trình xử lý cho ảnh RGB và Depth Map tương ứng.

        Args:
            rgb_image (np.ndarray): Ảnh màu RGB (H, W, 3).
            depth_map (np.ndarray): Depth Map (H, W) hoặc (H, W, 1).

        Returns:
            Dict[str, Any]: Một dictionary chứa các kết quả đã được chuyển đổi. 
                            Tối thiểu phải có key 'rgb' và 'depth'.
        
        Raises:
            NotImplementedError: Nếu phương thức chưa được triển khai.
        """
        raise NotImplementedError
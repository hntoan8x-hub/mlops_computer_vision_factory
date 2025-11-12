# shared_libs/data_processing/pointcloud_components/base_pointcloud_processor.py
import abc
import numpy as np
from typing import Any, Dict, Optional, Union, List

# Type hint cho Point Cloud Data: Ma trận N x M, N là số điểm, M là số thuộc tính (x, y, z, intensity, color...)
PointcloudData = np.ndarray

class BasePointcloudProcessor(abc.ABC):
    """
    Abstract Base Class cho tất cả các Point Cloud Processor.

    Định nghĩa giao diện chuẩn để xử lý dữ liệu 3D không gian rời rạc (Point Cloud).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Khởi tạo với cấu hình."""
        self.config = config if config is not None else {}

    @abc.abstractmethod
    def process(self, pointcloud: PointcloudData) -> PointcloudData:
        """
        Áp dụng quá trình xử lý cho Point Cloud.

        Args:
            pointcloud (PointcloudData): Dữ liệu Point Cloud đầu vào (N, M) array.

        Returns:
            PointcloudData: Dữ liệu Point Cloud đã được chuyển đổi (N', M') array.
        
        Raises:
            NotImplementedError: Nếu phương thức chưa được triển khai.
        """
        raise NotImplementedError
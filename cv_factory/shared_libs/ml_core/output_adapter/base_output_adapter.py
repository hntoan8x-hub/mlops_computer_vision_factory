# shared_libs/ml_core/output_adapter/base_output_adapter.py

import abc
import numpy as np
import torch
from typing import Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)

# Định nghĩa kiểu đầu vào thô và đầu ra chuẩn hóa
RawModelOutput = Any  # Thường là PyTorch Tensor, Tuple, Dict of Tensors
StandardizedOutput = Union[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]

class BaseOutputAdapter(abc.ABC):
    """
    Abstract Base Class cho các Output Adapter.
    Chịu trách nhiệm chuyển đổi đầu ra thô của mô hình (từ predict())
    sang định dạng chuẩn hóa, sẵn sàng cho Metric.update() hoặc Domain Postprocessor.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        """
        Thực hiện quá trình chuẩn hóa đầu ra mô hình.
        
        Args:
            raw_output (Any): Đầu ra thô của mô hình (Tensor, Tuple, Dict).
            **kwargs: Metadata cần thiết (ví dụ: image_size, class_map).

        Returns:
            StandardizedOutput: Định dạng chuẩn hóa (ví dụ: List[Dict] cho Detection, 
                                NumPy array cho Classification).
        """
        raise NotImplementedError

    def _to_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Helper chung để đảm bảo dữ liệu là NumPy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
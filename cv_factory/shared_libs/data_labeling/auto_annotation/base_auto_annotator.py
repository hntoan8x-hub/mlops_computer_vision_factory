# shared_libs/data_labeling/auto_annotation/base_auto_annotator.py
import numpy as np
import abc
import logging
from typing import List, Dict, Any, Union
from torch import nn, Tensor

# Import các Label Schema tương ứng (để đảm bảo output chuẩn hóa)
from ...configs.label_schema import ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel, EmbeddingLabel

logger = logging.getLogger(__name__)

# Định nghĩa kiểu Output chuẩn hóa cho tầng này
StandardLabel = Union[ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel, EmbeddingLabel]

class BaseAutoAnnotator(abc.ABC):
    """
    Abstract Base Class (ABC) cho tất cả các phương pháp tự động sinh nhãn (Proposal Annotators).
    Định nghĩa giao diện cho việc tải mô hình và sinh nhãn từ ảnh/dữ liệu thô.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo và tải mô hình cần thiết cho việc gán nhãn.
        """
        self.config = config
        self.model: Union[nn.Module, Any] = self._load_model()
        self.min_confidence: float = config.get("min_confidence", 0.7)

    def _load_model(self) -> Union[nn.Module, Any]:
        """
        Phương thức mặc định để tải mô hình. Subclass có thể override.
        Sử dụng config['model_path'] và config['model_type'].
        """
        model_path = self.config.get("model_path")
        model_type = self.config.get("model_type", "default")
        
        if not model_path:
            logger.warning(f"No model path provided for {self.__class__.__name__}. Using dummy/heuristic.")
            return None
            
        logger.info(f"Loading {model_type} model from {model_path}...")
        # NOTE: Logic tải PyTorch/TensorFlow/ONNX model sẽ được thêm ở đây
        
        return object() # Trả về đối tượng mô hình giả lập

    @abc.abstractmethod
    def _run_inference(self, image_data: np.ndarray) -> Any:
        """
        [CHỈNH SỬA] Chạy inference trên mô hình và trả về kết quả thô.
        Mỗi subclass sẽ triển khai logic này tùy theo mô hình (YOLO, SAM, CLIP).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _normalize_output(self, raw_prediction: Any, metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        [CHỈNH SỬA] Chuẩn hóa kết quả dự đoán thô thành các đối tượng Label Schema (Pydantic).
        """
        raise NotImplementedError

    def annotate(self, raw_input: Dict[str, Any]) -> List[StandardLabel]:
        """
        Giao diện công khai: Thực hiện toàn bộ quy trình sinh nhãn tự động.
        """
        image_data = raw_input.get("image_data")
        
        if image_data is None:
            raise ValueError(f"Image data is required for annotation.")

        # 1. Chạy Inference
        raw_prediction = self._run_inference(image_data)
        
        # 2. Chuẩn hóa và Validate
        # Sử dụng toàn bộ raw_input (bao gồm image_path) làm metadata
        return self._normalize_output(raw_prediction, raw_input)
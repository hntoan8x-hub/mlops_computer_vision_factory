# shared_libs/data_labeling/manual_annotation/base_manual_annotator.py

import abc
from typing import List, Dict, Any, Union
from ....data_labeling.configs.label_schema import ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel

# Sử dụng StandardLabel đã định nghĩa ở Auto Annotation
StandardLabel = Union[ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel]

class BaseManualAnnotator(abc.ABC):
    """
    Abstract Base Class cho các Annotator xử lý nhãn đã gán thủ công (Parser).
    Định nghĩa hợp đồng chuẩn hóa từ dữ liệu file thô (CSV, JSON, XML) 
    sang định dạng Pydantic Schema chuẩn.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def parse(self, raw_input: Any) -> List[StandardLabel]:
        """
        Thực hiện việc phân tích dữ liệu thô (từ file) và chuẩn hóa nó.

        Args:
            raw_input (Any): Dữ liệu nhãn thô (ví dụ: DataFrame từ CSV, Dict từ JSON).

        Returns:
            List[StandardLabel]: Danh sách các nhãn đã được chuẩn hóa (Pydantic objects).
        """
        raise NotImplementedError
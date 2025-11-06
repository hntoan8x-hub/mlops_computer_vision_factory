# shared_libs/data_labeling/semi_annotation/base_semi_annotator.py

import abc
from typing import List, Dict, Any, Union
from ....data_labeling.configs.label_schema import StandardLabel

class BaseSemiAnnotator(abc.ABC):
    """
    Abstract Base Class cho các phương pháp Annotation bán tự động (Semi-Annotation/HITL).
    Định nghĩa hợp đồng chuẩn hóa việc tinh chỉnh hoặc lựa chọn nhãn.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def refine(self, 
               proposals: List[StandardLabel], 
               user_feedback: Union[Dict[str, Any], None] = None
    ) -> List[StandardLabel]:
        """
        Thực hiện tinh chỉnh nhãn đề xuất (proposals) dựa trên logic hoặc phản hồi.
        
        Args:
            proposals (List[StandardLabel]): Danh sách nhãn đề xuất ban đầu (từ Auto Annotator).
            user_feedback (Dict | None): Dữ liệu sửa đổi/chấp nhận từ người dùng.

        Returns:
            List[StandardLabel]: Danh sách nhãn đã được tinh chỉnh/chấp nhận.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_samples(self, pool_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Lựa chọn các mẫu dữ liệu cần được gán nhãn/kiểm tra tiếp theo (Active Learning).
        
        Args:
            pool_metadata (List[Dict]): Toàn bộ metadata của dữ liệu chưa nhãn.

        Returns:
            List[Dict]: Danh sách metadata các mẫu được chọn để gửi đi gán nhãn.
        """
        raise NotImplementedError
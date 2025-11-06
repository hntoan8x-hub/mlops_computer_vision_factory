# shared_libs/data_labeling/manual_annotation/segmentation_parser.py

import logging
from typing import List, Dict, Any, Union
import pandas as pd
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import SegmentationLabel
from ....data_labeling.configs.labeler_config_schema import SegmentationLabelerConfig

logger = logging.getLogger(__name__)

class SegmentationParser(BaseManualAnnotator):
    """
    Parser chuyên biệt cho Segmentation: Xử lý file danh sách (CSV/JSON) 
    chứa đường dẫn ảnh và đường dẫn mask tương ứng.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Sử dụng config thô để tạo đối tượng config đã được validate (cho params)
        self.parser_config = SegmentationLabelerConfig(**config) 

    def parse(self, raw_input: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[StandardLabel]:
        """
        Phân tích dữ liệu thô (DataFrame/List of Dicts) và tạo SegmentationLabel objects.
        """
        if isinstance(raw_input, pd.DataFrame):
            df = raw_input
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        else:
            raise TypeError("SegmentationParser expects a Pandas DataFrame or List[Dict] input.")
            
        annotated_labels: List[SegmentationLabel] = []
        
        # Giả định tên cột (cần được lấy từ SegmentationLabelerConfig khi bạn định nghĩa nó)
        img_col = self.parser_config.image_path_column # Giả định tồn tại
        mask_col = self.parser_config.mask_path_column # Giả định tồn tại
        
        for _, row in df.iterrows():
            sample_data = {
                "image_path": row[img_col],
                "mask_path": row[mask_col]
            }
            try:
                # Sử dụng Pydantic Schema để kiểm tra cấu trúc
                validated_label = SegmentationLabel(**sample_data)
                annotated_labels.append(validated_label)
            except Exception as e:
                logger.warning(f"Skipping invalid segmentation entry: {e}")
                
        return annotated_labels
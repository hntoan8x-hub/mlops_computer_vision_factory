# shared_libs/data_labeling/configs/labeler_config_schema.py
from pydantic import BaseModel, Field, conint, confloat, constr
from typing import Dict, Any, List, Optional, Literal

# --- 1. Sub-Schemas: Chi tiết cấu hình cho từng loại Labeler ---

class ClassificationLabelerConfig(BaseModel):
    """Cấu hình cho ClassificationLabeler."""
    # Nguồn dữ liệu chứa nhãn (CSV/JSON/DB Query)
    label_source_uri: str = Field(..., description="URI/đường dẫn của file nhãn (e.g., labels.csv).")
    # Tên cột chứa đường dẫn ảnh trong file nhãn
    image_path_column: str = Field("image_path", description="Tên cột chứa đường dẫn ảnh.")
    # Tên cột chứa nhãn thô
    label_column: str = Field("label", description="Tên cột chứa nhãn thô (string hoặc int).")
    # Ánh xạ từ string label sang integer ID (nếu cần)
    class_map_path: Optional[str] = Field(None, description="Đường dẫn đến file JSON ánh xạ label_name -> id.")

class DetectionLabelerConfig(BaseModel):
    """Cấu hình cho DetectionLabeler."""
    label_source_uri: str = Field(..., description="URI của file nhãn (thường là COCO JSON hoặc danh sách file VOC XML).")
    # Định dạng dữ liệu nhãn đầu vào
    input_format: Literal["coco_json", "voc_xml", "yolo_txt"] = Field("coco_json", description="Định dạng file nhãn đầu vào.")
    # Chuẩn hóa BBox (ví dụ: chia cho width/height)
    normalize_bbox: bool = Field(True, description="Chuẩn hóa BBox về [0, 1].")

class OCRLabelerConfig(BaseModel):
    """Cấu hình cho OCRLabeler."""
    label_source_uri: str = Field(..., description="URI của file nhãn OCR (thường là file JSON chứa text và bbox).")
    # Config cho Tokenizer
    tokenizer_config: Dict[str, Any] = Field({}, description="Cấu hình cho Pre-trained Tokenizer (e.g., vocab path).")
    # Độ dài sequence tối đa
    max_sequence_length: conint(gt=0) = Field(128, description="Độ dài chuỗi token tối đa, dùng cho padding.")
    # Ký tự padding
    padding_token: str = Field("<pad>", description="Token được sử dụng để padding.")

# --- 2. Main Config Schema: Cấu trúc chung cho Labeling Task ---

class LabelerConfig(BaseModel):
    """
    Schema cấu hình chung cho một Labeler trong CV_Factory.
    Sử dụng trường 'params' để chứa cấu hình cụ thể cho từng loại task.
    """
    
    # Loại tác vụ (dùng cho LabelingFactory)
    task_type: Literal["classification", "detection", "segmentation", "ocr", "embedding"] = Field(
        ..., description="Loại tác vụ CV (VD: classification, detection)."
    )
    
    # Config cụ thể cho Labeler (sẽ được Factory resolve)
    # Tùy thuộc vào task_type, trường này sẽ chứa một trong các Sub-Schemas ở trên.
    params: Union[
        ClassificationLabelerConfig,
        DetectionLabelerConfig,
        OCRLabelerConfig,
        # Thêm các config khác ở đây
        Dict[str, Any] 
    ] = Field(..., description="Cấu hình chi tiết (ví dụ: đường dẫn file, column names).")
    
    # Config chung cho việc kiểm tra tính hợp lệ
    validation_ratio: confloat(ge=0, le=1) = Field(0.01, description="Tỷ lệ mẫu nhãn được kiểm tra ngẫu nhiên (validation).")
    
    # Cho phép Cache dữ liệu nhãn đã tải và chuẩn hóa
    cache_path: Optional[str] = Field(None, description="Đường dẫn để cache nhãn đã tải (ví dụ: dưới dạng Parquet).")

# --- 3. Wrapper cho danh sách các Labeler (nếu có nhiều task) ---

class LabelingProjectConfig(BaseModel):
    """Cấu hình tổng quan cho toàn bộ dự án Labeling."""
    labeling_pipelines: List[LabelerConfig] = Field(
        ..., description="Danh sách các pipeline gán nhãn, mỗi pipeline cho một task type."
    )
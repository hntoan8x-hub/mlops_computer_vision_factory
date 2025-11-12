import logging
from pydantic import Field, validator, NonNegativeInt, BaseModel, constr, conint
from typing import List, Dict, Any, Optional, Literal, Union

logger = logging.getLogger(__name__)

# --- Base Schema (Giả định được kế thừa từ một BaseConfig chung) ---
class BaseConfig(BaseModel):
    class Config:
        extra = "forbid" 
    enabled: bool = Field(True, description="Flag để bật/tắt component này.")
    params: Optional[Dict[str, Any]] = Field(None, description="Các tham số cụ thể của component.")

# --- 1. Atomic Component Step Schema (Định nghĩa riêng cho Text) ---

class TextComponentStepConfig(BaseConfig):
    """Schema cho một bước xử lý Text (Loader, Tokenizer, Augmenter, Validator)."""
    type: constr(to_lower=True) = Field(..., description="Tên/loại component Text (ví dụ: 'text_loader', 'text_tokenizer').")
    
    @validator('type')
    def validate_known_text_component(cls, v):
        """Xác thực rằng loại component Text được hỗ trợ."""
        supported_text_types = [
            'text_loader', 'text_tokenizer', 'text_augmenter', 'text_validator'
        ]
        if v not in supported_text_types:
            raise ValueError(f"Unknown Text component type: {v}. Must be one of: {', '.join(supported_text_types)}")
        return v

    @validator('params')
    def validate_tokenizer_params(cls, v: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Thi hành các quy tắc consistency cho Text Tokenizer."""
        if values.get('type') == 'text_tokenizer':
            if not v or 'max_length' not in v or 'vocab_size' not in v:
                raise ValueError("Text Tokenizer requires 'max_length' and 'vocab_size' parameters.")
            
            if v.get('padding_strategy') not in ['pre', 'post']:
                raise ValueError("Text Tokenizer: 'padding_strategy' must be 'pre' or 'post'.")
            
        return v
        
# --- 2. Text Processing Configuration (Schema Chính) ---

class TextProcessingConfig(BaseConfig):
    """
    Schema cho toàn bộ Text Processing pipeline. 
    Quản lý luồng xử lý dữ liệu ngôn ngữ (OCR, VQA, Captioning).
    """
    
    # 1. Policy Mode: 
    policy_mode: Literal["default", "conditional_metadata"] = Field(
        "default",
        description="Policy mode: 'default' sequential execution, hoặc 'conditional_metadata' (adaptive)."
    )
    
    # 2. Các bước xử lý cụ thể
    
    loader: TextComponentStepConfig = Field(
        ...,
        description="Cấu hình BẮT BUỘC cho TextLoader (tải text thô). Loại: 'text_loader'."
    )
    
    augmenter: Optional[TextComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho TextAugmenter (typo, swap, back-translation). Loại: 'text_augmenter'."
    )
    
    tokenizer: TextComponentStepConfig = Field(
        ...,
        description="Cấu hình BẮT BUỘC cho TextTokenizer (chuyển text -> token ID, padding). Loại: 'text_tokenizer'."
    )
    
    validator: Optional[TextComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho TextValidator (kiểm tra độ dài, từ vựng). Loại: 'text_validator'."
    )

    # 3. Quy tắc Governance
    
    @validator('loader', 'tokenizer', 'augmenter', 'validator', always=True)
    def validate_text_component_types(cls, v: Optional[TextComponentStepConfig], values: Dict[str, Any], field: Any) -> Optional[TextComponentStepConfig]:
        """Rule: Đảm bảo các component Text có type đúng."""
        if v is None:
            # Cho phép Augmenter và Validator là None
            return v
        
        expected_type = f"text_{field.name}"
        if v.type != expected_type:
            raise ValueError(
                f"TextProcessingConfig field '{field.name}' requires type '{expected_type}', but got '{v.type}'. (Ensure it is 'text_loader', 'text_tokenizer', etc.)"
            )
        return v
    
    @validator('augmenter', always=True)
    def validate_augmenter_placement(cls, v: Optional[TextComponentStepConfig], values: Dict[str, Any]) -> Optional[TextComponentStepConfig]:
        """Rule: Augmenter (xử lý string) phải chạy TRƯỚC Tokenizer (xử lý token ID)."""
        if v and v.enabled:
             logging.warning("Text Augmenter should be configured to run before Tokenizer for best results (Augmenter processes raw string).")
        return v
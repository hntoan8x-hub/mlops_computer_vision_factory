# shared_libs/ml_core/pipeline_components_cv/configs/schemas/text_schemas.py

from pydantic import BaseModel, Field, PositiveInt
from typing import Literal

# Khai báo các Literal constants cần thiết cho Text
PADDING_STRATEGY = Literal['pre', 'post']
AUG_METHOD = Literal['synonym', 'typo', 'back_translation']

# --- Text Domain Schemas ---
class TextTokenizerParams(BaseModel):
    max_sequence_length: PositiveInt = Field(128)
    vocab_size: PositiveInt = Field(20000)
    padding_strategy: PADDING_STRATEGY = Field('post')
    oov_token: str = Field('<unk>')

class TextAugmenterParams(BaseModel):
    augmentation_method: AUG_METHOD = Field('synonym')
    augmentation_rate: float = Field(0.1, ge=0.0, le=1.0)
    p: float = Field(0.3, ge=0.0, le=1.0)
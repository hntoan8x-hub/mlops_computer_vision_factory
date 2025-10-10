import numpy as np
from typing import Dict, Any, Union, List, Optional
from pydantic import BaseModel, Field, validator

# We define a custom validator for NumPy arrays as Pydantic doesn't natively support them.
class NumpyArray(BaseModel):
    data: List[List[List[Union[int, float]]]]
    shape: List[int]
    dtype: str

    @validator('shape')
    def validate_shape(cls, v, values):
        if len(v) != 3:
            raise ValueError('NumPy array must be 3-dimensional (H, W, C).')
        return v
    
    @validator('data')
    def validate_data_shape(cls, v, values):
        # A simple check to ensure data dimensions match the shape.
        if len(v) != values['shape'][0] or len(v[0]) != values['shape'][1] or len(v[0][0]) != values['shape'][2]:
            raise ValueError('Data dimensions do not match the specified shape.')
        return v

class ProcessedMedicalImage(BaseModel):
    """
    Schema for validating preprocessed image data before it's passed to the model.

    This ensures that the image is in the correct format (e.g., normalized, resized)
    for the model to consume.
    """
    image_id: str
    processed_image: NumpyArray
    metadata: Optional[Dict[str, Any]] = Field({})
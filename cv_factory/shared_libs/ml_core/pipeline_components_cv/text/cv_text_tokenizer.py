# shared_libs/ml_core/pipeline_components_cv/text/cv_text_tokenizer.py (FIXED)

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from shared_libs.ml_core.pipeline_components_cv.base.base_domain_adapter import BaseDomainAdapter

# Import Tokenizer thực tế từ data_processing/
from shared_libs.data_processing.text_components.text_tokenizer import TextTokenizer

logger = logging.getLogger(__name__)

TextData = Union[str, List[str]] # Raw text or list of raw text

class CVTextTokenizer(BaseDomainAdapter):
    """
    Adapter component for Text Tokenization (converting raw text to token IDs).

    This component is STATEFUL (ML) as it implements fit() to learn the vocabulary, 
    and is STATEFUL (Persistence) via delegation.
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter and the atomic TextTokenizer.

        Args:
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        # Instantiate the Adaptee
        processor = TextTokenizer(config=config)
        super().__init__(
            processor=processor, 
            name="CVTextTokenizer", 
            config=config
        )

    def fit(self, X: TextData, y: Optional[Any] = None) -> 'BaseDomainAdapter':
        """
        Fits the TextTokenizer to learn the vocabulary from the training data.
        
        Args:
            X (TextData): Raw text or List[raw text] for vocabulary learning.
            y (Optional[Any]): Target data (ignored).
            
        Returns:
            BaseDomainAdapter: The fitted component instance.
        """
        # Delegation: Call the fit method of the atomic Tokenizer
        self.processor.fit(X) 
        logger.info("CVTextTokenizer has fitted its vocabulary.")
        return self
        
    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: TextData, y: Optional[Any] = None) -> np.ndarray:
        """
        Executes Tokenization and Padding/Truncation.
        
        Args:
            X (TextData): Raw text or List[raw text].
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).
                                
        Returns:
            np.ndarray: Array of token IDs, padded/truncated.
        """
        # Delegation: self.processor.process(data=X)
        return self.processor.process(data=X)
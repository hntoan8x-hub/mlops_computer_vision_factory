# shared_libs/ml_core/pipeline_components_cv/text/cv_text_augmenter.py (FIXED)

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from shared_libs.ml_core.pipeline_components_cv.base.base_domain_adapter import BaseDomainAdapter

# Import Augmenter thực tế từ data_processing/
from shared_libs.data_processing.text_components.text_augmenter import TextAugmenter

logger = logging.getLogger(__name__)

class CVTextAugmenter(BaseDomainAdapter):
    """
    Adapter component for Text Augmentation (Synonyms, Typo, Back-translation).

    This component is Stateless (ML) and delegates execution to the atomic TextAugmenter.
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter and the atomic TextAugmenter.

        Args:
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        # Instantiate the Adaptee
        processor = TextAugmenter(config=config)
        super().__init__(
            processor=processor, 
            name="CVTextAugmenter", 
            config=config
        )

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: Union[str, List[str]], y: Optional[Any] = None) -> Union[str, List[str]]:
        """
        Executes Text Augmentation.
        
        Args:
            X (Union[str, List[str]]): Raw text input.
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).
                                
        Returns:
            Union[str, List[str]]: Augmented raw text.
        """
        # Delegation: self.processor.process(data=X)
        return self.processor.process(data=X)
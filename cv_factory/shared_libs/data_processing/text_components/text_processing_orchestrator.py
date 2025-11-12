# shared_libs/data_processing/text_components/text_processing_orchestrator.py
import logging
import time
from typing import Dict, Any, Optional
import numpy as np
from .text_processor_factory import TextProcessorFactory
from .base_text_processor import RawTextData, TokenizedData

logger = logging.getLogger(__name__)

class TextProcessingOrchestrator:
    """
    Orchestrates the full Text preprocessing pipeline: 
    Load -> Augment -> Tokenize -> Validate.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo Orchestrator bằng cách xây dựng tất cả các thành phần.
        """
        self.config = config
        logger.info("Initializing TextProcessingOrchestrator...")
        
        # Load components
        self.loader = TextProcessorFactory.create("text_loader", config.get("loader", {}))
        self.augmenter = TextProcessorFactory.create("text_augmenter", config.get("augmenter", {}))
        self.tokenizer = TextProcessorFactory.create("text_tokenizer", config.get("tokenizer", {}))
        self.validator = TextProcessorFactory.create("text_validator", config.get("validator", {}))
        
        # Thứ tự thực thi tuần tự (Augment trước Tokenize)
        self._pipeline = [
            ("loader", self.loader), 
            ("augmenter", self.augmenter), 
            ("tokenizer", self.tokenizer),
            ("validator", self.validator) 
        ]
        logger.info("Text processing pipeline constructed successfully.")
        
    def run(self, text_path: str) -> TokenizedData:
        """
        Thực thi toàn bộ pipeline xử lý Text.

        Args:
            text_path (str): Đường dẫn đến file Text hoặc annotation.

        Returns:
            TokenizedData: Dữ liệu đã tokenize và padding (Numpy array of integers).
        """
        start_time = time.time()
        
        # 1. Load (Bước đặc biệt, nhận path, trả về raw text)
        current_data: RawTextData = self.loader.process(text_path)
        logger.debug(f"Loader finished. Data Type: {type(current_data)}")
        
        # 2. Sequential Processing
        for name, component in self._pipeline[1:]: 
            
            component_config = self.config.get(name, {})
            if not component_config.get('enabled', True):
                 continue

            try:
                step_start = time.time()
                # Text Augmenter nhận RawTextData -> trả về RawTextData
                # Text Tokenizer nhận RawTextData -> trả về TokenizedData
                # Text Validator nhận TokenizedData -> trả về TokenizedData
                current_data = component.process(current_data) 
                
                logger.debug(f"Component '{name}' finished in {time.time() - step_start:.4f}s.")
            except Exception as e:
                logger.error(f"Component '{name}' failed with error: {e}")
                raise

        total_time = time.time() - start_time
        logger.info(f"Text processing completed successfully in {total_time:.4f}s.")
        
        # Đảm bảo đầu ra cuối cùng là TokenizedData
        if not isinstance(current_data, np.ndarray):
            logger.error("Final output must be a TokenizedData (Numpy array). Check pipeline configuration.")
            raise TypeError("Final output of Text Processing Orchestrator is not TokenizedData.")
            
        return current_data

    def fit(self, X: Any, y: Optional[Any] = None) -> 'TextProcessingOrchestrator':
        logger.info("Text pipeline fit called (No stateful components implemented).")
        return self

    def save(self, directory_path: str) -> None:
        logger.info("Text pipeline state saved.")
        pass
        
    def load(self, directory_path: str) -> None:
        logger.info("Text pipeline state loaded.")
        pass
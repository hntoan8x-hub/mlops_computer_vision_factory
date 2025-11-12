# cv_factory/shared_libs/ml_core/dataset/cv_dataset.py (FINAL TWEAKED FOR ROBUSTNESS)

import logging
from typing import Dict, Any, List, Optional, Union, Type
import numpy as np
import torch 

# Import Core Utilities
from shared_libs.core_utils.exceptions import DataIntegrityError 

# Import Architectural Components
from .base_cv_dataset import BaseCVDataset, Sample 
from shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator import CVPreprocessingOrchestrator 
from shared_libs.data_ingestion.factories.connector_factory import ConnectorFactory
from shared_libs.data_ingestion.base.base_data_connector import BaseDataConnector
from shared_libs.data_labeling.labeling_factory import LabelingFactory
from shared_libs.data_labeling.base_labeler import BaseLabeler
from shared_libs.ml_core.pipeline_components_cv.factories.component_factory import ComponentFactory 

logger = logging.getLogger(__name__)

class CVDataset(BaseCVDataset):
    """
    Concrete implementation of the BaseCVDataset.
    Orchestrates I/O, Labeling, and Preprocessing.
    
    HARDENED: Accepts live_data_uri_override to support dynamic Production data source.
    """

    def __init__(
        self, 
        dataset_id: str, 
        config: Dict[str, Any], 
        data_connector_config: Dict[str, Any],
        labeling_config: Dict[str, Any],
        preprocessing_config: Dict[str, Any],
        ml_component_factory: Type[ComponentFactory], 
        context: str = 'training',
        live_data_uri_override: Optional[str] = None # <<< THAM SỐ MỚI >>>
    ):
        
        # 1. Initialize Labeler (Giữ nguyên)
        labeler_type = labeling_config.get('task_type', 'classification')
        self.labeler: BaseLabeler = LabelingFactory.get_labeler(
            connector_id=f"{dataset_id}-{labeler_type}-labeler",
            raw_config=labeling_config
        )
        self.annotation_mode = self.labeler.annotation_mode
        self.live_data_uri_override = live_data_uri_override # Lưu trữ URI Override

        # 2. Initialize Data Connector 
        connector_type = data_connector_config.get('type', 'image')
        
        # --- LOGIC GHI ĐÈ URI TRƯỚC KHI KHỞI TẠO CONNECTOR ---
        final_connector_config = data_connector_config.copy()
        if self.live_data_uri_override:
             # Giả định URI cần ghi đè nằm ở khóa 'uri' trong config
             final_connector_config['uri'] = self.live_data_uri_override 
             logger.warning(f"Dataset '{dataset_id}' Data Connector URI overridden to: {self.live_data_uri_override}")
        # --- END LOGIC GHI ĐÈ ---

        data_connector = ConnectorFactory.get_connector(
            connector_type=connector_type,
            connector_config=final_connector_config, # Dùng config đã ghi đè
            connector_id=f"{dataset_id}-data-connector"
        )
        super().__init__(dataset_id, config, data_connector)

        # 3. Initialize Preprocessing Orchestrator (Giữ nguyên)
        self.preprocessor = CVPreprocessingOrchestrator(
            config=preprocessing_config,
            context=context,
            ml_component_factory=ml_component_factory 
        )
        self.context = context
        self.data_connector: BaseDataConnector = data_connector
        
        # 4. Prepare (load metadata)
        self.prepare()

    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Sử dụng Labeler để tải metadata.
        """
        return self.labeler.load_labels() 


    def __len__(self) -> int:
        """Returns the total number of samples."""
        if not self._is_prepared or self.metadata is None:
            return 0
        return len(self.metadata)


    def __getitem__(self, index: int) -> Sample:
        """
        Retrieves a single processed data sample. Orchestrates I/O, Annotation, and Preprocessing.
        """
        if not self._is_prepared or self.metadata is None:
            raise RuntimeError("Dataset is not prepared.")
            
        metadata_entry = self.metadata[index]
        source_uri = metadata_entry.get('image_path', metadata_entry.get('source_uri'))
        
        if not source_uri:
            raise DataIntegrityError(f"Metadata entry at index {index} is missing source URI.")

        try:
            # 1. I/O: Fetch the raw data (Input)
            # HARDENING: Sử dụng Context Manager (connector)
            with self.data_connector as connector: 
                 raw_image_data = connector.read(source_uri=source_uri) 

            # 2. Xử lý Annotation/Labeling (Logic giữ nguyên)
            
            annotator_input = {
                "image_path": source_uri,
                "image_data": raw_image_data,
                "metadata": metadata_entry 
            }
            
            if self.annotation_mode == "manual":
                final_labels = metadata_entry 
                
            elif self.annotation_mode in ["auto", "semi"]:
                
                if not self.labeler.auto_annotator:
                    raise RuntimeError("Auto mode requires an initialized Auto Annotator.")
                    
                proposals = self.labeler.auto_annotator.annotate(annotator_input)
                
                if self.annotation_mode == "semi":
                    if not self.labeler.semi_annotator:
                         logger.warning("Semi mode activated but no Semi Annotator initialized. Skipping refinement.")
                         refined_labels = proposals
                    else:
                         refined_labels = self.labeler.semi_annotator.refine(proposals, user_feedback=None)
                         
                    final_labels = refined_labels[0].model_dump() if refined_labels else {}

                else: 
                    final_labels = proposals[0].model_dump() if proposals else {} 

                metadata_entry.update(final_labels)
                final_labels = metadata_entry
            
            # 3. Processing: Áp dụng Preprocessing Pipeline lên ảnh thô
            # DELEGATION: Ủy quyền cho CVPreprocessingOrchestrator (Engine Façade)
            processed_input = self.preprocessor.run(raw_image_data)
            
            # 4. LABEL CONVERSION: Chuyển nhãn cuối cùng thành PyTorch Tensor
            processed_target = self.labeler.convert_to_tensor(metadata_entry) 
            
            # 5. Standardization: Create the final sample dictionary
            # NOTE: Giả định Sample là Dict[str, Any]
            sample = {
                'input': processed_input, 
                'target': processed_target, 
                'metadata': metadata_entry, 
                'index': index
            }
            return sample
            
        except DataIntegrityError:
            raise
        except Exception as e:
            logger.error(f"Failed to process sample at index {index} (URI: {source_uri}) in mode {self.annotation_mode}: {e}", exc_info=True)
            # HARDENING: Wrap lỗi I/O/Processing thành RuntimeError
            raise RuntimeError(f"Data loading failed for sample {index}: {e}")
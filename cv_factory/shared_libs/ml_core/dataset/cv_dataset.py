# cv_factory/shared_libs/ml_core/data/cv_dataset.py (UPDATED)

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch 

# Import Core Utilities
from shared_libs.core_utils.exceptions import DataIntegrityError 

# Import Architectural Components
from .base_cv_dataset import BaseCVDataset, Sample
from shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator import CVPreprocessingOrchestrator 
from shared_libs.data_ingestion.factories.connector_factory import ConnectorFactory
from shared_libs.data_ingestion.base.base_data_connector import BaseDataConnector
from shared_libs.data_labeling.factories.labeling_factory import LabelingFactory
from shared_libs.data_labeling.base_labeler import BaseLabeler

logger = logging.getLogger(__name__)

class CVDataset(BaseCVDataset):
    """
    Concrete implementation of the BaseCVDataset.
    Orchestrates I/O, Labeling (Manual/Auto/Semi), and Preprocessing.
    """

    def __init__(
        self, 
        dataset_id: str, 
        config: Dict[str, Any], 
        data_connector_config: Dict[str, Any],
        labeling_config: Dict[str, Any],
        preprocessing_config: Dict[str, Any],
        context: str = 'training'
    ):
        
        # 1. Initialize Labeler (Phải khởi tạo trước để lấy mode)
        labeler_type = labeling_config.get('task_type', 'classification')
        self.labeler: BaseLabeler = LabelingFactory.get_labeler(
            connector_id=f"{dataset_id}-{labeler_type}-labeler",
            raw_config=labeling_config
        )
        self.annotation_mode = self.labeler.annotation_mode # Lấy mode từ Labeler

        # 2. Initialize Data Connector (I/O cho cả Ảnh và Metadata)
        connector_type = data_connector_config.get('type', 'image')
        data_connector = ConnectorFactory.get_connector(
            connector_type=connector_type,
            connector_config=data_connector_config,
            connector_id=f"{dataset_id}-data-connector"
        )
        super().__init__(dataset_id, config, data_connector)

        # 3. Initialize Preprocessing Orchestrator (Dependency)
        self.preprocessor = CVPreprocessingOrchestrator(
            config=preprocessing_config,
            context=context
        )
        self.context = context
        self.data_connector: BaseDataConnector = data_connector
        
        # 4. Prepare (load metadata)
        self.prepare()


    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Sử dụng Labeler để tải metadata.
        Trong mode 'manual', metadata chứa cả nhãn.
        Trong mode 'auto/semi', metadata chỉ là danh sách ảnh cần xử lý.
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
            with self.data_connector as connector: 
                 raw_image_data = connector.read(source_uri=source_uri) 

            # 2. Xử lý Annotation/Labeling
            
            # Cấu trúc chung cho dữ liệu thô gửi đến Annotator
            annotator_input = {
                "image_path": source_uri,
                "image_data": raw_image_data,
                "metadata": metadata_entry 
            }
            
            if self.annotation_mode == "manual":
                # CHẾ ĐỘ MANUAL: Nhãn đã có trong metadata_entry, không cần sinh lại
                final_labels = metadata_entry 
                
            elif self.annotation_mode in ["auto", "semi"]:
                
                # CHẾ ĐỘ AUTO/SEMI: Phải sinh nhãn đề xuất (Proposals)
                if not self.labeler.auto_annotator:
                    raise RuntimeError("Auto mode requires an initialized Auto Annotator.")
                    
                # a. Sinh Proposals (Auto Annotation)
                proposals = self.labeler.auto_annotator.annotate(annotator_input)
                
                if self.annotation_mode == "semi":
                    # b. Tinh chỉnh Proposals (Semi Annotation/Refinement)
                    if not self.labeler.semi_annotator:
                         logger.warning("Semi mode activated but no Semi Annotator initialized. Skipping refinement.")
                         refined_labels = proposals
                    else:
                         # Giả định: Không có user feedback trong luồng training/loading
                         refined_labels = self.labeler.semi_annotator.refine(proposals, user_feedback=None)
                         
                    final_labels = refined_labels[0].model_dump() if refined_labels else {}

                else: # Chế độ 'auto' thuần túy
                    # Lấy nhãn đầu tiên (giả định 1 ảnh -> 1 đối tượng Label)
                    final_labels = proposals[0].model_dump() if proposals else {} 

                # Cập nhật metadata_entry với nhãn mới sinh
                metadata_entry.update(final_labels)
                final_labels = metadata_entry
            
            # 3. Processing: Áp dụng Preprocessing Pipeline lên ảnh thô
            processed_input = self.preprocessor.run(raw_image_data)
            
            # 4. LABEL CONVERSION: Chuyển nhãn cuối cùng (trong metadata_entry) thành PyTorch Tensor
            processed_target = self.labeler.convert_to_tensor(metadata_entry) 
            
            # 5. Standardization: Create the final sample dictionary
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
            raise RuntimeError(f"Data loading failed for sample {index}: {e}")
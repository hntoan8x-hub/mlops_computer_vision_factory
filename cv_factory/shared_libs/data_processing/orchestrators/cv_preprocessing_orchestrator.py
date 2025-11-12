# cv_factory/shared_libs/data_processing/orchestrators/cv_preprocessing_orchestrator.py

import logging
import numpy as np
import os
from typing import Dict, Any, Union, List, Optional, Type

# --- CORE IMPORTS: Component Orchestration (The Glue Engine) ---
from shared_libs.ml_core.pipeline_components_cv.orchestrator.component_orchestrator import ComponentOrchestrator
from shared_libs.ml_core.pipeline_components_cv.factories.component_factory import ComponentFactory 
# --- END CORE IMPORTS ---

# --- LOWER-LEVEL ORCHESTRATORS (Data Flow Adapters) ---
from .video_processing_orchestrator import VideoProcessingOrchestrator 
from ..depth_components.depth_processing_orchestrator import DepthProcessingOrchestrator 
from ..mask_components.mask_processing_orchestrator import MaskProcessingOrchestrator        # NEW MASK
from ..pointcloud_components.pointcloud_processing_orchestrator import PointcloudProcessingOrchestrator # NEW POINTCLOUD
from ..text_components.text_processing_orchestrator import TextProcessingOrchestrator         # NEW TEXT

# --- CONFIG & UTILS ---
# Sau khi module hóa, ProcessingConfig sẽ import từ nhiều file schema con
from ...configs.preprocessing_config_schema import ProcessingConfig
# Giả định các hàm kiểm tra type đã tồn tại và được định nghĩa trong _utils/data_type_utils.py
from .._utils.data_type_utils import is_video_data, is_depth_paths, is_mask_paths, is_pointcloud_data, is_text_data 

logger = logging.getLogger(__name__)

# Define standardized input/output data types
# Bổ sung các loại đầu vào cho Point Cloud (data array) và Text (string/list)
PreprocessingInput = Union[Any, List[Any], Dict[str, str]] 
PreprocessingOutput = Union[Any, List[Any], Dict[str, Any]]


class CVPreprocessingOrchestrator:
    """
    High-level Façade/Orchestrator cho toàn bộ Computer Vision preprocessing pipeline.

    Nó đóng vai trò là bộ định tuyến (router) cấp cao, chọn Orchestrator/Engine phù hợp 
    dựa trên loại dữ liệu đầu vào (Image Array, Video Path, Depth Paths, v.v.).
    """

    def __init__(self, config: Dict[str, Any], context: str, ml_component_factory: Type[ComponentFactory]):
        
        try:
            # 1. Validation Master Config (sẽ import từ các schema con)
            self.validated_config: ProcessingConfig = ProcessingConfig(**config)
        except Exception as e:
            logger.error(f"Master ProcessingConfig validation failed: {e}")
            raise ValueError(f"Invalid Master Configuration: {e}")
            
        self.context = context.lower()
        self.processor_type = 'engine' 
        
        # 2. Khởi tạo các Adapters (Orchestrators con)
        self.video_processor: Optional[VideoProcessingOrchestrator] = None
        self.depth_processor: Optional[DepthProcessingOrchestrator] = None
        self.mask_processor: Optional[MaskProcessingOrchestrator] = None         # NEW MASK
        self.pointcloud_processor: Optional[PointcloudProcessingOrchestrator] = None # NEW POINTCLOUD
        self.text_processor: Optional[TextProcessingOrchestrator] = None          # NEW TEXT
        
        self.ml_component_factory = ml_component_factory 
        
        self._initialize_sub_orchestrators(config) 
        
    
    def _initialize_sub_orchestrators(self, config: Dict[str, Any]):
        """Initializes the lower-level Orchestrators."""
        try:
            # 1. IMAGE PROCESSING ENGINE (ComponentOrchestrator) - Luồng cốt lõi
            image_pipeline_steps: List[Dict[str, Any]] = self._extract_image_pipeline_steps()

            self.image_pipeline_engine = ComponentOrchestrator(
                config=image_pipeline_steps,
            )
            logger.info("Image Pipeline Engine (ComponentOrchestrator) built.")
                
            # 2. Video Processor 
            video_config = config.get('video_processing') 
            if video_config and video_config.get('enabled', False):
                self.video_processor = VideoProcessingOrchestrator(config=video_config)
            
            # 3. Depth Processor
            depth_config = config.get('depth_processing')
            if depth_config and depth_config.get('enabled', False):
                self.depth_processor = DepthProcessingOrchestrator(config=depth_config)

            # 4. Mask Processor (NEW LOGIC)
            mask_config = config.get('mask_processing')
            if mask_config and mask_config.get('enabled', False):
                self.mask_processor = MaskProcessingOrchestrator(config=mask_config)

            # 5. Point Cloud Processor (NEW LOGIC)
            pointcloud_config = config.get('pointcloud_processing')
            if pointcloud_config and pointcloud_config.get('enabled', False):
                self.pointcloud_processor = PointcloudProcessingOrchestrator(config=pointcloud_config)
                
            # 6. Text Processor (NEW LOGIC)
            text_config = config.get('text_processing')
            if text_config and text_config.get('enabled', False):
                self.text_processor = TextProcessingOrchestrator(config=text_config)

            
            logger.info("CV Preprocessing initialized with 5 specialized flows.")

        except Exception as e:
            logger.error(f"Failed to initialize sub-orchestrators/Engine: {e}")
            raise RuntimeError(f"Preprocessing initialization failed: {e}")

    def _extract_image_pipeline_steps(self) -> List[Dict[str, Any]]:
        """
        Tập hợp tất cả các bước Image/Feature từ Pydantic config thành một List[Dict] tuần tự.
        """
        steps: List[Dict[str, Any]] = []
        
        # 1. Cleaning Steps 
        steps.extend(self.validated_config.cleaning.steps) 

        # 2. Augmentation Steps
        if self.context == 'training' and self.validated_config.augmentation.enabled:
            steps.extend(self.validated_config.augmentation.steps)
            
        # 3. Feature Engineering / Embedding Steps
        steps.extend(self.validated_config.feature_engineering.components)

        return [step.model_dump() for step in steps]


    def run(self, data: PreprocessingInput, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> PreprocessingOutput:
        """
        Executes the entire preprocessing flow, dynamically routing based on input data type/task.
        """
        if data is None:
            return None
        
        # --- 1. DEPTH PIPELINE (RGB/Depth Paths Input) ---
        if self.depth_processor and is_depth_paths(data):
            logger.info("Detected Depth Task Input. Executing Depth Processing Flow.")
            processed_data = self.depth_processor.run(rgb_path=data.get('rgb_path'), depth_path=data.get('depth_path'))
            
            # Chaining: Image Engine xử lý RGB đã căn chỉnh
            final_rgb = self._run_image_pipeline(processed_data['rgb'])
            return {'rgb': final_rgb, 'depth': processed_data['depth']}

        # --- 2. MASK PIPELINE (RGB/Mask Paths Input) ---
        elif self.mask_processor and is_mask_paths(data):
            logger.info("Detected Mask Task Input. Executing Mask Processing Flow.")
            processed_data = self.mask_processor.run(rgb_path=data.get('rgb_path'), mask_path=data.get('mask_path'))
            
            # Chaining: Image Engine xử lý RGB đã căn chỉnh
            final_rgb = self._run_image_pipeline(processed_data['rgb'])
            return {'rgb': final_rgb, 'mask': processed_data['mask']}

        # --- 3. POINT CLOUD PIPELINE (Path/Data Array Input) ---
        elif self.pointcloud_processor and is_pointcloud_data(data):
            logger.info("Detected Point Cloud Input. Executing Point Cloud Processing Flow.")
            # Point Cloud là luồng độc lập, không chaining với Image Engine
            return self.pointcloud_processor.run(data=data)

        # --- 4. TEXT PIPELINE (String/List Input) ---
        elif self.text_processor and is_text_data(data):
            logger.info("Detected Text Input. Executing Text Processing Flow.")
            # Text là luồng độc lập, không chaining với Image Engine
            return self.text_processor.run(data=data)

        # --- 5. VIDEO PIPELINE (4D Array/Path Input) ---
        elif self.video_processor and is_video_data(data): 
            logger.info("Detected Video Input. Executing Video Processing Flow.")
            
            # 1. Video Processing (Frame Sampling/Cleaning)
            list_of_frames = self.video_processor.transform(data, metadata=metadata, **kwargs)
            
            # 2. Image Processing (on each frame) - Nối chuỗi với Image Engine
            final_features = []
            for frame in list_of_frames:
                final_features.append(self._run_image_pipeline(frame))
                
            return final_features 
        
        # --- 6. IMAGE PIPELINE (3D Array Input or Fallback) ---
        else:
            logger.debug("Falling back to standard Image Pipeline.")
            return self._run_image_pipeline(data)


    def _run_image_pipeline(self, data: PreprocessingInput) -> PreprocessingOutput:
        """Helper method to run the standard Image/Feature pipeline on a single image or batch."""
        final_output = self.image_pipeline_engine.transform(data)
        return final_output

    # --- Full Lifecycle Delegation (Fit, Save, Load) ---
    
    def fit(self, X: PreprocessingInput, y: Optional[Any] = None) -> 'CVPreprocessingOrchestrator':
        """Delegates the fitting process to all initialized stateful sub-orchestrators."""
        logger.info("Starting fitting process for all stateful preprocessing components.")
        
        self.image_pipeline_engine.fit(X, y)
            
        if self.video_processor:
            self.video_processor.fit(X, y)
            
        if self.depth_processor:
            self.depth_processor.fit(X, y) 

        if self.mask_processor:
            self.mask_processor.fit(X, y) # NEW FIT

        if self.pointcloud_processor:
            self.pointcloud_processor.fit(X, y) # NEW FIT

        if self.text_processor:
            self.text_processor.fit(X, y) # NEW FIT
            
        logger.info("Preprocessing pipeline fitting completed.")
        return self
        
    def save(self, directory_path: str) -> None:
        """Delegates saving the state of the entire preprocessing pipeline."""
        logger.info(f"Saving full preprocessing pipeline state to {directory_path}...")
        
        self.image_pipeline_engine.save(os.path.join(directory_path, "image_engine"))
            
        if self.video_processor:
            self.video_processor.save(os.path.join(directory_path, "video_processor")) 

        if self.depth_processor:
            self.depth_processor.save(os.path.join(directory_path, "depth_processor"))

        if self.mask_processor:
            self.mask_processor.save(os.path.join(directory_path, "mask_processor")) # NEW SAVE

        if self.pointcloud_processor:
            self.pointcloud_processor.save(os.path.join(directory_path, "pointcloud_processor")) # NEW SAVE

        if self.text_processor:
            self.text_processor.save(os.path.join(directory_path, "text_processor")) # NEW SAVE

        logger.info("Full preprocessing pipeline state saved successfully.")

    def load(self, directory_path: str) -> None:
        """Delegates loading the state of the entire preprocessing pipeline."""
        logger.info(f"Loading full preprocessing pipeline state from {directory_path}...")
        
        self.image_pipeline_engine.load(os.path.join(directory_path, "image_engine"))
            
        if self.video_processor:
            self.video_processor.load(os.path.join(directory_path, "video_processor")) 

        if self.depth_processor:
            self.depth_processor.load(os.path.join(directory_path, "depth_processor")) 

        if self.mask_processor:
            self.mask_processor.load(os.path.join(directory_path, "mask_processor")) # NEW LOAD

        if self.pointcloud_processor:
            self.pointcloud_processor.load(os.path.join(directory_path, "pointcloud_processor")) # NEW LOAD

        if self.text_processor:
            self.text_processor.load(os.path.join(directory_path, "text_processor")) # NEW LOAD

        logger.info("Full preprocessing pipeline state loaded successfully.")
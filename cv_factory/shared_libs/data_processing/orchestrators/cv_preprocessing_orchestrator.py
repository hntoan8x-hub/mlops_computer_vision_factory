# cv_factory/shared_libs/data_processing/orchestrators/cv_preprocessing_orchestrator.py (FINAL INTEGRATED)

import logging
import numpy as np
import os
from typing import Dict, Any, Union, List, Optional, Type

# Import Lower-Level Orchestrators and Master Schema
from ..augmenters.image_augmenter_orchestrator import ImageAugmenterOrchestrator 
from ..cleaners.image_cleaner_orchestrator import ImageCleanerOrchestrator 
from ..embedders.image_embedder_orchestrator import ImageEmbedderOrchestrator 
from ..feature_extractors.image_feature_extractor_orchestrator import ImageFeatureExtractorOrchestrator 
from .video_processing_orchestrator import VideoProcessingOrchestrator 

# CRITICAL: Import the decoupled utility function
from .._utils.data_type_utils import is_video_data 
from ...configs.preprocessing_config_schema import ProcessingConfig, FeatureExtractionConfig 

logger = logging.getLogger(__name__)

# Define standardized input/output data types
PreprocessingInput = Union[Any, List[Any]]
PreprocessingOutput = Union[Any, List[Any]]


class CVPreprocessingOrchestrator:
    """
    High-level Façade/Orchestrator for the entire Computer Vision preprocessing pipeline.

    Handles data flows for both Image (3D) and Video (4D) inputs by coordinating 
    specialized sub-orchestrators.
    """

    def __init__(self, config: Dict[str, Any], context: str):
        """
        Initializes the Orchestrator by validating configuration and building the pipeline.

        Args:
            config (Dict[str, Any]): The master configuration dictionary for the pipeline.
            context (str): The execution context ('training', 'inference', 'evaluation').

        Raises:
            RuntimeError: If initialization of any sub-orchestrator fails.
        """
        try:
            self.validated_config: ProcessingConfig = ProcessingConfig(**config)
        except Exception as e:
            logger.error(f"Master ProcessingConfig validation failed: {e}")
            raise ValueError(f"Invalid Master Configuration: {e}")
            
        self.context = context.lower()
        self.processor_type = 'none' 
        
        # Image Pipeline Sub-Orchestrators
        self.image_cleaner: ImageCleanerOrchestrator
        self.image_augmenter: ImageAugmenterOrchestrator
        self.feature_processor: Optional[Union[ImageEmbedderOrchestrator, ImageFeatureExtractorOrchestrator]] = None
        
        # Video Pipeline Sub-Orchestrator
        self.video_processor: Optional[VideoProcessingOrchestrator] = None

        self._initialize_sub_orchestrators(config) 
        
    # --- Initialization Logic (Remains the same) ---
    
    def _determine_feature_processor(self, feature_config: FeatureExtractionConfig) -> Optional[Type[Union[ImageEmbedderOrchestrator, ImageFeatureExtractorOrchestrator]]]:
        # Logic remains the same (select Embedder or Feature Extractor class)
        embedder_types = ['cnn_embedder', 'vit_embedder']
        has_embedder = any(step.type in embedder_types for step in feature_config.components)
        
        if has_embedder:
            self.processor_type = 'embedding'
            return ImageEmbedderOrchestrator
        elif feature_config.components:
            self.processor_type = 'feature_extraction'
            return ImageFeatureExtractorOrchestrator
        return None

    def _initialize_sub_orchestrators(self, config: Dict[str, Any]):
        """Initializes the lower-level Orchestrators."""
        try:
            # 1. Image Cleaners/Augmenters/Feature Processors
            self.image_cleaner = ImageCleanerOrchestrator(config=self.validated_config.cleaning.dict())
            self.image_augmenter = ImageAugmenterOrchestrator(config=self.validated_config.augmentation.dict())
            
            processor_cls = self._determine_feature_processor(self.validated_config.feature_engineering)
            if processor_cls:
                self.feature_processor = processor_cls(config=self.validated_config.feature_engineering.dict())
                
            # 2. Video Processor (NEW)
            video_config = config.get('video_processing') 
            if video_config and video_config.get('enabled', False):
                 self.video_processor = VideoProcessingOrchestrator(config=video_config)
            
            logger.info(f"CV Preprocessing initialized. Processor: {self.processor_type}. Video Flow Enabled: {self.video_processor is not None}")

        except Exception as e:
            logger.error(f"Failed to initialize sub-orchestrators: {e}")
            raise RuntimeError(f"Preprocessing initialization failed: {e}")

    # --- Core Execution Delegation (Transform/Run) ---
    
    def run(self, data: PreprocessingInput, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> PreprocessingOutput:
        """
        Executes the entire preprocessing flow, dynamically switching between Image and Video pipelines.

        Args:
            data (PreprocessingInput): The raw input data (Image or Video).
            metadata (Optional[Dict[str, Any]]): Metadata (e.g., color format, video FPS).
            **kwargs: Additional arguments passed to augmenters (e.g., 'labels').

        Returns:
            PreprocessingOutput: The final processed data (e.g., clean image or feature vector/list of feature vectors).
        """
        if data is None:
            return None

        # --- A. VIDEO PIPELINE (4D Array Input) ---
        # CRITICAL FIX: Use the imported is_video_data utility function
        if self.video_processor and is_video_data(data): 
            logger.info("Detected Video Input. Executing Video Processing Flow.")
            
            # 1. Video Processing: Cleaners -> Sampler (Output: List of 3D Frames)
            list_of_frames = self.video_processor.transform(data, metadata=metadata)
            
            # 2. Image Processing (on each frame)
            final_features = []
            for frame in list_of_frames:
                # Reuse the EXISTING Image Flow (Cleaner -> Augmenter -> Feature) for each frame
                feature_vector = self._run_image_pipeline(frame, metadata=metadata, **kwargs)
                final_features.append(feature_vector)
                
            return final_features 
        
        # --- B. IMAGE PIPELINE (3D Array Input or fallback) ---
        else:
            return self._run_image_pipeline(data, metadata=metadata, **kwargs)


    def _run_image_pipeline(self, data: PreprocessingInput, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> PreprocessingOutput:
        """Helper method to run the standard Image/Feature pipeline on a single image or batch."""
        
        # 1. Cleaning (Delegate, pass metadata for Adaptive Cleaning)
        cleaned_data = self.image_cleaner.transform(data, metadata=metadata)
        
        # 2. Augmentation - Only applied if context is 'training'
        augmented_data = cleaned_data
        is_aug_enabled = self.validated_config.augmentation.enabled and len(self.validated_config.augmentation.steps) > 0
        
        if self.context == 'training' and is_aug_enabled:
            augmented_data = self.image_augmenter.transform(cleaned_data, **kwargs)
        
        # 3. Feature Extraction / Embedding - Optional step
        final_output = augmented_data
        if self.feature_processor:
            final_output = self.feature_processor.transform(augmented_data, metadata=metadata) 

        return final_output

    # --- Full Lifecycle Delegation (Fit, Save, Load) ---
    
    def fit(self, X: PreprocessingInput, y: Optional[Any] = None) -> 'CVPreprocessingOrchestrator':
        """Delegates the fitting process to all initialized stateful sub-orchestrators."""
        logger.info("Starting fitting process for all stateful preprocessing components.")
        
        self.image_cleaner.fit(X, y)
        self.image_augmenter.fit(X, y)
            
        if self.feature_processor:
            self.feature_processor.fit(X, y)
            
        if self.video_processor:
             self.video_processor.fit(X, y)
            
        logger.info("Preprocessing pipeline fitting completed.")
        return self
        
    def save(self, directory_path: str) -> None:
        """Delegates saving the state of the entire preprocessing pipeline."""
        logger.info(f"Saving full preprocessing pipeline state to {directory_path}...")
        
        self.image_cleaner.save(os.path.join(directory_path, "cleaner"))
        self.image_augmenter.save(os.path.join(directory_path, "augmenter"))
            
        if self.feature_processor:
            self.feature_processor.save(os.path.join(directory_path, "processor"))
            
        if self.video_processor:
            self.video_processor.save(os.path.join(directory_path, "video_processor")) 

        logger.info("Full preprocessing pipeline state saved successfully.")

    def load(self, directory_path: str) -> None:
        """Delegates loading the state of the entire preprocessing pipeline."""
        logger.info(f"Loading full preprocessing pipeline state from {directory_path}...")
        
        self.image_cleaner.load(os.path.join(directory_path, "cleaner"))
        self.image_augmenter.load(os.path.join(directory_path, "augmenter"))
            
        if self.feature_processor:
            self.feature_processor.load(os.path.join(directory_path, "processor"))
            
        if self.video_processor:
            self.video_processor.load(os.path.join(directory_path, "video_processor")) 

        logger.info("Full preprocessing pipeline state loaded successfully.")
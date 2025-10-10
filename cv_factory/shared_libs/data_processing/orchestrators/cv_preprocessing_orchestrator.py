# cv_factory/shared_libs/data_processing/orchestrators/cv_preprocessing_orchestrator.py (FINAL RESTRUCTURED)

import logging
from typing import Dict, Any, Union, List, Optional, Type
import copy

# Import Lower-Level Orchestrators and Master Schema
from ..augmenters.augmenter_orchestrator import AugmenterOrchestrator
from ..cleaners.cleaner_orchestrator import CleanerOrchestrator
from ..embedders.embedder_orchestrator import EmbedderOrchestrator
from ..feature_extractors.feature_extractor_orchestrator import FeatureExtractorOrchestrator
# Import the actual Master Schema from the provided file
from ...configs.preprocessing_config_schema import ProcessingConfig, FeatureExtractionConfig 

logger = logging.getLogger(__name__)

# Define standardized input/output data types
PreprocessingInput = Union[Any, List[Any]]
PreprocessingOutput = Union[Any, List[Any]]

class CVPreprocessingOrchestrator:
    """
    High-level Façade/Orchestrator for the entire Computer Vision preprocessing pipeline.

    It validates the master configuration (ProcessingConfig) and delegates all lifecycle 
    operations (fit, transform/run, save, load) to the specialized sub-orchestrators.
    """

    def __init__(self, config: Dict[str, Any], context: str):
        """
        Initializes the Orchestrator, validating configuration against ProcessingConfig.
        """
        # 1. Configuration Validation: Use the master schema
        # This will fail if 'cleaning', 'augmentation', or 'feature_engineering' keys are missing/invalid.
        self.validated_config: ProcessingConfig = ProcessingConfig(**config)
        self.context = context.lower()
        self.processor_type = 'none' 
        
        self.cleaner: CleanerOrchestrator
        self.augmenter: AugmenterOrchestrator
        self.feature_processor: Optional[Union[EmbedderOrchestrator, FeatureExtractorOrchestrator]] = None
        
        self._initialize_sub_orchestrators()

    def _determine_feature_processor(self, feature_config: FeatureExtractionConfig) -> Optional[Type[Union[EmbedderOrchestrator, FeatureExtractorOrchestrator]]]:
        """
        Analyzes the steps in feature_engineering to determine whether to use Embedder or Feature Extractor.
        
        Returns the appropriate class type or None.
        """
        if not feature_config.enabled:
            return None
            
        # Check if any component is a deep learning embedder
        embedder_types = ['cnn_embedder', 'vit_embedder']
        
        # NOTE: FeatureExtractionConfig has a single 'components' list for all steps
        has_embedder = any(step.type in embedder_types for step in feature_config.components)
        
        if has_embedder:
            self.processor_type = 'embedding'
            return EmbedderOrchestrator
        elif feature_config.components:
            self.processor_type = 'feature_extraction'
            return FeatureExtractorOrchestrator
            
        return None

    def _initialize_sub_orchestrators(self):
        """
        Initializes the lower-level Orchestrators based on the validated configuration.
        """
        try:
            # 1. Cleaner (Uses config['cleaning'])
            self.cleaner = CleanerOrchestrator(
                config=self.validated_config.cleaning.dict()
            )
            
            # 2. Augmenter (Uses config['augmentation'])
            self.augmenter = AugmenterOrchestrator(
                config=self.validated_config.augmentation.dict()
            )
            
            # 3. Feature/Embedding (Uses config['feature_engineering'])
            processor_cls = self._determine_feature_processor(self.validated_config.feature_engineering)
            
            if processor_cls:
                # The FeatureExtractorOrchestrator/EmbedderOrchestrator must be able to parse
                # the generic FeatureExtractionConfig structure they are passed.
                self.feature_processor = processor_cls(
                    config=self.validated_config.feature_engineering.dict()
                )
            
            logger.info(f"Initialized CV Preprocessing Orchestrator in '{self.context}' context. Final Processor: {self.processor_type}")

        except Exception as e:
            logger.error(f"Failed to initialize sub-orchestrators: {e}")
            raise RuntimeError(f"Preprocessing initialization failed: {e}")

    # --- Core Execution Delegation (Transform/Run) ---
    
    def run(self, data: PreprocessingInput, **kwargs) -> PreprocessingOutput:
        """
        Executes the entire preprocessing flow: CLEAN -> [AUGMENT] -> [FEATURE/EMBED].
        
        This method is the primary entry point for CVPredictor/CVDataset.
        """
        if data is None:
            logger.warning("Input data for preprocessing is None. Returning None.")
            return None

        # 1. Cleaning (Delegate to consistent 'transform')
        cleaned_data = self.cleaner.transform(data)
        logger.debug("Step 1: Data Cleaning completed.")

        # 2. Augmentation - Only applied if enabled and context is 'training'
        augmented_data = cleaned_data
        is_aug_enabled = self.validated_config.augmentation.enabled and len(self.validated_config.augmentation.steps) > 0
        
        if self.context == 'training' and is_aug_enabled:
            # Delegate to consistent 'transform', passing through **kwargs (e.g., 'labels')
            augmented_data = self.augmenter.transform(cleaned_data, **kwargs)
            logger.debug("Step 2: Data Augmentation completed.")

        # 3. Feature Extraction / Embedding - Optional step
        final_output = augmented_data
        if self.feature_processor:
            # Delegate to the appropriate 'transform' (which is the public 'extract' or 'embed' method)
            # NOTE: We assume 'transform' is the correct delegation target for the engine
            final_output = self.feature_processor.transform(augmented_data) 
            logger.debug(f"Step 3: {self.processor_type.capitalize()} completed.")

        return final_output

    # --- Full Lifecycle Delegation (Fit, Save, Load) ---
    
    def fit(self, X: PreprocessingInput, y: Optional[Any] = None) -> 'CVPreprocessingOrchestrator':
        """Delegates the fitting process to all initialized stateful sub-orchestrators."""
        logger.info("Starting fitting process for all stateful preprocessing components.")
        
        # Delegation: Cleaner, Augmenter, and Feature Processor must support fit()
        self.cleaner.fit(X, y)
        self.augmenter.fit(X, y)
            
        if self.feature_processor:
            self.feature_processor.fit(X, y)
            
        logger.info("Preprocessing pipeline fitting completed.")
        return self
        
    def save(self, directory_path: str) -> None:
        """Delegates saving the state of the entire preprocessing pipeline."""
        logger.info(f"Saving full preprocessing pipeline state to {directory_path}...")
        
        self.cleaner.save(f"{directory_path}/cleaner")
        self.augmenter.save(f"{directory_path}/augmenter")
            
        if self.feature_processor:
            self.feature_processor.save(f"{directory_path}/processor")

        logger.info("Full preprocessing pipeline state saved successfully.")

    def load(self, directory_path: str) -> None:
        """Delegates loading the state of the entire preprocessing pipeline."""
        logger.info(f"Loading full preprocessing pipeline state from {directory_path}...")
        
        self.cleaner.load(f"{directory_path}/cleaner")
        self.augmenter.load(f"{directory_path}/augmenter")
            
        if self.feature_processor:
            self.feature_processor.load(f"{directory_path}/processor")

        logger.info("Full preprocessing pipeline state loaded successfully.")
# shared_libs/data_processing/orchestrators/video_processing_orchestrator.py

import logging
import os
import pickle
from typing import Dict, Any, Optional, List, Union

# --- Video Abstractions and Contracts ---
from ..video_components._base.base_video_cleaner import BaseVideoCleaner, VideoData
from ..video_components._base.base_frame_sampler import BaseFrameSampler, ImageData 

# --- Video Factories & Samplers (DI) ---
from ..video_components.cleaners.video_cleaner_factory import VideoCleanerFactory 
# NOTE: We only need PolicySampler or FrameSamplerFactory, depending on the config structure.
# Assuming config defines the POLICY, we use the specific PolicySampler component directly.
from ..video_components.samplers.policy_sampler import PolicySampler 
from ..video_components.samplers.frame_sampler_factory import FrameSamplerFactory 

# --- Pydantic Schema ---
from ...configs.preprocessing_config_schema import VideoProcessingConfig 

logger = logging.getLogger(__name__)

# Constants for MLOps lifecycle file paths
CLEANER_DIR = "cleaners"
SAMPLER_DIR = "samplers"

class VideoProcessingOrchestrator:
    """
    Orchestrates the entire video preprocessing flow: Video Cleaning -> Frame Sampling.

    This class acts as a self-contained execution engine, ensuring configuration validation, 
    managing the policy-based frame sampling, and implementing the MLOps lifecycle (save/load).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by validating configuration and building the video pipeline.

        Args:
            config (Dict[str, Any]): A dictionary containing the video cleaning and sampling configuration.

        Raises:
            ValueError: If configuration validation fails.
            RuntimeError: If initialization of components fails.
        """
        try:
            # 1. Configuration Validation (Quality Gate)
            self.config_schema: VideoProcessingConfig = VideoProcessingConfig(**config)
        except Exception as e:
            logger.error(f"VideoProcessingConfig Validation Failed. Error: {e}")
            raise ValueError(f"Invalid Video Configuration provided: {e}")
            
        # 2. Pipeline Construction
        self._cleaners: List[BaseVideoCleaner] = []
        self._sampler: Optional[BaseFrameSampler] = None # Will be a PolicySampler instance
        
        self._build_pipeline()

        logger.info(f"VideoProcessingOrchestrator initialized with {len(self._cleaners)} cleaners and policy: {self._sampler.__class__.__name__ if self._sampler else 'None'}.")


    def _build_pipeline(self):
        """Builds the internal pipeline from the validated configuration schema."""
        
        # 1. Build Video Cleaners (from self.config_schema.cleaners)
        for step in self.config_schema.cleaners:
            try:
                if step.enabled:
                    component = VideoCleanerFactory.create(
                        cleaner_type=step.type,
                        config=step.get('params')
                    )
                    self._cleaners.append(component)
            except Exception as e:
                raise RuntimeError(f"Failed to build Video Cleaner {step.type}: {e}")

        # 2. Build Frame Sampler (Use the first defined sampler step for PolicySampler instantiation)
        sampler_steps = self.config_schema.samplers
        if sampler_steps:
            # HARDENING: Assume the first step in 'samplers' defines the policy and parameters.
            main_sampler_step = sampler_steps[0]
            
            # NOTE: We assume the configuration structure for PolicySampler is defined 
            # within the main sampler step's params, or we use the step.type as policy_type.
            
            # Use the step type as the policy type for the PolicySampler wrapper.
            policy_type = main_sampler_step.type 
            
            # All steps' parameters are aggregated as config_params for the PolicySampler.
            # This allows the PolicySampler to access config for all potential underlying samplers.
            all_sampler_params = [step.params for step in sampler_steps if step.params]
            
            # CRITICAL: We instantiate the PolicySampler, which then selects the concrete sampler internally.
            self._sampler = PolicySampler(
                 policy_type=policy_type,
                 config_params=main_sampler_step.params # Use params of the main sampler step
            )
        
        if not self._sampler:
            logger.warning("No Frame Sampler defined. Video processing cannot bridge to the image pipeline.")


    def transform(self, video_data: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Executes the video processing flow: Video Cleaning -> Policy-based Frame Sampling.

        Args:
            video_data (VideoData): The raw input video data (4D tensor or list of 4D tensors).
            metadata (Optional[Dict[str, Any]]): Metadata used for adaptive cleaning/sampling.
            **kwargs: Additional keyword arguments.

        Returns:
            ImageData: The extracted list of 3D image frames (the bridge output).
        
        Raises:
            RuntimeError: If execution fails or no sampler is available.
        """
        current_video = video_data

        # 1. Video Cleaning (4D -> 4D)
        for cleaner in self._cleaners:
            current_video = cleaner.transform(current_video, metadata=metadata, **kwargs)
        logger.debug("Video Cleaning completed.")

        # 2. Frame Sampling (4D -> List[3D]) - Bridge Step
        if not self._sampler:
            raise RuntimeError("Cannot execute transform: PolicySampler is required to convert VideoData to ImageData.")
        
        # Delegation: PolicySampler selects and runs the appropriate concrete sampler.
        final_frames = self._sampler.sample(current_video, metadata=metadata, **kwargs)
        logger.debug(f"Frame Sampling completed. Extracted {len(final_frames)} frames.")

        return final_frames

    # --- MLOps Lifecycle Methods (HARDENED Save/Load) ---

    def fit(self, X: VideoData, y: Optional[Any] = None) -> 'VideoProcessingOrchestrator':
        """Fits stateful components (if any) by delegating to cleaners and sampler."""
        for cleaner in self._cleaners:
            if hasattr(cleaner, 'fit'):
                cleaner.fit(X, y)
        if self._sampler and hasattr(self._sampler, 'fit'):
            self._sampler.fit(X, y)
            
        logger.info("Video pipeline fitting completed.")
        return self

    def save(self, directory_path: str) -> None:
        """Saves the state of the video processing pipeline by delegating to individual components."""
        os.makedirs(directory_path, exist_ok=True)
        
        # Save cleaners
        cleaner_path = os.path.join(directory_path, CLEANER_DIR)
        os.makedirs(cleaner_path, exist_ok=True)
        for i, cleaner in enumerate(self._cleaners):
            if hasattr(cleaner, 'save'):
                cleaner.save(os.path.join(cleaner_path, f"{cleaner.__class__.__name__}_{i}.pkl"))

        # Save sampler (The PolicySampler should delegate saving of its active component)
        sampler_path = os.path.join(directory_path, SAMPLER_DIR)
        os.makedirs(sampler_path, exist_ok=True)
        if self._sampler and hasattr(self._sampler, 'save'):
            # Save the PolicySampler itself (e.g., policy type and config)
            self._sampler.save(os.path.join(sampler_path, "policy_sampler.pkl")) 
            
        logger.info(f"Video processing pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """
        Loads the state of the video processing pipeline, re-initializing components based on the config.
        """
        # Load Cleaners
        cleaner_path = os.path.join(directory_path, CLEANER_DIR)
        for i, step_config in enumerate(self.config_schema.cleaners):
            if step_config.enabled:
                cleaner = VideoCleanerFactory.create(cleaner_type=step_config.type, config=step_config.params)
                file_path = os.path.join(cleaner_path, f"{cleaner.__class__.__name__}_{i}.pkl")
                if os.path.exists(file_path) and hasattr(cleaner, 'load'):
                    cleaner.load(file_path)
                self._cleaners.append(cleaner)
                
        # Load Sampler
        sampler_path = os.path.join(directory_path, SAMPLER_DIR)
        if self.config_schema.samplers:
            # Re-initialize sampler based on the schema first
            main_sampler_step = self.config_schema.samplers[0]
            self._sampler = PolicySampler(
                 policy_type=main_sampler_step.type,
                 config_params=main_sampler_step.params
            )
            # Load the state of the PolicySampler
            policy_sampler_file = os.path.join(sampler_path, "policy_sampler.pkl")
            if os.path.exists(policy_sampler_file) and hasattr(self._sampler, 'load'):
                 self._sampler.load(policy_sampler_file)
            
        logger.info(f"Video processing pipeline state loaded from {directory_path}.")
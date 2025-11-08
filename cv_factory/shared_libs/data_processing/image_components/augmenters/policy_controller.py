# shared_libs/data_processing/image_components/augmenters/policy_controller.py

import random
import logging
from typing import List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class AugmentPolicyController:
    """
    Policy Controller for selecting augmentation sequence dynamically.

    This controller supports sequential execution, random subset selection, 
    and policy-based selection (like RandAugment), and determines the magnitude (M) 
    to be applied to the selected operations.
    """

    def __init__(self, policy_type: str = "sequential", n_select: Optional[int] = None, magnitude: float = 0.5):
        """
        Initializes the Policy Controller.

        Args:
            policy_type (str): The policy mode ('sequential', 'random_subset', 'randaugment').
            n_select (Optional[int]): Number of augmenters (N) to select in policy modes.
            magnitude (float): The default intensity (M) for RandAugment, typically between 0.0 and 1.0.

        Raises:
            ValueError: If the policy type is unsupported or magnitude is invalid.
        """
        supported_policies = ["sequential", "random_subset", "randaugment"]
        policy_type_lower = policy_type.lower()
        
        if policy_type_lower not in supported_policies:
            raise ValueError(f"Unsupported policy type: {policy_type}. Must be one of {supported_policies}.")
        if not 0.0 <= magnitude <= 1.0:
            logger.warning("Magnitude should typically be between 0.0 and 1.0.")

        self.policy_type = policy_type_lower
        self.n_select = n_select
        self.magnitude = magnitude # Giá trị magnitude cố định từ config
        
        logger.info(f"Initialized AugmentPolicyController. Policy: {self.policy_type}, N: {self.n_select}, M: {self.magnitude}")


    def select_pipeline(self, pipeline: List[Any]) -> Tuple[List[Any], float]:
        """
        Selects a subset of the pipeline based on the active policy type and returns 
        the list of components along with the magnitude value to use.

        Args:
            pipeline (List[Any]): The full list of initialized augmenter components.

        Returns:
            Tuple[List[Any], float]: The selected components and the magnitude (M) value.
        """
        n_total = len(pipeline)
        
        # 1. Determine Selected Pipeline
        if self.policy_type == "sequential":
            selected_pipeline = pipeline
            M = 1.0 # Sequential mode usually runs at full intensity
            
        elif self.policy_type == "random_subset" or self.policy_type == "randaugment":
            # Determine k (the number to select)
            k = self.n_select if self.n_select is not None else 2 
            k = min(k, n_total)
            
            if k == 0:
                 logger.warning("Pipeline is empty or N_select is 0. Returning empty pipeline.")
                 return [], 0.0
            
            selected_pipeline = random.sample(pipeline, k)
            
            # Determine Magnitude (M)
            if self.policy_type == "randaugment":
                # M is fixed by the controller's config (e.g., M=0.5)
                M = self.magnitude 
            else:
                # For random_subset, magnitude often defaults to max intensity
                M = 1.0

        else: # Fallback (should be sequential due to __init__ validation)
            selected_pipeline = pipeline
            M = 1.0

        logger.debug(f"Policy '{self.policy_type}' selected {len(selected_pipeline)} augmenters with Magnitude {M:.2f}.")
        
        return selected_pipeline, M
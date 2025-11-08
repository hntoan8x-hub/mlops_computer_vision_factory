# shared_libs/data_processing/image_components/cleaners/cleaner_policy_controller.py

import logging
from typing import List, Any, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class CleanerPolicyController:
    """
    Policy Controller for selecting and configuring image cleaning steps dynamically.

    Supports conditional execution and adaptive parameter selection based on metadata.
    """

    def __init__(self, policy_type: str = "default"):
        """
        Initializes the Policy Controller.

        Args:
            policy_type (str): The policy mode ('default' or 'conditional_metadata').
        """
        self.policy_type = policy_type.lower()
        
        logger.info(f"Initialized CleanerPolicyController with policy: {self.policy_type}")

    def select_and_configure_pipeline(self, pipeline: List[Any], metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Selects and potentially reconfigures the cleaning steps based on the policy.

        Args:
            pipeline (List[Any]): The full list of initialized image cleaner components.
            metadata (Optional[Dict[str, Any]]): Metadata of the input image (e.g., color_format='BGR', source='S3').

        Returns:
            List[Any]: The final list of configured components to run.
        """
        if self.policy_type == "default":
            # Mode 1: Run all cleaners sequentially as configured (Current behavior)
            return pipeline
            
        elif self.policy_type == "conditional_metadata" and metadata:
            # Mode 2: Conditional execution based on metadata (e.g., skip ColorSpace if already RGB)
            
            final_pipeline = []
            input_color_format = metadata.get('color_format', 'Unknown').upper()

            for component in pipeline:
                component_name = component.__class__.__name__
                
                if component_name == "ColorSpaceCleaner":
                    # Example Rule: If the image is already RGB, skip the BGR2RGB conversion
                    if input_color_format == "RGB" and "BGR2RGB" in component.conversion_code:
                        logger.debug(f"Skipping {component_name} as input is already {input_color_format}.")
                        continue
                
                # Add component if no skip condition is met
                final_pipeline.append(component)

            return final_pipeline

        # Fallback to default sequential execution
        return pipeline
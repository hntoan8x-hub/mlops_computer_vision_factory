# domain_models/medical_imaging/factory/domain_factory.py

import logging
from typing import Dict, Any, Optional, Type
import importlib

# Import các thành phần domain
from ..postprocessor.medical_postprocessor import MedicalPostprocessor
from ..model.medical_entity import FinalDiagnosis

logger = logging.getLogger(__name__)

class MedicalDomainFactory:
    """
    Factory for creating and configuring domain-specific components.
    
    This centralizes the instantiation logic for the Medical domain.
    """

    @staticmethod
    def create_postprocessor(config: Dict[str, Any]) -> MedicalPostprocessor:
        """
        Creates and configures the MedicalPostprocessor instance.
        """
        postprocessor_params = config.get('postprocessor', {}).get('params', {})
        
        # Load any necessary domain-specific data/rules here if needed
        
        return MedicalPostprocessor(**postprocessor_params)

    @staticmethod
    def get_final_entity_schema() -> Type[FinalDiagnosis]:
        """
        Returns the Pydantic schema for the final domain output.
        """
        return FinalDiagnosis
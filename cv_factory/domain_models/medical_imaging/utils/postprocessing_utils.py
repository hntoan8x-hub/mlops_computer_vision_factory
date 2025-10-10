import logging
import json
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

def convert_to_fhir(prediction_output: Union[Dict, Any]) -> Dict[str, Any]:
    """
    Converts a model's prediction output to a FHIR (Fast Healthcare Interoperability Resources)
    Observation format. This is a conceptual example.

    Args:
        prediction_output (Union[Dict, Any]): The prediction result from the model.

    Returns:
        Dict[str, Any]: A FHIR-compliant Observation resource.
    """
    # This is a simplified, conceptual FHIR structure.
    fhir_resource = {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "8661-1",
                "display": "Lung nodule segmentation"
            }]
        },
        "valueCodeableConcept": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "12345",
                "display": prediction_output.get("class_name", "nodule")
            }]
        },
        "extension": [{
            "url": "http://example.com/model-confidence",
            "valueDecimal": prediction_output.get("confidence", 0.0)
        }]
    }
    
    return fhir_resource
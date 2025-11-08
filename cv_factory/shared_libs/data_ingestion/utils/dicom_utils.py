# cv_factory/shared_libs/data_ingestion/utils/dicom_utils.py (Hardened)

import logging
from typing import List, Union

from pydicom import Dataset
from pydicom.tag import Tag

logger = logging.getLogger(__name__)

# List of common DICOM tags to be anonymized (e.g., PatientName, PatientID).
PHI_TAGS_TO_ANONYMIZE = [
    Tag(0x0010, 0x0010),  # PatientName
    Tag(0x0010, 0x0020),  # PatientID
    Tag(0x0008, 0x0020),  # StudyDate
    Tag(0x0010, 0x0030),  # PatientBirthDate
]

def anonymize_dicom_file(dicom_dataset: Dataset, tags_to_anonymize: Union[List[Tag], None] = None) -> Dataset:
    """
    Anonymizes a DICOM dataset by removing or replacing specified PHI tags.

    Args:
        dicom_dataset: The pydicom Dataset object to be anonymized.
        tags_to_anonymize: A list of pydicom Tags to anonymize. Defaults to PHI_TAGS_TO_ANONYMIZE.

    Returns:
        Dataset: The anonymized pydicom Dataset object.
    """
    if tags_to_anonymize is None:
        tags_to_anonymize = PHI_TAGS_TO_ANONYMIZE
    
    # Hardening: Create a copy to ensure the original dataset is not modified,
    # especially critical when sharing datasets across different pipeline steps.
    anonymized_dataset = dicom_dataset.copy()

    for tag in tags_to_anonymize:
        if tag in anonymized_dataset:
            # Set value to None/empty string or use pydicom's deidentify method
            anonymized_dataset[tag].value = "ANONYMIZED_CVF"
            logger.debug(f"Anonymized tag: {anonymized_dataset[tag].keyword} (0x{tag.tag:08x})")
        else:
            logger.debug(f"Tag 0x{tag.tag:08x} not found in dataset. Skipping.")
            
    # Remove Private Tags entirely (extra security layer)
    anonymized_dataset.remove_private_tags()
    
    return anonymized_dataset
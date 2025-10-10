import logging
from typing import List, Union

from pydicom import Dataset
from pydicom.tag import Tag

logger = logging.getLogger(__name__)

# List of common DICOM tags to be anonymized (e.g., PatientName, PatientID).
# For a full list, refer to the DICOM standard.
PHI_TAGS_TO_ANONYMIZE = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0008, 0x0020),  # StudyDate
    (0x0010, 0x0030),  # PatientBirthDate
    # Add other tags as needed for compliance
]

def anonymize_dicom_file(dicom_dataset: Dataset, tags_to_anonymize: Union[List[Tag], None] = None) -> None:
    """
    Anonymizes a DICOM dataset by removing or replacing specified PHI tags.

    Args:
        dicom_dataset (Dataset): The pydicom Dataset object to be anonymized.
        tags_to_anonymize (Union[List[Tag], None]): A list of pydicom Tags to anonymize.
                                                      Defaults to a predefined list.
    """
    if tags_to_anonymize is None:
        tags_to_anonymize = PHI_TAGS_TO_ANONYMIZE

    for tag in tags_to_anonymize:
        if tag in dicom_dataset:
            dicom_dataset.deidentify(recursive=False) # Simple method to remove sensitive tags
            # For more complex anonymization, you might replace values with a placeholder
            # For example: dicom_dataset[tag].value = "ANONYMIZED"
            logger.info(f"Anonymized tag: {dicom_dataset[tag].keyword} (0x{tag:08x})")
        else:
            logger.debug(f"Tag 0x{tag:08x} not found in dataset. Skipping.")
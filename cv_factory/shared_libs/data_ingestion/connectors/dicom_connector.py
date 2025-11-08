# cv_factory/shared_libs/data_ingestion/connectors/dicom_connector.py

import logging
import pydicom # Standard library for DICOM
import numpy as np
import os
from typing import Dict, Any, Union
from ..base.base_data_connector import BaseDataConnector, OutputData
# NOTE: Assuming these utility functions are implemented in the data_ingestion/utils/
from ..utils.dicom_utils import dicom_to_numpy, save_numpy_as_parquet, anonymize_dicom # Assuming anonymize_dicom is added

logger = logging.getLogger(__name__)

class DICOMConnector(BaseDataConnector):
    """
    Concrete connector for handling DICOM files, used primarily in medical imaging.
    
    Supports reading DICOM image data and writing extracted features/metadata 
    in production-safe formats like Parquet.
    """

    def connect(self) -> bool:
        """
        No external persistent connection required for local/file-based DICOM access.

        Returns:
            bool: Always True.
        """
        self.is_connected = True
        logger.info(f"[{self.connector_id}] DICOM Connector initialized.")
        return True

    def read(self, source_uri: str, as_numpy: bool = True, anonymize_phi: bool = False, **kwargs) -> OutputData:
        """
        Reads a DICOM file, optionally anonymizes PHI, and returns either 
        the pydicom dataset or the image array.

        Args:
            source_uri: Path to the DICOM file (local or mounted/cloud).
            as_numpy: If True, returns the standardized NumPy image pixel array.
            anonymize_phi: If True, scrubs Protected Health Information (PHI) before returning the data.
            **kwargs: Additional configuration parameters.

        Returns:
            OutputData: Either the pydicom Dataset or a standardized NumPy array.

        Raises:
            FileNotFoundError: If the DICOM file does not exist.
            IOError: If reading or processing the DICOM file fails.
        """
        if not os.path.exists(source_uri):
            raise FileNotFoundError(f"DICOM file not found: {source_uri}")
            
        try:
            # 1. Read the DICOM file
            dataset = pydicom.dcmread(source_uri)
            
            # 2. Hardening: Anonymize if required (Security step)
            if anonymize_phi:
                # Assuming anonymize_dicom handles inplace scrubbing or returns a new dataset
                dataset = anonymize_dicom(dataset) 
                logger.debug(f"[{self.connector_id}] PHI anonymization applied for: {source_uri}")
            
            # 3. Return data in requested format
            if as_numpy:
                # dicom_to_numpy should handle rescaling/windowing if necessary
                return dicom_to_numpy(dataset) 
            
            return dataset

        except pydicom.errors.InvalidDicomError as e:
            logger.error(f"[{self.connector_id}] Invalid DICOM file format: {source_uri}. Error: {e}")
            raise IOError(f"Invalid DICOM format: {e}")
        except Exception as e:
            logger.error(f"[{self.connector_id}] Error reading or converting DICOM file: {source_uri}. Error: {e}")
            raise IOError(f"DICOM Read Failed: {e}")


    def write(self, data: OutputData, destination_uri: str, file_format: str = 'parquet', **kwargs) -> str:
        """
        Writes extracted DICOM data (e.g., features, metadata) to a specified storage format.
        
        Args:
            data: Structured data (e.g., list of features/metadata dicts, or NumPy array).
            destination_uri: Path to the output file (e.g., /data/features.parquet).
            file_format: The target storage format (e.g., 'parquet', 'csv'). Defaults to 'parquet'.

        Returns:
            str: The final path/URI of the written data.

        Raises:
            NotImplementedError: If the requested file format is not supported.
            IOError: If writing the file fails.
        """
        try:
            if file_format.lower() == 'parquet':
                # save_numpy_as_parquet should handle persistence for structured data
                final_path = save_numpy_as_parquet(data, destination_uri)
                logger.info(f"[{self.connector_id}] DICOM data successfully written as Parquet to {final_path}")
                return final_path
            
            # Add support for saving processed image arrays as non-DICOM formats if needed
            # elif file_format.lower() == 'png':
            #     ...
            
            raise NotImplementedError(f"DICOM writing not implemented for format: {file_format}")
        except Exception as e:
            logger.error(f"[{self.connector_id}] Failed to write data to {destination_uri} in format {file_format}: {e}")
            raise IOError(f"DICOM Write Failed: {e}")

    def close(self):
        """
        Closes the connector (no resources to release for this connector).
        """
        super().close()
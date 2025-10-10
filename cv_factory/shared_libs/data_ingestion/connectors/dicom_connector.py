# cv_factory/shared_libs/data_ingestion/connectors/dicom_connector.py

import logging
import pydicom # Standard library for DICOM
import numpy as np
import os
from typing import Dict, Any, Union
from ..base.base_data_connector import BaseDataConnector, OutputData
from ..utils.dicom_utils import dicom_to_numpy, save_numpy_as_parquet # Assuming utility functions exist

logger = logging.getLogger(__name__)

class DICOMConnector(BaseDataConnector):
    """
    Concrete connector for handling DICOM files, used primarily in medical imaging.
    Supports reading DICOM image data and writing extracted features/metadata.
    """

    def connect(self) -> bool:
        """No external connection required."""
        self.is_connected = True
        return True

    def read(self, source_uri: str, as_numpy: bool = True, **kwargs) -> OutputData:
        """
        Reads a DICOM file and returns either the pydicom dataset or the image array.

        Args:
            source_uri (str): Path to the DICOM file.
            as_numpy (bool): If True, returns the image pixel array.
        """
        if not os.path.exists(source_uri):
            raise FileNotFoundError(f"DICOM file not found: {source_uri}")
            
        dataset = pydicom.dcmread(source_uri)
        
        if as_numpy:
            # Assuming dicom_to_numpy converts pixel data to a standardized NumPy format
            return dicom_to_numpy(dataset) 
        
        return dataset

    def write(self, data: OutputData, destination_uri: str, file_format: str = 'parquet', **kwargs) -> str:
        """
        Writes extracted DICOM data (e.g., features, metadata) to a specified storage format.
        
        Args:
            data (OutputData): Structured data (e.g., list of features/metadata dicts, or NumPy array).
            destination_uri (str): Path to the output file (e.g., /data/features.parquet).
            file_format (str): The target storage format (e.g., 'parquet', 'csv').
        """
        if file_format == 'parquet':
            # Assuming save_numpy_as_parquet handles the storage logic
            final_path = save_numpy_as_parquet(data, destination_uri)
            logger.info(f"DICOM data successfully written as Parquet to {final_path}")
            return final_path
        
        # Add support for other formats like 'csv', 'json' if needed
        raise NotImplementedError(f"DICOM writing not implemented for format: {file_format}")

    def close(self):
        super().close()
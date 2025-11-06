# cv_factory/shared_libs/ml_core/data/base_cv_dataset.py

import abc
from typing import Any, Dict, List, Optional, Union
import logging

# We define a common input/output structure for item retrieval
Sample = Dict[str, Any]
Batch = Union[List[Sample], Dict[str, Any]]

logger = logging.getLogger(__name__)

class BaseCVDataset(abc.ABC):
    """
    Abstract Base Class for all Computer Vision datasets.

    This class provides a standardized interface for loading data samples,
    independent of the underlying storage mechanism (managed by DataConnectors)
    or the specific ML framework used (e.g., PyTorch DataLoader compatibility).
    """

    def __init__(self, dataset_id: str, config: Dict[str, Any], data_connector: Any):
        """
        Initializes the base dataset.

        Args:
            dataset_id (str): A unique identifier for the dataset (e.g., 'training_v1').
            config (Dict[str, Any]): Configuration specific to data loading and sampling.
            data_connector (Any): An instance of a BaseDataConnector or a similar I/O abstraction,
                                  used to fetch data items (e.g., ImageConnector).
        """
        self.dataset_id = dataset_id
        self.config = config
        self.data_connector = data_connector
        self.metadata: Optional[List[Dict[str, Any]]] = None
        self._is_prepared = False

    @abc.abstractmethod
    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Loads the necessary metadata (e.g., file paths, labels, bounding box coordinates)
        from the source using the data connector. This defines the overall structure 
        of the dataset.
        
        Returns:
            List[Dict[str, Any]]: A list of metadata entries, one per sample.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        if not self._is_prepared:
            raise RuntimeError("Dataset length requested before preparation. Call prepare() first.")
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Sample:
        """
        Retrieves a single data sample at the given index. 
        This is the core method for the data pipeline.

        It should typically use the internal data_connector to fetch the raw data 
        and then apply any necessary preprocessing/transforms.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Sample: A dictionary containing the processed sample (e.g., {'image': array, 'label': int}).
        """
        raise NotImplementedError
        
    def prepare(self):
        """
        Prepares the dataset by loading metadata. Must be called before accessing data.
        """
        if not self._is_prepared:
            logger.info(f"[{self.dataset_id}] Preparing dataset: loading metadata...")
            self.metadata = self.load_metadata()
            self._is_prepared = True
            logger.info(f"[{self.dataset_id}] Dataset prepared successfully with {len(self)} samples.")
        else:
            logger.info(f"[{self.dataset_id}] Dataset already prepared.")
            
    # --- Optional Production Utilities ---
            
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Returns summary information about the dataset.
        """
        return {
            'id': self.dataset_id,
            'size': len(self) if self._is_prepared else 0,
            'config': self.config,
            'prepared': self._is_prepared
        }

    # NOTE: In Python, BaseCVDataset can implicitly inherit from torch.utils.data.Dataset
    # or be adapted for tf.data.Dataset in its concrete implementation.
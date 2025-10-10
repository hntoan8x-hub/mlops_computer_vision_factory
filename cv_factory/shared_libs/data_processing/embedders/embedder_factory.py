import logging
from typing import Dict, Any, Type, Optional

from shared_libs.data_processing._base.base_embedder import BaseEmbedder
from shared_libs.data_processing.embedders.atomic.cnn_embedder import CNNEmbedder
from shared_libs.data_processing.embedders.atomic.vit_embedder import ViTEmbedder

logger = logging.getLogger(__name__)

class EmbedderFactory:
    """
    A factory class for creating different types of deep learning embedders.

    This class centralizes the creation logic, allowing for a config-driven
    approach to building embedding pipelines.
    """
    _EMBEDDER_MAP: Dict[str, Type[BaseEmbedder]] = {
        "cnn": CNNEmbedder,
        "vit": ViTEmbedder,
        # Add new embedders here
    }

    @classmethod
    def create(cls, embedder_type: str, config: Optional[Dict[str, Any]] = None) -> BaseEmbedder:
        """
        Creates and returns an embedder instance based on its type.

        Args:
            embedder_type (str): The type of embedder to create (e.g., "cnn", "vit").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                to pass to the embedder's constructor.

        Returns:
            BaseEmbedder: An instance of the requested embedder.

        Raises:
            ValueError: If the specified embedder_type is not supported.
        """
        config = config or {}
        embedder_cls = cls._EMBEDDER_MAP.get(embedder_type.lower())
        
        if not embedder_cls:
            supported_embedders = ", ".join(cls._EMBEDDER_MAP.keys())
            logger.error(f"Unsupported embedder type: '{embedder_type}'. Supported types are: {supported_embedders}")
            raise ValueError(f"Unsupported embedder type: '{embedder_type}'. Supported types are: {supported_embedders}")

        logger.info(f"Creating instance of {embedder_cls.__name__}...")
        return embedder_cls(**config)
import abc
from typing import Any

class BaseValidator(abc.ABC):
    """
    Abstract Base Class for all pipeline validators.

    This interface ensures that all validators have a consistent way to check
    the schema and integrity of data at different stages of the pipeline.
    """

    @abc.abstractmethod
    def validate_input(self, data: Any) -> None:
        """
        Validates the input data to ensure it conforms to the required schema.

        Args:
            data (Any): The data to be validated.
        
        Raises:
            ValidationError: If the data does not conform to the schema.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_output(self, data: Any) -> None:
        """
        Validates the output data from a component.

        Args:
            data (Any): The data to be validated.

        Raises:
            ValidationError: If the data does not conform to the schema.
        """
        raise NotImplementedError
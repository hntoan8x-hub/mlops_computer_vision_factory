# cv_factory/shared_libs/core_utils/exceptions.py

class FactoryBaseException(Exception):
    """
    Base exception for all custom errors across the entire MLOps Factory.
    """
    pass

class DataIntegrityError(FactoryBaseException):
    """Raised when data structure, schema, or integrity checks fail (e.g., missing columns, wrong dimension)."""
    pass

class ConfigurationError(FactoryBaseException):
    """Raised when a configuration file error occurs after Pydantic validation (e.g., logic errors in config)."""
    pass

class PersistenceError(FactoryBaseException):
    """Raised when save/load/I/O operations fail unexpectedly."""
    pass
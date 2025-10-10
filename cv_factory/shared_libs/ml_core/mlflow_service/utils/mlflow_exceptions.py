class MLflowServiceException(Exception):
    """Base exception for all MLflow service-related errors."""
    pass

class MLflowAPIError(MLflowServiceException):
    """Raised when an API call to MLflow fails."""
    pass

class UnsupportedMLflowFeature(MLflowServiceException):
    """Raised when a requested feature is not supported."""
    pass
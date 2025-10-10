class SelectorException(Exception):
    """Base exception for all selector-related errors."""
    pass

class UnsupportedSelectorError(SelectorException):
    """Raised when an unsupported selector type is requested."""
    pass

class NoValidModelFound(SelectorException):
    """Raised when no model meets the selection criteria."""
    pass
class OrchestratorException(Exception):
    """Base exception for all orchestrator-related errors."""
    pass

class InvalidConfigError(OrchestratorException):
    """Raised when the configuration for an orchestrator is invalid."""
    pass

class WorkflowExecutionError(OrchestratorException):
    """Raised when an error occurs during the execution of a workflow step."""
    pass
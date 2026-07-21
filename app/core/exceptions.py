"""
Custom application exceptions.
"""


class PersonalAIException(Exception):
    """
    Base exception for the application.
    """

    def __init__(
        self,
        message: str,
        code: str = "APPLICATION_ERROR",
    ) -> None:
        super().__init__(message)

        self.message = message
        self.code = code


class LLMUnavailableError(PersonalAIException):
    """
    Raised when Ollama is unavailable.
    """

    def __init__(
        self,
        message: str = "Unable to connect to the local LLM.",
    ) -> None:

        super().__init__(
            message=message,
            code="LLM_UNAVAILABLE",
        )


class VectorStoreError(PersonalAIException):
    """
    Raised for vector database failures.
    """

    def __init__(
        self,
        message: str = "Vector database error.",
    ) -> None:

        super().__init__(
            message=message,
            code="VECTORSTORE_ERROR",
        )
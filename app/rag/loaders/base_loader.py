"""
Base document loader.
"""

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Base class for all document loaders.
    """

    @abstractmethod
    def load(
        self,
        file_path: str,
    ) -> str:
        """
        Load a document and return its text.

        Args:
            file_path: Path to the document.

        Returns:
            Extracted document text.
        """
        raise NotImplementedError
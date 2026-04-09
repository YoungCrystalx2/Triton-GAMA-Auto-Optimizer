from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for specialized mutation agents."""

    @abstractmethod
    def create_prompt(self, code: str) -> str:
        """Create a specialized mutation prompt."""
        raise NotImplementedError

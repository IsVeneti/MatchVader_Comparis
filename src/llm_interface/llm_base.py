from abc import ABC, abstractmethod
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AbstractLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from a single prompt."""
        pass


    def generate_with_schema(self, prompt: str, schema: Type[T]) -> T:
        """
        Generate a structured response from a prompt using the given schema.

        If your model supports structured output, inherit from SupportsStructuredOutput
        and implement `generate_structured`.

        Raises:
            NotImplementedError if structured output is not supported.
        """
        if hasattr(self, "generate_structured"):
            return self.generate_structured(prompt, schema)  # type: ignore
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support structured output. "
        )


class SupportsStructuredOutput(ABC):
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """Generate a structured object that conforms to the given Pydantic schema."""
        pass

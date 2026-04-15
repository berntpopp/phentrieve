from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMProvider(ABC):
    @abstractmethod
    def run_structured_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ):
        raise NotImplementedError

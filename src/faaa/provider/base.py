# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Sequence, Type, TypeVar

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from pydantic import BaseModel

from faaa.core.tool import ToolSchema

# Define generic variable, restricted to BaseModel subclasses
T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base class defining the interface for LLM clients."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_try: int = 3,
        default_model: str | None = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the LLM service API
            max_try: Maximum number of retry attempts
            default_model: Default model to use for API requests
        """
        self._api_key = api_key
        self._base_url = base_url
        self._max_try = max_try
        self._default_model = default_model

    @property
    def max_try(self) -> int:
        return self._max_try

    @max_try.setter
    def max_try(self, value: int):
        self._max_try = value

    @property
    def default_model(self) -> str | None:
        return self._default_model

    @default_model.setter
    def default_model(self, value: str | None):
        self._default_model = value

    @abstractmethod
    async def chat(
        self,
        messages: str | Sequence[ChatCompletionMessageParam],
        model: str,
        max_tokens: int,
    ) -> ChatCompletionMessage:
        """
        Asynchronously sends a chat request and returns the model's response.

        Parameters:
            messages (str or list[dict]): List of message dictionaries containing the conversation history.
            model (str): The model to use for generating completions.
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            ChatCompletionMessage: The response message from the model.
        """
        pass

    @abstractmethod
    async def embeddings(self, input_text: str, model: str):
        """
        Generate embeddings for input text using a specified model.

        Args:
            input_text (str): The text to generate embeddings for.
            model (str): The model to use for generating embeddings.

        Returns:
            list: A list containing the embedding data.
        """
        pass

    @abstractmethod
    async def structured_output(
        self,
        messages: Iterable[ChatCompletionMessageParam] | str,
        structured_outputs: Type[T],
        model: str,
        max_tokens: int,
        max_try: int | None = None,
    ) -> T:
        """
        Asynchronously parses messages and returns structured output.

        Args:
            messages: Messages to be parsed.
            structured_outputs: The expected type of the structured output.
            model: The model to be used for parsing.
            max_tokens: Maximum number of tokens allowed.
            max_try: Maximum number of retry attempts.

        Returns:
            T: The parsed structured output.
        """
        pass

    @abstractmethod
    async def function_call(
        self,
        messages: list[ChatCompletionMessageParam] | str,
        tool_schemas: list[ToolSchema],
        *,
        max_try: int | None = None,
    ):
        """
        Asynchronously calls a function with the provided messages and tool schemas.

        Args:
            messages (list[ChatCompletionMessageParam] | str): The messages to be processed, either as a list of ChatCompletionMessageParam objects or a single string.
            tool_schemas (list[ToolSchema]): A list of tool schemas to be used in the function call.
            max_try (int | None, optional): The maximum number of attempts to try the function call. Defaults to None.

        Returns:
            None
        """
        pass

    @abstractmethod
    async def tool_description(self, func: Callable) -> ToolSchema:
        """
        Asynchronously retrieves the tool description.

        Args:
            func (Callable): A callable function that provides the tool description.

        Returns:
            ToolSchema: The schema containing the tool description.
        """
        pass

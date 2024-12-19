# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import asyncio
import inspect
import os
from typing import Callable, Sequence, Type, TypeVar

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from pydantic import BaseModel

from faaa.core.tool_schema import Tool, convert_tool_schema_to_openai_tool
from faaa.exception import RefusalError
from faaa.prompt import CODE_SUMMARY_INSTRUCTION, TOOL_CALLING_INSTRUCTION

load_dotenv()

# 定义泛型变量，限制为 BaseModel 的子类
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    _instance = None
    _API_KEY = os.getenv("OPENAI_API_KEY")
    _BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    _max_try = 3

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMClient, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        self._client = AsyncOpenAI(base_url=self._BASE_URL, api_key=self._API_KEY)

    @property
    def max_try(self):
        return self._max_try

    @max_try.setter
    def max_try(self, value: int):
        self._max_try = value

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI):
        self._client = value

    async def chat(
        self,
        messages: str | Sequence[ChatCompletionMessageParam],
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 500,
    ) -> ChatCompletionMessage:
        """
        Asynchronously sends a chat request to the OpenAI API and returns the model's response.

        Parameters:
            messages (str or list[dict]): List of message dictionaries containing the conversation history.
            model (str, optional): The model to use for generating completions. Defaults to "openai/gpt-4o-mini".
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.

        Returns:
            openai.types.chat.ChatCompletion.Choice.Message: The response message from the model.

        Raises:
            RefusalError: If the token limit is exceeded.
            Exception: For other API-related errors.

        Example:
            messages = [
                {"role": "user", "content": "Hello, how are you?"}
            ]
            response = await chat(messages)
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            completion = await self._client.chat.completions.create(
                messages=messages, model=model, max_tokens=max_tokens
            )
            return completion.choices[0].message
        except Exception as e:
            if isinstance(e, openai.LengthFinishReasonError):
                raise RefusalError(f"Too many tokens: {e}")
            else:
                raise e

    async def embeddings(self, input_text: str, model: str = "openai/text-embedding-ada-002"):
        """
        Generate embeddings for input text using a specified model.

        This asynchronous method creates embeddings using the OpenAI API. Embeddings are vector
        representations of text that capture semantic meaning, useful for tasks like semantic
        search and text similarity comparisons.

        Args:
            input_text (str): The text to generate embeddings for.
            model (str, optional): The model to use for generating embeddings.
                Defaults to "openai/text-embedding-ada-002".

        Returns:
            list: A list containing the embedding data from the API response.

        Raises:
            Exception: Propagates any exceptions that occur during the API call.

        Example:
            ```
            embeddings = await llm.embeddings("Hello world")
            ```
        """
        try:
            response = await self._client.embeddings.create(input=input_text, model=model)
            return response.data
        except Exception as e:
            raise e

    async def parse(
        self,
        *msg: ChatCompletionMessageParam,
        structured_outputs: Type[T],
        model: str = "openai/gpt-4o-mini",
        max_token: int = 200,
        max_try: int | None = None,  # Add max_try parameter
    ) -> T:
        """
        Asynchronously parses a list of messages using a specified language model and returns the structured output.

        Args:
            *msg (ChatCompletionMessageParam): A variable number of message dictionaries to be parsed.
            structured_outputs (Type[T]): The expected type of the structured output.
            model (str, optional): The model to be used for parsing. Defaults to "openai/gpt-4o-mini".
            max_token (int, optional): The maximum number of tokens allowed in the response. Defaults to 200.
            max_try (int, optional): Maximum number of retry attempts. Defaults to 3.

        Returns:
            T: The parsed structured output.

        Raises:
            RefusalError: If the response contains a refusal message or if the token limit is exceeded.
            ValueError: If no choices are found in the completion response.
            Exception: For other exceptions that may occur during parsing.
        """
        attempt = 0
        last_error = None

        if max_try is None:
            max_try = self._max_try

        while attempt < max_try:
            try:
                completion = await self._client.beta.chat.completions.parse(
                    messages=msg,
                    model=model,
                    response_format=structured_outputs,
                    max_tokens=max_token,
                )
                if completion.choices:
                    response = completion.choices[0].message
                    if hasattr(response, "parsed") and response.parsed:
                        return response.parsed
                    elif hasattr(response, "refusal") and response.refusal:
                        last_error = RefusalError(response.refusal)
                        break  # Don't retry on explicit refusal
                else:
                    last_error = ValueError("No choices found in the completion response")
            except openai.LengthFinishReasonError as e:
                last_error = RefusalError(f"Too many tokens: {e}")
            except Exception as e:
                last_error = e

            attempt += 1
            if attempt < max_try:
                # Exponential backoff between retries
                await asyncio.sleep(2**attempt)

        # If we've exhausted all retries, raise the last error
        if isinstance(last_error, RefusalError):
            raise last_error
        else:
            raise ValueError(f"Failed to parse after {max_try} attempts: {last_error}")

    async def function_call(
        self,
        msg: list[ChatCompletionMessageParam] | str,
        tool_schemas: list[Tool],
        *,
        max_try: int | None = None,
    ):
        attempt = 0
        last_error = None

        if max_try is None:
            max_try = self._max_try
        if isinstance(msg, str):
            msg = [{"role": "user", "content": msg}]
        msg.insert(0, {"role": "system", "content": TOOL_CALLING_INSTRUCTION})
        while attempt < max_try:
            try:
                tools = [convert_tool_schema_to_openai_tool(schema) for schema in tool_schemas]
                completion = await self._client.chat.completions.create(
                    messages=msg,
                    model="openai/gpt-4o-mini",
                    tools=tools,
                    tool_choice="auto",
                )
                if completion.choices:
                    response = completion.choices[0].message
                    if response.tool_calls:
                        return response.tool_calls
                    elif response.refusal:
                        last_error = RefusalError(response.refusal)
                    else:
                        last_error = ValueError(
                            f"No tool calls found in the completion response: {response.content}"
                        )
                else:
                    last_error = ValueError("No choices found in the completion response")
            except Exception as e:
                if isinstance(e, openai.LengthFinishReasonError):
                    last_error = RefusalError(f"Too many tokens: {e}")
                else:
                    last_error = e

            attempt += 1
            if attempt < max_try:
                # Exponential backoff between retries
                await asyncio.sleep(2**attempt)

        # If we've exhausted all retries, raise the last error
        if isinstance(last_error, RefusalError):
            raise last_error
        else:
            raise ValueError(f"Failed to parse after {max_try} attempts: {last_error}")

    async def generate_tool_description(self, func: Callable) -> Tool:
        """
        Asynchronously generates a schema for a given function using LLM.

        This method takes a Python callable and extracts its metadata (name, signature, docstring)
        to generate a structured ToolSchema by leveraging an LLM to analyze the function.

        Args:
            func (Callable): The Python function to analyze and generate schema for.
                            Must be a valid callable object.

        Returns:
            ToolSchema: A structured schema containing function metadata including:
                       - name
                       - description
                       - tags
                       - parameters

        Raises:
            ValueError: If the provided func argument is not a callable
            Exception: If schema generation fails

        Example:
            ```python
            async def my_func(x: int) -> str:
                return str(x)

            schema = await generate_tool_description(my_func)
            ```
        """
        if not callable(func):
            raise ValueError("The provided func must be a callable")

        # Get function details
        name = func.__name__
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)
        code = inspect.getsource(func).strip()

        code_msg = f"""
        <Function name>
        {name}
        </Function name>

        <Function signature>
        {signature}
        </Function signature>

        {'<Function docstring>' if docstring else '<Function source code>'}
        {docstring if docstring else code}
        {'</Function docstring>' if docstring else '</Function source code>'}
        """

        try:
            return await self.parse(
                {"role": "system", "content": CODE_SUMMARY_INSTRUCTION},
                {"role": "user", "content": code_msg},
                structured_outputs=Tool,
            )
        except Exception as e:
            raise e

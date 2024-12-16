# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import inspect
import os
from typing import Callable, Type, TypeVar

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel

from faaa.exception import RefusalError
from faaa.schema import ToolSchema

load_dotenv()

# 定义泛型变量，限制为 BaseModel 的子类
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    _instance = None
    _API_KEY = os.getenv("OPENAI_API_KEY")
    _BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    max_try = 3

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMClient, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        self._client = AsyncOpenAI(base_url=self._BASE_URL, api_key=self._API_KEY)

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI):
        self._client = value

    async def chat(
        self, messages: str | list[dict], model: str = "openai/gpt-4o-mini", max_tokens: int = 500
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
        *msg: list[dict],
        structured_outputs: Type[T],
        model: str = "openai/gpt-4o-mini",
        max_token: int = 200,
    ) -> T:
        """
        Asynchronously parses a list of messages using a specified language model and returns the structured output.

        Args:
            *msg (list[dict]): A variable number of message dictionaries to be parsed.
            structured_outputs (Type[T]): The expected type of the structured output.
            model (str, optional): The model to be used for parsing. Defaults to "openai/gpt-4o-mini".
            max_token (int, optional): The maximum number of tokens allowed in the response. Defaults to 200.

        Returns:
            T: The parsed structured output.

        Raises:
            RefusalError: If the response contains a refusal message or if the token limit is exceeded.
            ValueError: If no choices are found in the completion response.
            Exception: For other exceptions that may occur during parsing.
        """
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
                    # handle refusal
                    raise RefusalError(response.refusal)
            else:
                raise ValueError("No choices found in the completion response")
        except Exception as e:
            # Handle edge cases
            if isinstance(e, openai.LengthFinishReasonError):
                # Retry with a higher max tokens
                raise RefusalError(f"Too many tokens: {e}")
            else:
                # Handle other exceptions
                raise e

    @classmethod
    def convert_tool_schema_to_openai_tool(cls, tool_schema: ToolSchema) -> dict:
        """
        Convert a ToolSchema instance to match the tools definition used in the OpenAI's API.

        Args:
            tool_schema (ToolSchema): The ToolSchema instance to convert.

        Returns:
            dict: A dictionary matching the OpenAI's tools API definition.
        """
        return {
            "name": tool_schema.name,
            "description": tool_schema.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                    }
                    for param in tool_schema.parameters
                },
                "required": [param.name for param in tool_schema.parameters if param.required],
            },
        }

    async def function_call(self, msg: list[dict], tool_schemas: list[ToolSchema]):
        try_count = 0
        while try_count < self.max_try:
            try:
                tools = [self.convert_tool_schema_to_openai_tool(schema) for schema in tool_schemas]
                completion = await self._client.chat.completions.create(
                    messages=msg,
                    model="openai/gpt-4o-mini",
                    functions=tools,
                    function_call="auto",
                )
                if completion.choices:
                    response = completion.choices[0].message
                    if response.function_call:
                        return response.function_call
                    elif response.refusal:
                        try_count += 1
                        continue
                else:
                    raise ValueError("No choices found in the completion response")
            except Exception as e:
                if isinstance(e, openai.LengthFinishReasonError):
                    raise RefusalError(f"Too many tokens: {e}")
                else:
                    raise e
        raise RefusalError("LLM refused to run your function")

    async def generate_tool_schema(self, func: Callable) -> ToolSchema:
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

            schema = await generate_tool_schema(my_func)
            ```
        """
        if not callable(func):
            raise ValueError("The provided func must be a callable")

        # Get function details
        name = func.__name__
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)
        code = inspect.getsource(func).strip()

        instruct = """
        You are an intelligent assistant. Your task is to generate a JSON schema in the
        `response_format` structured output format for the provided Python function details.

        The provided information includes:
        1. Function name
        2. Function signature
        3. Function docstring or source code

        Include:
        - Function name
        - Function description (from docstring or code)
        - Tags (max 3 tags describing function usage)
        - Parameters (name, type, description, required status. 
          If the parameter type or description is not clear, you can assume it yourself)
        """

        code_msg = f"""
        Please process:
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
                {"role": "system", "content": instruct},
                {"role": "user", "content": code_msg},
                structured_outputs=ToolSchema,
            )
        except Exception as e:
            raise e

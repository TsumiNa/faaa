# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import asyncio
import inspect
import os
from typing import Callable, Iterable, Sequence, Type, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI, LengthFinishReasonError
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel

from faaa.core.exception import RefusalError
from faaa.core.prompt import CODE_SUMMARY_INSTRUCTION, TOOL_CALLING_INSTRUCTION
from faaa.core.tool import ToolMetaSchema
from faaa.provider.base import BaseLLMClient

load_dotenv()

# Define generic variable, restricted to BaseModel subclasses
T = TypeVar("T", bound=BaseModel)


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_try: int = 3,
        default_model: str = "openai/gpt-4o-mini",
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        super().__init__(api_key=api_key, base_url=base_url, max_try=max_try, default_model=default_model)
        self._initialize_client()

    def _initialize_client(self):
        self._client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI):
        self._client = value

    async def chat(
        self,
        messages: str | Sequence[ChatCompletionMessageParam],
        model: str | None = None,
        max_tokens: int = 500,
    ) -> ChatCompletionMessage:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            completion = await self._client.chat.completions.create(
                messages=messages, model=model or self.default_model, max_tokens=max_tokens
            )
            return completion.choices[0].message
        except Exception as e:
            if isinstance(e, LengthFinishReasonError):
                raise RefusalError(f"Too many tokens: {e}")
            else:
                raise e

    async def embeddings(self, input_text: str, model: str | None = None):
        try:
            response = await self._client.embeddings.create(
                input=input_text, model=model or "openai/text-embedding-ada-002"
            )
            return response.data
        except Exception as e:
            raise e

    async def structured_output(
        self,
        messages: Iterable[ChatCompletionMessageParam] | str,
        structured_outputs: Type[T],
        model: str | None = None,
        max_tokens: int = 500,
        max_try: int | None = None,
    ) -> T:
        attempt = 0
        last_error = None

        if max_try is None:
            max_try = self._max_try

        if isinstance(messages, str):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that generates structured outputs in JSON format. "
                        "Please ensure your response adheres strictly to the specified structure."
                    ),
                },
                {"role": "user", "content": messages},
            ]

        while attempt < max_try:
            try:
                completion = await self._client.beta.chat.completions.parse(
                    messages=messages,
                    model=model or self.default_model,
                    response_format=structured_outputs,
                    max_tokens=max_tokens,
                )
                if completion.choices:
                    response = completion.choices[0].message
                    if hasattr(response, "parsed") and response.parsed:
                        return response.parsed
                    elif hasattr(response, "refusal") and response.refusal:
                        raise RefusalError(response.refusal)
                    else:
                        last_error = ValueError(
                            f"No structured output found in the completion response: {response}"
                        )
                else:
                    last_error = ValueError("No choices found in the completion response")
            except LengthFinishReasonError as e:
                raise RefusalError(f"Too many tokens: {e}")
            except Exception as e:
                last_error = e

            attempt += 1
            if attempt < max_try:
                # Exponential backoff between retries
                await asyncio.sleep(1**attempt)

        raise last_error

    async def function_call(
        self,
        messages: list[ChatCompletionMessageParam] | str,
        tool_schemas: list[ToolMetaSchema],
        *,
        max_try: int | None = None,
    ):
        attempt = 0
        last_error = None

        if max_try is None:
            max_try = self._max_try
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        messages.insert(0, {"role": "system", "content": TOOL_CALLING_INSTRUCTION})
        while attempt < max_try:
            try:
                tools = [self.build_openai_tool_parameter(schema) for schema in tool_schemas]
                completion = await self._client.chat.completions.create(
                    messages=messages,
                    model=self.default_model,
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
                        return response
                else:
                    last_error = ValueError("No choices found in the completion response")
            except Exception as e:
                if isinstance(e, LengthFinishReasonError):
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

    async def tool_description(self, func: Callable) -> ToolMetaSchema:
        if not callable(func):
            raise ValueError("The provided func must be a callable")

        # Get function details
        name = func.__name__
        signature = inspect.signature(func)
        docstring = inspect.cleandoc(func.__doc__ or "")
        code = inspect.getsource(func).strip()

        code_msg = f"""
        <Function name>
        {name}
        </Function name>

        <Function signature>
        {signature}
        </Function signature>

        {'<Function docstring>' if docstring else '<Function source code>'}
        {docstring if docstring != "" else code}
        {'</Function docstring>' if docstring else '</Function source code>'}
        """

        try:
            query = [
                {"role": "system", "content": CODE_SUMMARY_INSTRUCTION},
                {"role": "user", "content": code_msg},
            ]
            return await self.structured_output(
                query,
                structured_outputs=ToolMetaSchema,
            )
        except Exception as e:
            raise e

    @classmethod
    def build_openai_tool_parameter(cls, tool_schema: ToolMetaSchema) -> ChatCompletionToolParam:
        """
        Constructs a ChatCompletionToolParam object based on the provided ToolSchema.

        Args:
            tool_schema (ToolSchema): The schema defining the tool's parameters, name, and description.

        Returns:
            ChatCompletionToolParam: An object containing the tool's type, name, description, and parameters formatted for OpenAI's chat completion.
        """
        _parameters = dict(
            type="object",
            properties={
                param.name: {
                    "type": param.type,
                    "description": param.description,
                }
                for param in tool_schema.parameters
            },
            required=[param.name for param in tool_schema.parameters if param.required],
        )

        return ChatCompletionToolParam(
            {
                "type": "function",
                "function": {
                    "name": tool_schema.name,
                    "description": tool_schema.description,
                    "parameters": _parameters,
                },
            }
        )

# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import pytest
from openai.types.chat import ChatCompletionMessageParam

from faaa.core.exception import RefusalError
from faaa.core.tool_schema import ToolSchema
from faaa.provider import OpenAIClient


@pytest.mark.asyncio
async def test_chat_success():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    response = await client.chat(messages)
    assert response is not None


@pytest.mark.asyncio
async def test_chat_too_many_tokens():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    with pytest.raises(RefusalError):
        await client.chat(messages, max_tokens=1)


@pytest.mark.asyncio
async def test_chat_other_exception():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    client.client = None  # Force an error
    with pytest.raises(Exception):
        await client.chat(messages)


@pytest.mark.asyncio
async def test_embeddings_success():
    client = OpenAIClient()
    input_text = "Hello world"
    response = await client.embeddings(input_text)
    assert response is not None


@pytest.mark.asyncio
async def test_embeddings_exception():
    client = OpenAIClient()
    input_text = "Hello world"
    client.client = None  # Force an error
    with pytest.raises(Exception):
        await client.embeddings(input_text)


@pytest.mark.asyncio
async def test_parse_success():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    response = await client.structured_output(messages, structured_outputs=ToolSchema)
    assert response is not None


@pytest.mark.asyncio
async def test_parse_refusal_error():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    client.client = None  # Force an error
    with pytest.raises(Exception):
        await client.structured_output(messages, structured_outputs=ToolSchema)


@pytest.mark.asyncio
async def test_parse_no_choices():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    client.client = None  # Force an error
    with pytest.raises(Exception):
        await client.structured_output(messages, structured_outputs=ToolSchema)


@pytest.mark.asyncio
async def test_parse_too_many_tokens():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    with pytest.raises(RefusalError):
        await client.structured_output(messages, structured_outputs=ToolSchema, max_tokens=1)


@pytest.mark.asyncio
async def test_parse_other_exception():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    client.client = None  # Force an error
    with pytest.raises(Exception):
        await client.structured_output(messages, structured_outputs=ToolSchema)


@pytest.mark.asyncio
async def test_function_call_success():
    client = OpenAIClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [
        ToolSchema(
            name="test_function",
            description="A test function",
            parameters=[],
        )
    ]
    response = await client.function_call(messages, tool_schemas)
    assert response is not None


@pytest.mark.asyncio
async def test_function_call_refusal():
    client = OpenAIClient()
    messages = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [
        ToolSchema(
            name="test_function",
            description="A test function",
            parameters=[],
        )
    ]
    client.client = None  # Force an error
    with pytest.raises(Exception):
        await client.function_call(messages, tool_schemas)


@pytest.mark.asyncio
async def test_function_call_no_choices():
    client = OpenAIClient()
    messages = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [
        ToolSchema(
            name="test_function",
            description="A test function",
            parameters=[],
        )
    ]
    client.client = None  # Force an error
    with pytest.raises(Exception):
        await client.function_call(messages, tool_schemas)


@pytest.mark.asyncio
async def test_function_call_too_many_tokens():
    client = OpenAIClient()
    messages = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [
        ToolSchema(
            name="test_function",
            description="A test function",
            parameters=[],
        )
    ]
    with pytest.raises(ValueError):
        await client.function_call(messages, tool_schemas, max_try=1)


@pytest.mark.asyncio
async def test_function_call_other_exception():
    client = OpenAIClient()
    messages = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [
        ToolSchema(
            name="test_function",
            description="A test function",
            parameters=[],
        )
    ]
    client.client = None  # Force an error
    with pytest.raises(Exception):
        await client.function_call(messages, tool_schemas)

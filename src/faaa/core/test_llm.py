from unittest.mock import AsyncMock, Mock, patch

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam

from faaa.core.llm import LLMClient, RefusalError
from faaa.core.tool_schema import Tool


@pytest.mark.asyncio
async def test_chat_success():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    expected_response = {"role": "assistant", "content": "I'm good, thank you!"}

    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=AsyncMock(choices=[AsyncMock(message=expected_response)])),
    ):
        response = await client.chat(messages)
        assert response == expected_response


@pytest.mark.asyncio
async def test_chat_too_many_tokens():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello, how are you?"}]

    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(side_effect=openai.LengthFinishReasonError(completion=Mock(usage="test_usage"))),
    ):
        with pytest.raises(RefusalError, match="Too many tokens"):
            await client.chat(messages)


@pytest.mark.asyncio
async def test_chat_other_exception():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello, how are you?"}]

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(side_effect=Exception("API error"))
    ):
        with pytest.raises(Exception, match="API error"):
            await client.chat(messages)


@pytest.mark.asyncio
async def test_embeddings_success():
    client = LLMClient()
    input_text = "Hello world"
    expected_response = [0.1, 0.2, 0.3]

    with patch.object(
        client._client.embeddings,
        "create",
        new=AsyncMock(return_value=AsyncMock(data=expected_response)),
    ):
        response = await client.embeddings(input_text)
        assert response == expected_response


@pytest.mark.asyncio
async def test_embeddings_exception():
    client = LLMClient()
    input_text = "Hello world"

    with patch.object(client._client.embeddings, "create", new=AsyncMock(side_effect=Exception("API error"))):
        with pytest.raises(Exception, match="API error"):
            await client.embeddings(input_text)


@pytest.mark.asyncio
async def test_parse_success():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    structured_outputs = Tool
    expected_response = Mock(parsed={"key": "value"})

    with patch.object(
        client._client.beta.chat.completions,
        "parse",
        new=AsyncMock(return_value=AsyncMock(choices=[AsyncMock(message=expected_response)])),
    ):
        response = await client.parse(*messages, structured_outputs=structured_outputs)
        assert response == expected_response.parsed


@pytest.mark.asyncio
async def test_parse_refusal_error():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    structured_outputs = Tool
    refusal_message = Mock(refusal="Refusal reason")

    with patch.object(
        client._client.beta.chat.completions,
        "parse",
        new=AsyncMock(return_value=AsyncMock(choices=[AsyncMock(message=refusal_message)])),
    ):
        with pytest.raises(RefusalError, match="Refusal reason"):
            await client.parse(*messages, structured_outputs=structured_outputs)


@pytest.mark.asyncio
async def test_parse_no_choices():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    structured_outputs = Tool

    with patch.object(
        client._client.beta.chat.completions,
        "parse",
        new=AsyncMock(return_value=AsyncMock(choices=[])),
    ):
        with pytest.raises(ValueError, match="No choices found in the completion response"):
            await client.parse(*messages, structured_outputs=structured_outputs)


@pytest.mark.asyncio
async def test_parse_too_many_tokens():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    structured_outputs = Tool

    with patch.object(
        client._client.beta.chat.completions,
        "parse",
        new=AsyncMock(side_effect=openai.LengthFinishReasonError(completion=Mock(usage="test_usage"))),
    ):
        with pytest.raises(RefusalError, match="Too many tokens"):
            await client.parse(*messages, structured_outputs=structured_outputs)


@pytest.mark.asyncio
async def test_parse_other_exception():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Parse this message"}]
    structured_outputs = Tool

    with patch.object(
        client._client.beta.chat.completions,
        "parse",
        new=AsyncMock(side_effect=Exception("API error")),
    ):
        with pytest.raises(Exception, match="API error"):
            await client.parse(*messages, structured_outputs=structured_outputs)


@pytest.mark.asyncio
async def test_function_call_success():
    client = LLMClient()
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [Mock(parameters=[], spec=Tool)]
    expected_function_call = {"name": "test_function", "arguments": {}}

    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(
            return_value=AsyncMock(choices=[AsyncMock(message=Mock(function_call=expected_function_call))])
        ),
    ):
        response = await client.function_call(messages, tool_schemas)
        assert response == expected_function_call


@pytest.mark.asyncio
async def test_function_call_refusal():
    client = LLMClient()
    messages = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [
        Mock(parameters=[Mock(name="param1", type="string", description="A parameter", required=True)])
    ]
    refusal_message = Mock(refusal="Refusal reason")

    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=AsyncMock(choices=[AsyncMock(message=refusal_message)])),
    ):
        with pytest.raises(RefusalError, match="LLM refused to run your function"):
            await client.function_call(messages, tool_schemas)


@pytest.mark.asyncio
async def test_function_call_no_choices():
    client = LLMClient()
    messages = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [Mock(parameters=[])]

    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=AsyncMock(choices=[])),
    ):
        with pytest.raises(ValueError, match="No choices found in the completion response"):
            await client.function_call(messages, tool_schemas)


@pytest.mark.asyncio
async def test_function_call_too_many_tokens():
    client = LLMClient()
    messages = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [Mock(parameters=[])]

    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(side_effect=openai.LengthFinishReasonError(completion=Mock(usage="test_usage"))),
    ):
        with pytest.raises(RefusalError, match="Too many tokens"):
            await client.function_call(messages, tool_schemas)


@pytest.mark.asyncio
async def test_function_call_other_exception():
    client = LLMClient()
    messages = [{"role": "user", "content": "Call this function"}]
    tool_schemas = [Mock(spec=Tool)]
    tool_schemas[0].name = "mock_tool"
    tool_schemas[0].description = "mock_description"
    tool_schemas[0].parameters = []

    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(side_effect=Exception("API error")),
    ):
        with pytest.raises(Exception, match="API error"):
            await client.function_call(messages, tool_schemas)

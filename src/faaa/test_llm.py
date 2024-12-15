from unittest.mock import MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from faaa.llm import RefusalError, get_client


class MyOutput(BaseModel):
    text: str


@pytest.fixture
def mock_openai_client():
    with patch("faaa.llm.__client") as mock_client:
        yield mock_client


def test_get_client_plain_text(mock_openai_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message="Hello, world!")]
    mock_openai_client.chat.completions.create.return_value = mock_response

    client = get_client(max_token=100)
    response = client([{"role": "user", "content": "Hello"}])
    assert response == "Hello, world!"
    mock_openai_client.chat.completions.create.assert_called_once()


def test_get_client_structured_output(mock_openai_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(parsed=MyOutput(text="Hello, world!")))]
    mock_openai_client.beta.chat.completions.parse.return_value = mock_response

    client = get_client(structured_outputs=MyOutput)
    response = client([{"role": "user", "content": "Hello"}])
    assert response == MyOutput(text="Hello, world!")
    mock_openai_client.beta.chat.completions.parse.assert_called_once()


def test_get_client_refusal_error(mock_openai_client):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(parsed=None, refusal="Refused to generate response"))
    ]
    mock_openai_client.beta.chat.completions.parse.return_value = mock_response

    client = get_client(structured_outputs=MyOutput)
    with pytest.raises(RefusalError):
        client([{"role": "user", "content": "Hello"}])


def test_get_client_length_finish_reason_error(mock_openai_client):
    mock_completion = MagicMock()
    mock_completion.usage = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = openai.LengthFinishReasonError(
        completion=mock_completion
    )

    client = get_client(max_token=100)
    with pytest.raises(RefusalError):
        client([{"role": "user", "content": "Hello"}])

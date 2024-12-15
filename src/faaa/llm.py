# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import os
from typing import Any, Callable

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from .exception import RefusalError

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

__client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def get_client(*, max_token: int = 500, structured_outputs: BaseModel | None = None) -> Callable[..., Any]:
    """Returns a callable function that interacts with an LLM client for text generation or structured output parsing.

    This function creates and returns a specialized response parsing function based on whether structured outputs
    are required. It handles both regular text completions and structured data parsing with error handling.

    Args:
        max_token (int, optional): Maximum number of tokens for the response. Defaults to 500.
        structured_outputs (BaseModel | None, optional): Pydantic model for structured output parsing.
            If None, returns plain text responses. Defaults to None.

    Returns:
        Callable[..., Any]: A function that processes messages and returns either parsed structured data
            or plain text responses.

    Raises:
        RefusalError: When the model refuses to generate a response or token limit is exceeded.
        Exception: For other API or processing errors.

    Example:
        >>> client = get_client(max_token=1000)
        >>> response = client([{"role": "user", "content": "Hello"}])

        >>> from pydantic import BaseModel
        >>> class MyOutput(BaseModel):
        ...     text: str
        >>> structured_client = get_client(structured_outputs=MyOutput)
        >>> parsed_response = structured_client([{"role": "user", "content": "Hello"}],
        ...                                    response_format={"type": "object"})
    """
    if structured_outputs:

        def _parse_response(
            *msg,
        ):
            try:
                completion = __client.beta.chat.completions.parse(
                    messages=msg,
                    model="openai/gpt-4o-mini",
                    response_format=structured_outputs,
                    max_tokens=200,
                )
                if completion.choices:
                    response = completion.choices[0].message
                    if response.parsed:
                        return response.parsed
                    elif response.refusal:
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

        return _parse_response

    def _parse_response(*msg):
        try:
            completion = __client.chat.completions.create(
                messages=msg, model="openai/gpt-4o-mini", max_tokens=max_token
            )
            return completion.choices[0].message
        except Exception as e:
            # Handle edge cases
            if isinstance(e, openai.LengthFinishReasonError):
                # Raise RefusalError for length finish reason
                raise RefusalError(f"Too many tokens: {e}")
            else:
                # Handle other exceptions
                raise e

    return _parse_response

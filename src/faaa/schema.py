# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import inspect
from typing import List

from pydantic import BaseModel

from faaa.llm import get_client


class ParameterSchema(BaseModel):
    """
    ParameterSchema is a Pydantic model that represents a parameter with the following attributes:

    Attributes:
        name (str): The name of the parameter.
        type (str): The type of the parameter.
        description (str): A brief description of the parameter.
        required (bool): Whether this parameter is required or has a default value.
    """

    name: str
    type: str
    description: str
    required: bool


class ToolSchema(BaseModel):
    """
    ToolSchema represents the schema for a tool with its name, description, and parameters.

    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        parameters (List[ParameterSchema]): A list of parameters associated with the tool.
    """

    name: str
    description: str
    parameters: List[ParameterSchema]


def generate_tool_schema(func) -> ToolSchema:
    """
    Generate a JSON schema for a Python function in ToolSchema format.

    This function takes a Python function as input and generates a standardized JSON schema
    that describes the function's name, description, and parameters. It uses introspection
    to extract the function's source code, signature, and docstring, then leverages an LLM
    to generate the structured schema.

    Args:
        func: The Python function to generate a schema for.
            Must be a valid Python function with accessible source code.

    Returns:
        ToolSchema: A structured schema containing:
            - name: The function name
            - description: Function description from docstring
            - parameters: List of parameter details including name, type and description
            - required: Whether this parameter is required or has a default value.

    Raises:
        Exception: If schema generation fails or LLM encounters an error

    Example:
        >>> def add(x: int, y: int) -> int:
        ...     '''Add two numbers'''
        ...     return x + y
        >>> schema = generate_tool_schema(add)
    """
    # Get function source code and details
    source_code = inspect.getsource(func).strip()
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""

    # Create prompt for LLM
    instruct = """
    You are a helpful customer support assistant. Assist the user to do the following task.

    Generate a JSON schema for the give information about a python function.
    The given information will include:
    1. Function code
    2. Function docstring
    3. Function signature

    Required output format:
    {
        "name": "function_name",
        "description": "function description",
        "parameters": [
            {
                "name": "param name",
                "type": "param type",
                "description": "param description",
                "required": "Whether this parameter is required or has a default value"
            },
        ]
    }
    """

    code = f"""
    Please help me to process the following information:

    Source:
    {source_code}

    Signature:
    {signature}

    Documentation:
    {docstring}
    """

    # Get LLM response and parse
    llm = get_client(structured_outputs=ToolSchema)
    try:
        return llm({"role": "system", "content": instruct}, {"role": "user", "content": code})
    except Exception as e:
        raise e

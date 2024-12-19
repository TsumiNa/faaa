# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

from typing import List

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """
    ToolParameter is a Pydantic model that represents a parameter with the following attributes:

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


class Tool(BaseModel):
    """
    Tool represents the schema for a tool with its name, description, and parameters.

    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        tags (List[str]): A list of tags describe the catalog of the tool.
        parameters (List[ParameterSchema]): A list of parameters associated with the tool.
    """

    name: str
    description: str
    tags: List[str]
    parameters: List[ToolParameter]


class ToolCallParam(BaseModel):
    """
    ToolCallParam is a model that defines the schema for tool call parameters.

    Attributes:
        type (str): The type of the parameter, default is "object".
        required (list[str]): A list of required parameter names.
        properties (dict[str, dict[str, str]]): A dictionary defining the properties of the parameters. Each key is a parameter name, and the value is another dictionary that defines the attributes of the parameter.
    """

    type: str = "object"
    required: list[str]
    properties: dict[str, dict[str, str]]


def convert_tool_schema_to_openai_tool(tool_schema: Tool) -> ChatCompletionToolParam:
    """
    Converts a ToolSchema object to a ChatCompletionToolParam object for OpenAI.

    Args:
        tool_schema (ToolSchema): The schema of the tool to be converted.

    Returns:
        ChatCompletionToolParam: The converted tool schema in the format required by OpenAI.
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

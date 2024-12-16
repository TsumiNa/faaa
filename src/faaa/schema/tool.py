# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

from typing import List

from pydantic import BaseModel


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
        tags (List[str]): A list of tags describe the catalog of the tool.
        parameters (List[ParameterSchema]): A list of parameters associated with the tool.
    """

    name: str
    description: str
    tags: List[str]
    parameters: List[ParameterSchema]


def convert_tool_schema_to_openai_tool(tool_schema: ToolSchema) -> dict:
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

# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

from typing import Callable, List

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


class ToolMetaSchema(BaseModel):
    """
    ToolMetaSchema represents the schema for a tool with its name, description, and parameters.

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


class ToolSchema(BaseModel):
    """
    A schema class for representing an agent with various attributes.

    Attributes:
        func (Callable): The function associated with the agent.
        code_id (str): The unique identifier for the code.
        tool (ToolMetaSchema): The tool metadata associated with the agent.
    """

    func: Callable
    code_id: str
    tool: ToolMetaSchema

# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

# %%

import yaml
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


class Tool(BaseModel):
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
    tags: list[str]
    parameters: list[ParameterSchema]


def pydantic_to_yaml(pydantic_obj: BaseModel) -> str:
    """
    Converts a Pydantic object to a YAML-formatted string without brackets or quotes.

    Args:
        pydantic_obj (BaseModel): The Pydantic object to convert.

    Returns:
        str: A YAML-formatted string representing the Pydantic object.
    """
    if not isinstance(pydantic_obj, BaseModel):
        raise ValueError("Input must be a Pydantic BaseModel object.")

    # Convert the Pydantic object to a dictionary
    data = pydantic_obj.model_dump()

    # Convert the dictionary to a YAML-formatted string
    yaml_output = yaml.dump(data, sort_keys=False, default_flow_style=False)

    return yaml_output


# Create an example instance
tool = Tool(name="example_tool", description="A sample tool", tags=["sample", "demo"], parameters=[])

query = "Hello, world!"
query = f"""
<Query>{query}</Query>
{'\n'.join(['<Tool>\n'+pydantic_to_yaml(s)+'</Tool>' for s in [tool, tool]])}
"""

print(query)
# %%

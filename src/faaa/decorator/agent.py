# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import inspect
from typing import Any, Callable

from faaa.function.llm import LLMClient
from faaa.schema import ToolSchema
from faaa.util import generate_id


class Agent:
    _instance = None
    _tool_schemas: dict[str, ToolSchema] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Agent, cls).__new__(cls)
            cls._tool_schemas = {}
        return cls._instance

    def __init__(self, **kwargs):
        if not hasattr(self, "_initialized"):
            self._tool_schemas = {}
            self._initialized = True
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"<Agent instance with schemas: {self._tool_schemas}>"

    def __getitem__(self, tool_name):
        return self._tool_schemas.get(tool_name)

    def tool(
        self,
        prefix_path: str | None = None,
        tags: list[str] | None = None,
        dependencies: list[Callable] | None = None,
        responses: dict[int, dict[str, Any]] | None = None,
    ):
        def _ret(func: Callable):
            # Extract the information of the function
            _ = inspect.getsource(func).strip()
            code_id = generate_id(_)
            tool_schema = LLMClient().parse(func)

            # Set API schema
            return func

        return _ret


# Example usage:
# singleton = Agent()
# singleton.set_tool_schema("tool1", generate_tool_schema("tool1"))
# schema = singleton.get_tool_schema("tool1")

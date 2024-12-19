# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import inspect
import os
from typing import Callable

from pydantic import BaseModel

from faaa.core.llm import LLMClient
from faaa.core.tool_schema import Tool
from faaa.util import generate_id


class AgentError(Exception):
    def __init__(self, message: str | BaseException | None = None):
        self.message = f"Agent error: {message}"
        super().__init__(self.message)


class _AgentSchema(BaseModel):
    """
    _AgentSchema is a schema class that inherits from ToolSchema.

    Attributes:
        func (Callable): A callable function associated with the agent.
        entry_points (str): Entry points for the agent.
        code_id (str): Identifier for the code associated with the agent.
    """

    func: Callable
    entry_points: str
    code_id: str
    tool: Tool


class Agent:
    _llm_client = LLMClient()

    def __init__(self, **kwargs):
        self._agents: dict[str, _AgentSchema] = {}
        self._prefix_path = "/agent/v1"
        pass

    def __repr__(self):
        return f"<Agent instance with schemas: {self._agents}>"

    @property
    def agents(self):
        return self._agents

    @classmethod
    def get_function_file_name(cls, func: Callable) -> str:
        try:
            # Get the file path where the function is located
            file_path = inspect.getfile(func)
            # Check if the file path actually exists
            if os.path.exists(file_path):
                return "/" + os.path.splitext(os.path.basename(file_path))[0]
            else:
                # Return "/" if the path does not exist
                return "/"

        except TypeError:  # Built-in functions may trigger TypeError
            # Return "/" for built-in or unknown functions
            return "/"

    def _register_tool_schema(self, func: Callable, code_id: str, tool_schema: Tool, entry_points: str):
        self._agents[code_id] = _AgentSchema(
            func=func, entry_points=entry_points, code_id=code_id, tool=tool_schema
        )

    def tool(
        self,
    ):
        async def _decorator(func: Callable):
            # decorated function
            async def _func(*args, **kwargs):
                # some logic here before calling the function
                # ...

                # Call the function
                return await func(*args, **kwargs)

            # Extract the information of the function
            _ = inspect.getsource(func).strip()
            code_id = generate_id(_)  # Generate a unique ID for the function
            tool_schema = await self._llm_client.generate_tool_description(func)
            file_name = self.get_function_file_name(func)
            prefix_path = f"{self._prefix_path}/{file_name}"
            entry_points = f"{prefix_path}/{tool_schema.name}"
            self._register_tool_schema(
                _func,
                code_id,
                tool_schema,
                entry_points=entry_points,
            )

            return _func

        return _decorator

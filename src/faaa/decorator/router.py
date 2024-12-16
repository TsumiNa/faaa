# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import inspect
import os
from typing import Callable

from fastapi import APIRouter

from faaa.schema.tool import ToolSchema


class AutoRouter:
    def __init__(self, prefix_path: str | None = None):
        self._router = APIRouter()
        self._prefix_path = prefix_path or "/agents"

    @property
    def router(self):
        return self._router

    @classmethod
    def get_function_file_name(cls, func: Callable) -> str:
        try:
            # Get the file path where the function is located
            file_path = inspect.getfile(func)
            # Check if the file path actually exists
            if os.path.exists(file_path):
                return os.path.splitext(os.path.basename(file_path))[0]
            else:
                # Return "/" if the path does not exist
                return "/"

        except TypeError:  # Built-in functions may trigger TypeError
            # Return "/" for built-in or unknown functions
            return "/"

    def __call__(self, func: Callable, tool_schema: ToolSchema):
        file_name = self.get_function_file_name(func)

        @self._router.get(f"/{self._prefix_path}/{file_name}/{func.__name__}", response_model=tool_schema)
        async def _get():
            return tool_schema

        @self._router.post(f"/{self._prefix_path}/{file_name}/{func.__name__}", response_model=str)
        async def _post(q: str | None = None):
            return func

        return func

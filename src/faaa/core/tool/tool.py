# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import asyncio
import inspect
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, Coroutine

from faaa.core.tool.schema import ToolSchema
from faaa.provider import OpenAIClient
from faaa.util import generate_id


class Tool:
    def __init__(self, **kwargs):
        self._llm_client = None
        self._tools: dict[str, ToolSchema] = {}
        self._thread_pool_executor: ThreadPoolExecutor | None = None
        self._process_pool_executor: ProcessPoolExecutor | None = None
        self._registration_tasks: list[Coroutine[Any, Any, ToolSchema | None]] = []

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = OpenAIClient()
        return self._llm_client

    async def _init_tools(self) -> dict[str, ToolSchema] | None:
        if not self._registration_tasks:
            return None
        self._tools = {t.code_id: t for t in await asyncio.gather(*self._registration_tasks) if t is not None}

        self._registration_tasks.clear()  # Clear tasks after execution
        return self._tools

    def __repr__(self):
        return f"<Agent instance with schemas: {self._tools}>"

    @property
    def tools(self):
        return self._tools

    def update_config(self, **kwargs):
        """Update agent configuration with provided kwargs."""
        pass

    @classmethod
    def _get_source_code(self, func: Callable) -> str:
        """Get source code of a function. This is a separate function to be picklable."""
        return inspect.getsource(func).strip()

    @classmethod
    def _get_function_file_name(cls, func: Callable) -> str:
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

    async def _func_register(self, original_func: Callable, wrapped_func: Callable) -> ToolSchema | None:
        """
        Register a function as a tool.

        Args:
            original_func: The original function (used for metadata)
            wrapped_func: The wrapped function (used for execution)
        """
        if not callable(original_func) or not callable(wrapped_func):
            raise ValueError("Both original_func and wrapped_func must be callable")

        _ = inspect.getsource(original_func).strip()
        code_id = generate_id(_)

        # Skip if already registered
        if code_id in self._tools:
            return

        # Generate tool schema in thread pool (I/O-bound)
        if self._thread_pool_executor is None:
            raise ValueError("ThreadPoolExecutor not initialized.")

        tool_schema = await self.llm_client.tool_description(original_func)

        return ToolSchema(
            func=wrapped_func,  # Store wrapped function for execution
            code_id=code_id,
            tool=tool_schema,
        )

    def add(self, *, use_process=False):
        def decorator(func):
            if not callable(func):
                raise ValueError("The provided func must be a callable")

            if asyncio.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)

                wrapped = async_wrapper
            else:

                @wraps(func)
                async def sync_wrapper(*args, **kwargs):
                    loop = asyncio.get_running_loop()
                    executor = self._process_pool_executor if use_process else self._thread_pool_executor

                    if executor is None:
                        raise ValueError(
                            f"{(ProcessPoolExecutor if use_process else ThreadPoolExecutor).__name__} not initialized."
                        )
                    return await loop.run_in_executor(executor, func, *args, **kwargs)

                wrapped = sync_wrapper

            # Add registration task with both original and wrapped functions
            self._registration_tasks.append(self._func_register(func, wrapped))
            return wrapped

        return decorator

# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

# %%
from http import client

from faaa.decorator import agent
from faaa.decorator.router import AutoRouter

auto_router = AutoRouter()
relative_path = auto_router.get_relative_path(agent.Agent)
print(relative_path)

# %%
from faaa.decorator import agent
from faaa.function.llm import LLMClient

client = LLMClient()

await client.chat("Hello, world!")

# %%

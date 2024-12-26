# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

# agent_package/agent.py

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from faaa.core.agent.schema import DynamicPlanContainer, DynamicPlanTracer
from faaa.core.prompt import DYNAMIC_PLAN_INSTRUCTION
from faaa.core.tool import Tool, ToolSchema
from faaa.provider import OpenAIClient
from faaa.util import generate_id, pydantic_to_yaml


class GeneratePlanRequest(BaseModel):
    task: str


class GeneratePlanResponse(BaseModel):
    status: int
    plan: list[DynamicPlanTracer]


class Agent:
    def __init__(
        self,
        *,
        max_thread_workers: int | None = None,
        fast_api: Optional[FastAPI] = None,
        config: Optional[Dict] = None,
    ):
        """
        初始化 Agent。

        :param fast_api: 用户创建的 FastAPI 实例。如果为 None，则 Agent 可以独立使用。
        :param config: Agent 的配置字典。
        """
        self.config = config or {}
        self.fast_api = fast_api

        # 设置日志
        self.logger = logger

        if self.fast_api:
            self._integrate_with_fastapi()

        self._tools: dict[str, ToolSchema] = {}  # Stores registered tools
        self._tool_list: list[Tool] = []  # Stores agents pending registration
        self._llm_client = OpenAIClient()

        self._thread_executor = (
            ThreadPoolExecutor(max_workers=max_thread_workers) if max_thread_workers else ThreadPoolExecutor()
        )
        cpus = os.cpu_count() or 1
        self._process_executor = ProcessPoolExecutor(cpus - 1) if cpus >= 2 else ProcessPoolExecutor()

    def _integrate_with_fastapi(self):
        """
        将 Agent 集成到传入的 FastAPI 实例中，注册路由和生命周期事件。
        """
        # 注册生命周期事件
        self.fast_api.add_event_handler("startup", self.start)
        self.fast_api.add_event_handler("shutdown", self.stop)

        # 注册路由
        self.fast_api.post("/agent/v1/generate_plan")(self.generate_plan_route)
        self.fast_api.get("/agent/v1/status")(self.status_route)

        # 注册异常处理
        self.fast_api.add_exception_handler(Exception, self.exception_handler)

    def include_tools(self, *tool: Tool, **kwargs):
        """
        Include agents for later registration.
        Updates agent configs and stores them for processing during register_agent.

        Args:
            *agent: Agent instances to include
            **kwargs: Configuration parameters to update agents with
        """
        for t in tool:
            t.update_config(**kwargs)
            self._tool_list.append(t)

    async def _init_agents(self):
        """
        Register all tools from agents in _agent_list.
        Merges registered tools into self._agents and clears _agent_list.
        """
        # Register tools for all agents
        for tool in self._tool_list:
            tool._thread_pool_executor = self._thread_executor
            tool._process_pool_executor = self._process_executor
            # Merge agent's tools into self._agents
            self._tools.update(await tool._init_tools())

        # Clear agent list after registration
        self._tool_list.clear()
        return self

    async def _save_agent_state(self):
        """
        Save the current state of the agent.
        """
        pass

    async def start(self):
        """
        启动 Agent，初始化 AgentCore 并注册代理。
        """
        self.logger.info("Starting up Agent...")

        await self._init_agents()
        self.logger.info("Agent is ready.")

    async def stop(self):
        """
        停止 Agent，清理 AgentCore 的资源。
        """
        self.logger.info("Shutting down Agent...")
        self._thread_executor.shutdown(wait=True)
        self._process_executor.shutdown(wait=True)
        self.logger.info("Agent has been shut down.")

    async def generate_plan_route(self, input_data: GeneratePlanRequest):
        """
        处理 /api/v1/generate_plan 路由的请求。
        """
        self.logger.info(f"Received generate_plan request with data: {input_data.dict()}")
        try:
            result = await self.generate_plan(input_data.task)
            return GeneratePlanResponse(status=200, plan=result)
        except Exception as e:
            self.logger.error(f"Error in generate_plan: {e}")
            raise e

    async def generate_plan(self, query: str) -> list[DynamicPlanTracer]:
        if not self._tools:
            id = generate_id("No agents available")
            return DynamicPlanTracer(
                id=id,
                description="No agents available",
                steps=[],
                recommendation_tools=[],
                recommendation_score=0.0,
            )

        query = f"""
<Query>\n{query}\n</Query>
{'\n'.join(['<Tool>\n'+pydantic_to_yaml(s.tool)+'</Tool>' for s in self._tools.values()])}
""".strip()
        query = [{"role": "system", "content": DYNAMIC_PLAN_INSTRUCTION}, {"role": "user", "content": query}]
        DPs = await self._llm_client.structured_output(
            query,
            structured_outputs=DynamicPlanContainer,
            max_try=1,
            max_tokens=1000,
            model="openai/gpt-4o-2024-11-20",
            # model="openai/o1-preview",
        )

        return [
            DynamicPlanTracer(id=generate_id(plan.description), **plan.model_dump()) for plan in DPs.plans
        ]

    async def status_route(self):
        """
        处理 /api/v1/status 路由的请求。
        """
        self.logger.info("Received status request.")
        return {"status": "Agent is running"}

    async def exception_handler(self, request: Request, exc: Exception):
        """
        全局异常处理器。
        """
        self.logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"},
        )

    @asynccontextmanager
    async def run(self):
        await self.start()
        yield self
        await self.stop()

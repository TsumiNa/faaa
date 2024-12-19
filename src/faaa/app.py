# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT


from fastapi import FastAPI
from pydantic import BaseModel

from faaa.core.llm import LLMClient
from faaa.core.tool_schema import ToolParameter
from faaa.decorator.agent import Agent, _AgentSchema
from faaa.prompt import DYNAMIC_PLAN_INSTRUCTION
from faaa.util import pydantic_to_yaml


class FAError(Exception):
    def __init__(self, message: str | BaseException | None = None):
        self.message = f"FA error: {message}"
        super().__init__(self.message)


class PlanStep(BaseModel):
    """
    PlanStep represents a step in a plan with detailed information.

    Attributes:
        description (str): A short description of the step.
        suggested_tool (str): The tool suggested for this step.
        sub_query (str): The sub query for this step.
        explanation (str): Explanation of why this step is needed.
        retry (int): The number of retries for this step.
    """

    description: str  # A short description of the step
    suggested_tool: str  # The tool suggested for this step
    sub_query: str  # The sub query for this step
    explanation: str  # Explanation of why this step is needed
    retry: int  # The number of retries for this step


class RecommendationTool(BaseModel):
    """
    A model representing a recommended tool.

    Attributes:
        name (str): The name of the recommended tool.
        description (str): A brief description of the recommended tool.
        parameters (list[ParameterSchema]): A list of parameters associated with the recommended tool.
    """

    name: str
    description: str
    parameters: list[ToolParameter]


class DynamicPlan(BaseModel):
    """
    DynamicPlan represents a dynamic plan with a unique identifier, description, steps,
    recommended tools, and a recommendation score.

    Attributes:
        id (str): A unique identifier for the plan.
        description (str): A description of the overall plan.
        steps (list[PlanStep] | None): A list of steps in the plan.
        recommendation_tool (list[RecommendedTool] | None): The recommended tool for the plan.
        recommendation_score (float): The recommendation score for the plan.
    """

    id: str  # A unique identifier for the plan
    description: str  # A description of the overall plan
    steps: list[PlanStep] | None  # A list of steps in the plan
    recommendation_tool: list[RecommendationTool] | None  # The recommended tool for the plan
    recommendation_score: float  # The recommendation score for the plan


class FA:
    def __init__(self):
        self._agents: dict[str, _AgentSchema] = {}
        self._fastapi_app = FastAPI()
        self._llm_client = LLMClient()

    def register_agent(self, agent: Agent):
        self._agents.update(agent.agents)

    async def _generate_plan(self, query: str):
        query = f"""
        <Query>{query}</Query>
        {'\n'.join(['<Tool>\n'+pydantic_to_yaml(s.tool)+'</Tool>' for s in self._agents.values()])}
        """
        query = [{"role": "system", "content": DYNAMIC_PLAN_INSTRUCTION}, {"role": "user", "content": query}]
        return await self._llm_client.parse(query, structured_outputs=DynamicPlan)

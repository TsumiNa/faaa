# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel

from faaa.core.tool import ToolParameter


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
    RecommendationTool is a model representing a recommendation tool with its associated attributes.

    Attributes:
        name (str): The name of the recommendation tool.
        description (str): A brief description of the recommendation tool.
        reason (str): The reason or purpose for the recommendation tool.
        parameters (list[ToolParameter]): A list of parameters associated with the recommendation tool.
    """

    name: str
    description: str
    reason: str
    parameters: list[ToolParameter]


class DynamicPlan(BaseModel):
    """
    DynamicPlan represents a dynamic plan with a description, steps, recommendation tools, and a recommendation score.

    Attributes:
        description (str): A description of the overall plan.
        steps (list[PlanStep]): A list of steps in the plan.
        recommendation_tools (list[RecommendationTool]): The recommended tools for the plan.
        recommendation_score (float): The recommendation score for the plan.
    """

    description: str  # A description of the overall plan
    steps: list[PlanStep]  # A list of steps in the plan
    recommendation_tools: list[RecommendationTool]  # The recommended tool for the plan
    recommendation_score: float  # The recommendation score for the plan


class DynamicPlanContainer(BaseModel):
    """
    A container for dynamic plans.

    Attributes:
        plans (list[DynamicPlan]): A list of dynamic plans.
    """

    plans: list[DynamicPlan]


class DynamicPlanTracer(DynamicPlan):
    """
    DynamicPlanTracer represents a dynamic plan with a unique identifier, description, steps,
    recommended tools, and a recommendation score. It also includes a trace of the plan generation process.

    Attributes:
        id (str): A unique identifier for the plan.
        n_execution (int): The number of executions for the plan.
        parent_id (str): The parent plan's identifier.
    """

    id: str  # A unique identifier for the plan
    n_execution: int = 0  # The number of executions for the plan
    parent_id: str | None = None  # The parent plan's identifier

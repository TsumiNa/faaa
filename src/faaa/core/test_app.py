from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI

from faaa.core.app import DynamicPlan, DynamicPlanTracer, FaaA
from faaa.core.tool_schema import ToolParameter, ToolSchema
from faaa.decorator.agent import Agent


@pytest.fixture
def mock_agent():
    agent = Mock(spec=Agent)
    agent.update_config = Mock()
    agent.register_tools = AsyncMock(return_value={"test_tool": Mock()})
    return agent


@pytest.mark.asyncio
async def test_init():
    app = FaaA()
    assert isinstance(app._fastapi_app, FastAPI)
    assert len(app._agents) == 0
    assert len(app._agent_list) == 0


@pytest.mark.asyncio
async def test_include_agents(mock_agent):
    app = FaaA()
    app.include_agents(mock_agent, test_param="test_value")

    assert len(app._agent_list) == 1
    mock_agent.update_config.assert_called_once_with(test_param="test_value")


@pytest.mark.asyncio
async def test_register_agents(mock_agent):
    app = FaaA()
    app.include_agents(mock_agent)

    await app.register_agents()

    assert len(app._agents) == 1
    assert len(app._agent_list) == 0
    mock_agent.register_tools.assert_called_once()


@pytest.mark.asyncio
async def test_generate_plan_no_agents():
    app = FaaA()
    plan = await app.generate_plan("test query")

    assert isinstance(plan, DynamicPlanTracer)
    assert plan.description == "No agents available"
    assert plan.steps is None
    assert plan.recommendation_tools is None
    assert plan.recommendation_score == 0.0


@pytest.mark.asyncio
async def test_generate_plan_with_agents():
    app = FaaA()
    mock_tool = ToolSchema(
        name="test_tool",
        description="test description",
        tags=["test"],
        parameters=[ToolParameter(name="param1", type="string", description="test param", required=True)],
    )
    app._agents = {"test_tool": Mock(tool=mock_tool)}

    expected_plan = DynamicPlan(
        description="Test plan", steps=[], recommendation_tools=[], recommendation_score=0.8
    )

    with patch.object(app._llm_client, "structured_output", new=AsyncMock(return_value=expected_plan)):
        with patch("faaa.app.generate_id", return_value="test_id"):
            plan = await app.generate_plan("test query")

            assert isinstance(plan, DynamicPlanTracer)
            assert plan.id == "test_id"
            assert plan.description == expected_plan.description
            assert plan.steps == expected_plan.steps
            assert plan.recommendation_tools == expected_plan.recommendation_tools
            assert plan.recommendation_score == expected_plan.recommendation_score


@pytest.mark.asyncio
async def test_context_manager():
    async with FaaA() as app:
        assert isinstance(app, FaaA)

    # Verify executors are shut down
    assert app._thread_executor._shutdown  # ThreadPoolExecutor uses _shutdown internally
    # For ProcessPoolExecutor, verify it's been shutdown by checking if we can submit new tasks
    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        app._process_executor.submit(lambda: None)


@pytest.mark.asyncio
async def test_repr_no_agents():
    app = FaaA()
    assert repr(app) == "FA(agents={})"


@pytest.mark.asyncio
async def test_repr_with_agents():
    app = FaaA()
    mock_tool = ToolSchema(
        name="test_tool",
        description="test description",
        tags=["test"],
        parameters=[ToolParameter(name="param1", type="string", description="test param", required=True)],
    )
    app._agents = {"test_tool": Mock(tool=mock_tool)}

    repr_str = repr(app)
    assert repr_str.startswith("FA(agents={")
    assert "test_tool" in repr_str

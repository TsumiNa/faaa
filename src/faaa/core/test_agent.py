import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from faaa.core.tool_schema import ToolParameter, ToolSchema
from faaa.decorator.agent import Agent, AgentError, _AgentToolSchema


# Test fixtures
@pytest_asyncio.fixture
async def agent():
    agent = Agent(prefix_path="/test/v1")
    agent._thread_pool_executor = ThreadPoolExecutor()
    agent._process_pool_executor = ProcessPoolExecutor()
    yield agent
    # Clean up executors
    if agent._thread_pool_executor:
        agent._thread_pool_executor.shutdown(wait=True)
    if agent._process_pool_executor:
        agent._process_pool_executor.shutdown(wait=True)


@pytest.fixture
def mock_tool_schema():
    return ToolSchema(
        name="test_function",
        description="A test function",
        tags=["test"],
        parameters=[ToolParameter(name="param1", type="string", description="Test parameter", required=True)],
    )


# Mock async function for testing
async def async_test_func(param1: str):
    return f"Async result: {param1}"


# Mock sync function for testing
def sync_test_func(param1: str):
    return f"Sync result: {param1}"


# Tests
@pytest.mark.asyncio
async def test_agent_initialization():
    agent = Agent()
    assert agent._prefix_path == "/agent/v1"
    assert isinstance(agent._tools, dict)
    assert agent._thread_pool_executor is None
    assert agent._process_pool_executor is None


@pytest.mark.asyncio
async def test_register_async_function(agent, mock_tool_schema):
    with patch.object(agent, "_llm_client") as mock_llm:
        # Setup mock
        mock_llm.generate_tool_description = AsyncMock(return_value=mock_tool_schema)

        # Register async function
        decorated_func = agent.register()(async_test_func)
        assert asyncio.iscoroutinefunction(decorated_func)

        # Register tools
        await agent.register_tools()

        # Verify registration
        assert len(agent._tools) == 1
        registered_tool = list(agent._tools.values())[0]
        assert isinstance(registered_tool, _AgentToolSchema)
        assert registered_tool.tool == mock_tool_schema
        assert registered_tool.entry_points.startswith("/test/v1")


@pytest.mark.asyncio
async def test_register_sync_function(agent, mock_tool_schema):
    with patch.object(agent, "_llm_client") as mock_llm:
        # Setup mock
        mock_llm.generate_tool_description = AsyncMock(return_value=mock_tool_schema)

        # Register sync function with thread executor
        decorated_func = agent.register()(sync_test_func)
        assert asyncio.iscoroutinefunction(decorated_func)

        # Register tools
        await agent.register_tools()

        # Verify registration
        assert len(agent._tools) == 1
        registered_tool = list(agent._tools.values())[0]
        assert isinstance(registered_tool, _AgentToolSchema)
        assert registered_tool.tool == mock_tool_schema


@pytest.mark.asyncio
async def test_register_sync_function_with_process_pool(agent, mock_tool_schema):
    with patch.object(agent, "_llm_client") as mock_llm:
        # Setup mock
        mock_llm.generate_tool_description = AsyncMock(return_value=mock_tool_schema)

        # Register sync function with process executor
        decorated_func = agent.register(use_process=True)(sync_test_func)
        assert asyncio.iscoroutinefunction(decorated_func)

        # Register tools
        await agent.register_tools()

        # Verify registration
        assert len(agent._tools) == 1
        registered_tool = list(agent._tools.values())[0]
        assert isinstance(registered_tool, _AgentToolSchema)
        assert registered_tool.tool == mock_tool_schema


@pytest.mark.asyncio
async def test_executor_not_initialized_error(agent):
    # Remove executors
    agent._thread_pool_executor = None
    agent._process_pool_executor = None

    # Test thread pool executor error
    decorated_func = agent.register()(sync_test_func)
    with pytest.raises(ValueError, match="ThreadPoolExecutor not initialized"):
        await decorated_func("test")

    # Test process pool executor error
    decorated_func = agent.register(use_process=True)(sync_test_func)
    with pytest.raises(ValueError, match="ProcessPoolExecutor not initialized"):
        await decorated_func("test")


@pytest.mark.asyncio
async def test_get_function_file_name():
    # Test with regular function
    assert Agent.get_function_file_name(sync_test_func) == "test_agent"

    # Test with built-in function
    assert Agent.get_function_file_name(len) == "/"


@pytest.mark.asyncio
async def test_duplicate_registration(agent, mock_tool_schema):
    with patch.object(agent, "_llm_client") as mock_llm:
        # Setup mock
        mock_llm.generate_tool_description = AsyncMock(return_value=mock_tool_schema)

        # Register same function twice
        agent.register()(sync_test_func)
        agent.register()(sync_test_func)
        await agent.register_tools()

        # Should only be registered once
        assert len(agent._tools) == 1


@pytest.mark.asyncio
async def test_actual_function_execution(agent, mock_tool_schema):
    # Test async function execution
    with patch.object(agent, "_llm_client") as mock_llm:
        mock_llm.generate_tool_description = AsyncMock(return_value=mock_tool_schema)
        async_decorated = agent.register()(async_test_func)
        await agent.register_tools()
        result = await async_decorated("test_param")
        assert result == "Async result: test_param"

    # Clear registrations and prepare for next test
    agent._registration_tasks.clear()
    agent._tools.clear()

    # Test sync function execution with thread pool
    with patch.object(agent, "_llm_client") as mock_llm:
        mock_llm.generate_tool_description = AsyncMock(return_value=mock_tool_schema)
        sync_decorated = agent.register()(sync_test_func)
        await agent.register_tools()
        result = await sync_decorated("test_param")
        assert result == "Sync result: test_param"

    # Clear registrations and prepare for next test
    agent._registration_tasks.clear()
    agent._tools.clear()

    # Test sync function execution with process pool
    with patch.object(agent, "_llm_client") as mock_llm:
        mock_llm.generate_tool_description = AsyncMock(return_value=mock_tool_schema)
        sync_process_decorated = agent.register(use_process=True)(sync_test_func)
        await agent.register_tools()
        result = await sync_process_decorated("test_param")
        assert result == "Sync result: test_param"


@pytest.mark.asyncio
async def test_llm_error_handling(agent):
    with patch.object(agent, "_llm_client") as mock_llm:
        # Setup mock to raise an exception
        mock_llm.generate_tool_description = AsyncMock(side_effect=Exception("LLM Error"))

        # Register function and attempt to register tools
        agent.register()(sync_test_func)
        with pytest.raises(Exception, match="LLM Error"):
            await agent.register_tools()


@pytest.mark.asyncio
async def test_tools_property(agent, mock_tool_schema):
    with patch.object(agent, "_llm_client") as mock_llm:
        mock_llm.generate_tool_description = AsyncMock(return_value=mock_tool_schema)

        # Register a function
        agent.register()(sync_test_func)
        await agent.register_tools()

        # Test tools property
        tools = agent.tools
        assert isinstance(tools, dict)
        assert len(tools) == 1
        assert isinstance(list(tools.values())[0], _AgentToolSchema)


def test_agent_repr(agent):
    # Test empty agent repr
    assert repr(agent) == "<Agent instance with schemas: {}>"

    # Add a mock agent schema
    mock_schema = _AgentToolSchema(
        func=sync_test_func,
        entry_points="/test/v1/test",
        code_id="test_id",
        tool=ToolSchema(name="test", description="test", tags=["test"], parameters=[]),
    )
    agent._tools["test_id"] = mock_schema

    # Test repr with schema
    assert repr(agent) == "<Agent instance with schemas: {'test_id': " + repr(mock_schema) + "}>"


def test_agent_error():
    error_msg = "Test error message"
    error = AgentError(error_msg)
    assert str(error) == f"Agent error: {error_msg}"

    # Test with exception as message
    base_error = ValueError("Base error")
    error = AgentError(base_error)
    assert str(error) == "Agent error: Base error"


@pytest.mark.asyncio
async def test_update_config(agent):
    # Test updating config with new values
    agent.update_config(new_param="test_value")
    # Currently update_config is a no-op, so no assertions needed
    # This test ensures the method exists and can be called without errors


@pytest.mark.asyncio
async def test_invalid_function_registration():
    agent = Agent()

    # Try to register a non-callable
    with pytest.raises(ValueError, match="The provided func must be a callable"):
        not_callable = "not a function"
        agent.register()(not_callable)


@pytest.mark.asyncio
async def test_executor_cleanup(agent):
    # Verify executors are initialized
    assert agent._thread_pool_executor is not None
    assert agent._process_pool_executor is not None

    # Shutdown executors
    agent._thread_pool_executor.shutdown()
    agent._process_pool_executor.shutdown()

    # Try to use executors after shutdown
    decorated_func = agent.register()(sync_test_func)
    with pytest.raises(RuntimeError):
        await decorated_func("test")

    decorated_func = agent.register(use_process=True)(sync_test_func)
    with pytest.raises(RuntimeError):
        await decorated_func("test")

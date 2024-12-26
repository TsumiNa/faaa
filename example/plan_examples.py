"""使用FaaA的generate_plan功能的示例。

运行此示例前需要：
1. 在项目根目录创建.env文件，包含：
   OPENAI_API_KEY="your-api-key"
   OPENAI_BASE_URL="https://openrouter.ai/api/v1"

2. 将"your-api-key"替换为你的实际OpenAI API密钥
"""

from agent_functions import calculate_fibonacci, fetch_delayed_greeting, prime_factors
from fastapi import FastAPI

from faaa import Agent
from faaa.core import Tool

# 用户自定义的 FastAPI 实例
app = FastAPI(
    title="Custom Agent API",
    description="这是一个自定义的 FastAPI 应用，集成了 Agent 功能。",
    version="1.0.0",
)


# 用户可以在这里添加额外的路由、依赖、中间件等
@app.get("/custom_route")
async def custom_route():
    return {"message": "这是一个用户自定义的路由！"}


# 创建并配置agents
tool = Tool()

tool.add(use_process=False)(calculate_fibonacci)
tool.add(use_process=True)(prime_factors)
tool.add()(fetch_delayed_greeting)


# 初始化 Agent，并将 FastAPI 实例传递给它
agent = Agent(fast_api=app, config={"key": "value"})
agent.include_tools(tool)

# 初始化FaaA
# async with agent.run() as a:
#     plan = await a.generate_plan("我需要计算斐波那契数列中的第10个数字。")
#     a.logger.info(plan)

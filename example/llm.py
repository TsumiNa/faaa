import asyncio

from pydantic import BaseModel

from faaa.core.llm import LLMClient
from faaa.core.tool_schema import Tool


class ExampleOutput(BaseModel):
    summary: str
    details: str


async def main():
    llm_client = LLMClient()

    # messages = [
    #     {"role": "user", "content": "Can you summarize the following text?"},
    #     {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
    # ]

    # try:
    #     result = await llm_client.parse(
    #         *messages, structured_outputs=ExampleOutput, model="openai/gpt-4o-mini", max_token=200, max_try=3
    #     )
    #     print(result)
    # except Exception as e:
    #     print(f"Error: {e}")

    # Define a simple tool schema
    tool_schemas = [Tool(name="example_tool", description="This is an example tool.", tags=[], parameters=[])]

    # Example message
    message = "Use the example tool to demonstrate its functionality."

    try:
        tool_calls = await llm_client.function_call(msg=message, tool_schemas=tool_schemas, max_try=1)
        print("Tool Calls:", tool_calls)
    except Exception as e:
        print("An error occurred:", e)
        raise e


if __name__ == "__main__":
    asyncio.run(main())

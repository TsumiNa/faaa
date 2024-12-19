import asyncio

from faaa.core.tool_schema import Tool, ToolParameter, convert_tool_schema_to_openai_tool


async def example_generate_tool_schema():
    async def sample_function(x: int, y: str) -> bool:
        """
        This is a sample function that takes an integer and a string and returns a boolean.

        Args:
            x (int): An integer value.
            y (str): A string value.

        Returns:
            bool: True if x is greater than 10 and y is not empty, False otherwise.
        """
        return x > 10 and bool(y)

    # llm_client = LLMClient()
    # tool_schema = await llm_client.generate_tool_description(sample_function)
    # print(tool_schema)

    # Define a sample tool schema
    sample_tool = Tool(
        name="example_tool",
        description="This is an example tool.",
        tags=["example", "test"],
        parameters=[
            ToolParameter(name="param1", type="string", description="The first parameter", required=True),
            ToolParameter(name="param2", type="integer", description="The second parameter", required=False),
        ],
    )

    # Convert the tool schema to OpenAI tool format
    openai_tool = convert_tool_schema_to_openai_tool(sample_tool)

    # Print the converted tool schema
    print(openai_tool)


if __name__ == "__main__":
    asyncio.run(example_generate_tool_schema())

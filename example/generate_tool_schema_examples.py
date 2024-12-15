from faaa.schema import generate_tool_schema


def example_add_function():
    def add(x: int, y: int) -> int:
        """Add two numbers together"""
        return x + y

    schema = generate_tool_schema(add)
    print(schema.model_dump_json(indent=2))


def example_process_data_function():
    def process_data(data: list, threshold: float = 0.5, debug: bool = False) -> dict:
        """
        Process a list of data values using the given threshold.

        Args:
            data: List of values to process
            threshold: Cutoff threshold value
            debug: Enable debug output
        """
        return {}

    schema = generate_tool_schema(process_data)
    print(schema.model_dump_json(indent=2))


def example_minimal_function():
    def minimal_func(a, b):
        return a + b

    schema = generate_tool_schema(minimal_func)
    print(schema.model_dump_json(indent=2))


if __name__ == "__main__":
    example_add_function()
    example_process_data_function()
    example_minimal_function()

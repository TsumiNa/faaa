from .schema import ToolSchema, generate_tool_schema


def test_generate_tool_schema_basic():
    def add(x: int, y: int) -> int:
        """Add two numbers together"""
        return x + y

    schema = generate_tool_schema(add)

    assert isinstance(schema, ToolSchema)
    assert schema.name == "add"
    assert "Add two numbers" in schema.description
    assert len(schema.parameters) == 2

    param_names = [p.name for p in schema.parameters]
    assert "x" in param_names
    assert "y" in param_names

    assert "add" in schema.model_dump_json()


def test_generate_tool_schema_with_complex_function():
    def process_data(data: list, threshold: float = 0.5, debug: bool = False) -> dict:
        """
        Process a list of data values using the given threshold.

        Args:
            data: List of values to process
            threshold: Cutoff threshold value
            debug: Enable debug output
        """
        result = {}
        for item in data:
            if item > threshold and debug:
                result[str(item)] = True
            else:
                result[str(item)] = False
        return result

    schema = generate_tool_schema(process_data)

    assert isinstance(schema, ToolSchema)
    assert schema.name == "process_data"
    assert "Process a list of data values" in schema.description
    assert len(schema.parameters) == 3

    param_names = [p.name for p in schema.parameters]
    assert all(name in param_names for name in ["data", "threshold", "debug"])

    param_types = [p.type for p in schema.parameters]
    assert "list" in param_types
    assert "float" in param_types
    assert "bool" in param_types

    assert "process_data" in schema.model_dump_json()


def test_generate_tool_schema_no_docstring():
    def minimal_func(a, b):
        return a + b

    schema = generate_tool_schema(minimal_func)

    assert isinstance(schema, ToolSchema)
    assert schema.name == "minimal_func"
    assert len(schema.parameters) == 2

    assert "minimal_func" in schema.model_dump_json()


def test_generate_tool_schema_empty_docstring():
    def func(a: str) -> str:
        return a.upper()

    schema = generate_tool_schema(func)
    assert isinstance(schema, ToolSchema)
    assert schema.name == "func"
    assert "uppercase" in schema.description


def test_generate_tool_schema_no_params():
    def get_constant() -> int:
        """Returns a constant value"""
        return 42

    schema = generate_tool_schema(get_constant)
    assert isinstance(schema, ToolSchema)
    assert schema.name == "get_constant"
    assert len(schema.parameters) == 0


def test_generate_tool_schema_complex_return():
    def get_stats(numbers: list[float]) -> tuple[float, float]:
        """Calculate mean and standard deviation"""
        return sum(numbers) / len(numbers), 0.0

    schema = generate_tool_schema(get_stats)
    assert isinstance(schema, ToolSchema)
    assert schema.name == "get_stats"
    assert len(schema.parameters) == 1
    assert schema.parameters[0].name == "numbers"

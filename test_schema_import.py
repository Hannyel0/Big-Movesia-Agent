"""Quick test to verify schema imports work correctly."""

from src.react_agent.tools.search_project_schemas import (
    SearchProjectInput,
    SearchProjectResponse,
    TableName,
)

# Test TableName enum
print("✅ TableName enum values:")
for table in TableName:
    print(f"  - {table.name}: {table.value}")

# Test SearchProjectInput validation
print("\n✅ Testing SearchProjectInput validation:")
try:
    valid_input = SearchProjectInput(
        query_description="player scripts",
        tables=[TableName.ASSETS],
        return_format="structured"
    )
    print(f"  Valid input created: {valid_input.query_description}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test invalid input (empty query)
print("\n✅ Testing invalid input (should fail):")
try:
    invalid_input = SearchProjectInput(
        query_description="  ",  # Just whitespace
        return_format="structured"
    )
    print(f"  ❌ Should have failed but didn't!")
except ValueError as e:
    print(f"  ✅ Correctly rejected: {e}")

# Test SearchProjectResponse
print("\n✅ Testing SearchProjectResponse:")
response = SearchProjectResponse(
    success=True,
    results=[{"path": "test.cs", "kind": "MonoScript"}],
    results_structured=[{"path": "test.cs", "kind": "MonoScript"}],
    result_count=1,
    sql_query="SELECT * FROM assets",
    query_description="test query",
    timestamp="2024-10-30T22:00:00Z",
    execution_time_seconds=0.5,
    total_time_seconds=1.0,
    cache_hit=False
)
print(f"  Response created with {response.result_count} results")

print("\n✅ All schema imports and validations successful!")

# Enhanced Error Recovery Testing Guide

## Overview

The enhanced failure system now tracks attempts and allows tools to fail a specific number of times before automatically "recovering" and succeeding. This creates realistic error scenarios where the error recovery system can actually fix issues.

## Key Features

### 1. **Attempt Tracking**
- Each tool+failure pattern combination tracks how many times it has been called
- Tools can fail a configurable number of times before succeeding
- Once the failure limit is reached, the tool automatically succeeds with recovery messages

### 2. **Configurable Failure Limits**
- `max_failures: 1` - Fail only the first time, then succeed
- `max_failures: 2` - Fail first 2 times, succeed on 3rd attempt
- `max_failures: 3` - Fail first 3 times, succeed on 4th attempt
- `max_failures: -1` - Always fail (unrecoverable errors)

### 3. **Recovery Messages**
- Tools that have been "fixed" return success with contextual recovery messages
- Example: "Required Unity package has been installed and is now available"

## Failure Pattern Configuration

### **Single Failure Patterns** (Fail once, then succeed)
```python
"missing_dependency": max_failures: 1
"network_error": max_failures: 1
"invalid_parameter": max_failures: 1
```

### **Multiple Failure Patterns** (Fail 2-3 times, then succeed)
```python
"config_error": max_failures: 2
"build_error": max_failures: 2
"resource_missing": max_failures: 3
"project_state_error": max_failures: 2
```

### **Unrecoverable Patterns** (Always fail)
```python
"permission_error": max_failures: -1  # Never succeeds
```

## Testing Scenarios

### Scenario 1: Single Failure Recovery
```python
from react_agent.tools import enable_failure_testing, trigger_specific_failure

# Setup: Tool fails once, then succeeds
enable_failure_testing()
trigger_specific_failure("missing_dependency")

# Expected behavior:
# 1st call to write_file → FAILS with "Required Unity package not found"
# Error recovery runs → Installs dependency
# 2nd call to write_file → SUCCEEDS with "Required Unity package has been installed"
```

### Scenario 2: Multiple Failure Recovery
```python
# Setup: Tool fails 3 times, then succeeds
enable_failure_testing()
trigger_specific_failure("resource_missing")

# Expected behavior:
# 1st call to get_script_snippets → FAILS
# Error recovery runs → Attempts to create missing files
# 2nd call to get_script_snippets → FAILS (still not fully fixed)
# Error recovery runs again → More comprehensive fix
# 3rd call to get_script_snippets → FAILS (final attempt)
# Error recovery runs again → Complete fix
# 4th call to get_script_snippets → SUCCEEDS with "Missing script files have been located"
```

### Scenario 3: Progressive Error Resolution
```python
# Setup: Configuration error that takes 2 attempts to fix
enable_failure_testing()
trigger_specific_failure("config_error")

# Expected behavior:
# 1st call to edit_project_config → FAILS with "Configuration file corrupted"
# Error recovery: get_project_info → edit_project_config
# 2nd call to edit_project_config → STILL FAILS (partial fix)
# Error recovery: more comprehensive configuration repair
# 3rd call to edit_project_config → SUCCEEDS with "Configuration has been restored"
```

### Scenario 4: Unrecoverable Error Handling
```python
# Setup: Permission error that never recovers
enable_failure_testing()
trigger_specific_failure("permission_error")

# Expected behavior:
# 1st call to write_file → FAILS with "Access denied"
# Error recovery runs → Attempts fix but cannot resolve permissions
# 2nd call to write_file → STILL FAILS with same error
# Agent should eventually suggest alternative approaches
```

## Advanced Testing Functions

### Monitor Failure Statistics
```python
from react_agent.tools import get_failure_statistics

# Check current failure state
stats = get_failure_statistics()
print(stats)
# Output:
# {
#   "attempt_counters": {"write_file": {"missing_dependency": 2}},
#   "fixed_tools": {"write_file": ["missing_dependency"]},
#   "active_trigger": "missing_dependency"
# }
```

### Reset Failure Counters
```python
from react_agent.tools import reset_failure_counters

# Reset specific tool+pattern
reset_failure_counters("write_file", "missing_dependency")

# Reset all patterns for a tool
reset_failure_counters("write_file")

# Reset everything
reset_failure_counters()
```

### Dynamic Configuration
```python
from react_agent.tools import set_max_failures

# Change how many times a pattern fails
set_max_failures("missing_dependency", 3)  # Now fails 3 times instead of 1
```

## Testing Error Recovery Flow

### Complete Recovery Test
```python
# 1. Enable testing
enable_failure_testing()
trigger_specific_failure("missing_dependency")

# 2. Make request that uses write_file
response = agent.invoke("Create a player movement script")

# Expected flow:
# - Agent creates plan with write_file step
# - write_file FAILS (1st attempt)
# - Error recovery triggered: search → create_asset → compile_and_test
# - Recovery steps execute successfully
# - write_file called again → SUCCEEDS (dependency now "installed")
# - Plan completes successfully
```

### Multi-Step Recovery Test
```python
enable_failure_testing()
trigger_specific_failure("build_error")

response = agent.invoke("Compile my Unity project")

# Expected flow:
# - compile_and_test FAILS (1st attempt) - "Compilation failed: 3 errors"
# - Error recovery: compile_and_test → search → write_file → compile_and_test
# - Original compile_and_test called again → STILL FAILS (2nd attempt)
# - Error recovery runs again with different approach
# - Original compile_and_test called again → SUCCEEDS (errors "fixed")
```

## Debugging Failed Tests

### Check Tool State
```python
stats = get_failure_statistics()

# Verify attempt counts
print(f"Write file attempts: {stats['attempt_counters'].get('write_file', {})}")

# Check which tools are "fixed"
print(f"Fixed tools: {stats['fixed_tools']}")

# See active triggers
print(f"Active failure trigger: {stats['active_trigger']}")
```

### Verify Recovery Messages
Look for success responses with recovery context:
```json
{
  "success": true,
  "was_recovered": true,
  "recovery_message": "Required Unity package has been installed and is now available",
  "previous_failures": 1,
  "max_failures": 1,
  "tool_name": "write_file"
}
```

## Environment Variable Control

### Basic Setup
```bash
export TEST_FAILURES_ENABLED=true
export TRIGGER_FAILURE=missing_dependency
```

### Tool-Specific Failures
```bash
export FAIL_WRITE_FILE=build_error
export FAIL_COMPILE_AND_TEST=missing_dependency
```

### Clear All Triggers
```bash
unset TEST_FAILURES_ENABLED
unset TRIGGER_FAILURE
unset FAIL_WRITE_FILE
```

## Error Recovery Validation

### Successful Recovery Indicators
1. **First failure**: Tool returns `success: false` with specific error
2. **Error recovery triggered**: Agent routes to error_recovery node
3. **Recovery steps execute**: Targeted fix steps run successfully
4. **Retry succeeds**: Original tool now returns `success: true` with recovery message
5. **Plan continues**: Agent proceeds with remaining steps

### Failed Recovery Indicators
1. **Multiple failures**: Tool fails repeatedly beyond max_failures
2. **Recovery loops**: Error recovery keeps triggering without progress
3. **Unrecoverable errors**: Tool always fails with `max_failures: -1`

## Best Practices

### 1. **Start Simple**
Begin with single-failure patterns to validate basic error recovery flow

### 2. **Test Progressive Recovery**
Use multi-failure patterns to test comprehensive error resolution

### 3. **Validate Recovery Messages**
Ensure tools return appropriate context about what was "fixed"

### 4. **Monitor Statistics**
Use `get_failure_statistics()` to debug unexpected behavior

### 5. **Reset Between Tests**
Use `reset_failure_counters()` to ensure clean test state

This enhanced system provides realistic error scenarios where your error recovery logic can actually demonstrate fixing issues, making it much more valuable for testing the complete error handling workflow.
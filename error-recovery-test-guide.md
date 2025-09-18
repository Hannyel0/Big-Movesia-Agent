# Error Recovery System Testing Guide

## Overview

The enhanced tools now include sophisticated failure simulation that allows you to test different error scenarios and validate the error recovery system. Each failure type is designed to test specific recovery patterns.

## How to Trigger Failures

### Method 1: Environment Variables (Recommended)

```python
import os
from react_agent.tools import enable_failure_testing, trigger_specific_failure

# Enable testing mode
enable_failure_testing()

# Trigger specific failure types
trigger_specific_failure("missing_dependency")    # Tests dependency resolution
trigger_specific_failure("config_error")          # Tests configuration fixes  
trigger_specific_failure("resource_missing")      # Tests resource creation
trigger_specific_failure("build_error")           # Tests compilation fixes
trigger_specific_failure("project_state_error")   # Tests project state recovery
```

### Method 2: Direct Environment Variables

```bash
export TEST_FAILURES_ENABLED=true
export TRIGGER_FAILURE=missing_dependency

# Or target specific tools
export FAIL_WRITE_FILE=build_error
export FAIL_COMPILE_AND_TEST=missing_dependency
```

### Method 3: Tool-Specific Failures

```python
from react_agent.tools import trigger_tool_failure

# Make write_file fail with missing dependency error
trigger_tool_failure("write_file", "missing_dependency")

# Make compile_and_test fail with build errors
trigger_tool_failure("compile_and_test", "build_error")
```

## Available Failure Scenarios

### 1. **Missing Dependency** (`missing_dependency`)
- **Affects**: `write_file`, `compile_and_test`
- **Error**: "Required Unity package 'com.unity.inputsystem' not found in project"
- **Expected Recovery**: 
  1. `search` → Research the missing package
  2. `create_asset` → Install/create the dependency
  3. `compile_and_test` → Verify fix
  4. Retry original step

### 2. **Configuration Error** (`config_error`)
- **Affects**: `edit_project_config`, `get_project_info`
- **Error**: "Configuration file corrupted or inaccessible"
- **Expected Recovery**:
  1. `get_project_info` → Analyze project state
  2. `edit_project_config` → Fix configuration
  3. Retry original step

### 3. **Resource Missing** (`resource_missing`)
- **Affects**: `get_script_snippets`, `scene_management`
- **Error**: "Specified script file 'PlayerController.cs' does not exist in project"
- **Expected Recovery**:
  1. `get_project_info` → Check project structure
  2. `create_asset` → Create missing resource
  3. Retry original step

### 4. **Build Error** (`build_error`)
- **Affects**: `compile_and_test`, `write_file`
- **Error**: "Compilation failed: 3 errors, 7 warnings in PlayerController.cs"
- **Expected Recovery**:
  1. `compile_and_test` → Identify specific errors
  2. `search` → Research solutions
  3. `write_file` → Fix the code
  4. `compile_and_test` → Verify fix
  5. Retry original step

### 5. **Project State Error** (`project_state_error`)
- **Affects**: `get_project_info`, `scene_management`, `compile_and_test`
- **Error**: "Unity Editor is not running or project is not loaded"
- **Expected Recovery**:
  1. `get_project_info` → Check project accessibility
  2. `scene_management` → Reset project state
  3. Retry original step

### 6. **Permission Error** (`permission_error`)
- **Affects**: `write_file`, `edit_project_config`  
- **Error**: "Access denied: insufficient permissions to modify project files"
- **Recovery**: **Not recoverable** - tests fallback handling

## Testing Specific Scenarios

### Test 1: Dependency Resolution
```python
# Test missing Unity package during script creation
enable_failure_testing()
trigger_tool_failure("write_file", "missing_dependency")

# Ask agent to create a character controller script
# Expected: Agent should search for Unity packages, create dependency, then retry
```

### Test 2: Configuration Recovery  
```python
# Test project configuration issues
trigger_specific_failure("config_error")

# Ask agent to get project info
# Expected: Agent should fix config, then retry project inspection
```

### Test 3: Build Error Resolution
```python
# Test compilation failure recovery
trigger_tool_failure("compile_and_test", "build_error") 

# Ask agent to compile project
# Expected: Agent should identify errors, research fixes, apply fixes, retry
```

### Test 4: Resource Creation
```python
# Test missing script file recovery
trigger_tool_failure("get_script_snippets", "resource_missing")

# Ask agent to get movement code examples
# Expected: Agent should create missing script file, then retry
```

### Test 5: Unrecoverable Error Handling
```python
# Test permission error (unrecoverable)
trigger_specific_failure("permission_error")

# Ask agent to write a file
# Expected: Agent should recognize unrecoverable error, suggest alternatives
```

## Monitoring Recovery Process

### Check Error Recovery Logs
The system will provide detailed narration during recovery:

```
"I've identified a significant problem that needs resolution. 
Let me implement a systematic solution to fix this properly.

Issue: Required Unity package 'com.unity.inputsystem' not found in project
Recovery approach: 3 steps to resolve this properly."
```

### Verify Recovery Steps
Watch for recovery steps being inserted into the plan:
1. **Diagnosis step**: "Analyze project state to understand the error context"
2. **Fix step**: "Research and install missing Unity Input System package"  
3. **Validation step**: "Test compilation to verify dependency resolution"
4. **Retry original**: Return to the failed step

## Advanced Testing

### Sequential Failures
```python
# Test multiple consecutive failures
trigger_tool_failure("write_file", "missing_dependency")
trigger_tool_failure("compile_and_test", "build_error")

# Ask agent to create and test a script
# Expected: Multiple recovery cycles, each addressing specific issues
```

### Recovery Failure Testing
```python
# Make recovery tools also fail to test fallback
trigger_tool_failure("search", "network_error")
trigger_tool_failure("write_file", "missing_dependency") 

# Expected: Agent should try alternative recovery approaches
```

## Clearing Test State

### Reset All Failures
```python
from react_agent.tools import clear_failure_triggers, disable_failure_testing

clear_failure_triggers()
disable_failure_testing()
```

### Or via Environment
```bash
unset TEST_FAILURES_ENABLED
unset TRIGGER_FAILURE
unset FAIL_WRITE_FILE
unset FAIL_COMPILE_AND_TEST
```

## Expected Behavior Validation

### Successful Recovery
1. **Error detected** → Assessment identifies critical error
2. **Recovery triggered** → System routes to error_recovery node
3. **Problem diagnosed** → LLM analyzes the specific issue
4. **Recovery plan created** → Targeted steps to fix root cause
5. **Recovery executed** → Systematic resolution of the problem
6. **Original step retried** → Return to failed step after fix
7. **Success achieved** → Original goal accomplished

### Failed Recovery
1. **Recovery attempted** → Multiple recovery steps tried
2. **Recovery fails** → Recovery itself encounters issues
3. **Fallback triggered** → System falls back to repair node
4. **Alternative approach** → Try different implementation strategy

## Troubleshooting Test Issues

### Recovery Not Triggering
- Check `TEST_FAILURES_ENABLED=true`
- Verify failure type matches tool capabilities
- Ensure assessment node detects critical errors

### Wrong Recovery Strategy
- Check error categorization in diagnosis
- Verify recovery plan generation
- Review LLM prompt effectiveness

### Recovery Loops
- Monitor retry count limits
- Check recovery completion detection
- Verify return to original step logic

This testing system provides comprehensive coverage of error scenarios while maintaining the ability to return to normal operation quickly.
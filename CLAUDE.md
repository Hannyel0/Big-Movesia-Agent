# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Commands
- **Run tests**: `pytest tests/` (uses pytest framework)
- **Integration tests**: `pytest tests/integration_tests/`  
- **Unit tests**: `pytest tests/unit_tests/`
- **Lint code**: `ruff check` and `ruff format` (configured in pyproject.toml)
- **Type checking**: `mypy src/` (mypy configured as dev dependency)

### LangGraph Commands
- **Run in LangGraph Studio**: Open project folder in LangGraph Studio (configured via langgraph.json)
- **Main graph entry**: `./src/react_agent/graph.py:graph`

## Architecture Overview

This is a LangGraph ReAct agent template with enhanced planning and assessment capabilities, built on top of the standard ReAct pattern.

### Core Components

**State Management** (`src/react_agent/state.py`):
- `State`: Main state class with execution plan tracking, step management, and retry logic
- `ExecutionPlan`: Structured plans with ordered steps and dependencies
- `PlanStep`: Individual step with status tracking, success criteria, and error handling
- `AssessmentOutcome`: Structured assessment results with confidence scoring

**Graph Structure** (`src/react_agent/graph.py`):
The agent follows a plan → act → assess → route cycle:
- `plan`: Creates structured execution plans from user queries
- `act`: Executes current step using appropriate tools
- `assess`: Evaluates step success against defined criteria
- `advance_step`: Moves to next step after successful completion
- `repair`: Handles failed steps through replanning
- `finish`: Completes successful execution

**Context Configuration** (`src/react_agent/context.py`):
- Extensive configuration options for planning strategy, retry limits, assessment thresholds
- Support for separate models for planning, assessment, and main execution
- Feature flags for replanning, parallel execution, and chain-of-thought reasoning

### Key Patterns

**Planning Strategy**: The agent creates detailed execution plans with measurable success criteria for each step, rather than ad-hoc tool calling.

**Assessment-Driven Execution**: Each step is assessed against its success criteria before proceeding, enabling retry logic and replanning.

**Retry and Repair Logic**: Failed steps trigger configurable retry attempts before initiating plan repair/revision.

**Step Dependencies**: Plans can specify step dependencies, allowing for more complex execution flows.

### Configuration

Environment setup requires `.env` file with API keys:
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for the main model
- `TAVILY_API_KEY` for search functionality

Default model: `anthropic/claude-3-5-sonnet-20240620`

### Testing

Tests use pytest with LangSmith integration (`@unit` decorator). Integration tests verify end-to-end graph execution with real API calls.
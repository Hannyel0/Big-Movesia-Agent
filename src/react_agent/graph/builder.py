"""Enhanced graph builder with logging wrapper and file operation approval node."""

from __future__ import annotations

import logging
import time
from typing import Callable, Any, Dict
from functools import wraps

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphInterrupt

from react_agent.context import Context
from react_agent.state import State, InputState
from react_agent.tools import TOOLS
from react_agent.graph.nodes.classify import classify as _classify
from react_agent.graph.nodes.direct_act import direct_act as _direct_act
from react_agent.graph.nodes.simple_plan import simple_plan as _simple_plan
from react_agent.graph.nodes.plan import plan as _plan
from react_agent.graph.nodes.act import act as _act
from react_agent.graph.nodes.assess import assess as _assess
from react_agent.graph.nodes.finish import finish as _finish
from react_agent.graph.nodes.progress import (
    advance_step as _advance_step,
    increment_retry as _increment_retry,
)
from react_agent.graph.nodes.error_recovery import (
    execute_error_recovery as _execute_error_recovery,
)
from react_agent.graph.nodes.micro_retry import micro_retry as _micro_retry
from react_agent.graph.nodes.file_approval import (
    check_file_approval as _check_file_approval,
)
from react_agent.graph.routing import (
    should_continue,
    route_after_classify,
    route_after_plan,
    route_after_simple_plan,
    route_after_direct_act,
    route_after_act,
    route_after_assess,
    route_after_error_recovery,
    route_after_micro_retry,
    route_after_tools,  # NEW IMPORT
)

logger = logging.getLogger(__name__)


def log_node_execution(node_name: str):
    """Decorator to log node execution with state changes."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger.info(f"🔷 [{node_name}] ===== STARTING =====")
            try:
                result = await func(*args, **kwargs)
                logger.info(f"🔷 [{node_name}] ===== COMPLETED =====")

                # Log important state changes
                if isinstance(result, dict):
                    if "plan" in result and result["plan"]:
                        plan = result["plan"]
                        if hasattr(plan, "steps"):
                            logger.info(
                                f"🔷 [{node_name}]   Updated plan: {len(plan.steps)} steps"
                            )
                        else:
                            logger.info(f"🔷 [{node_name}]   Updated plan")

                    if "messages" in result and result["messages"]:
                        logger.info(
                            f"🔷 [{node_name}]   Added {len(result['messages'])} message(s)"
                        )

                    if "current_assessment" in result and result["current_assessment"]:
                        assessment = result["current_assessment"]
                        if hasattr(assessment, "outcome"):
                            logger.info(
                                f"🔷 [{node_name}]   Assessment: {assessment.outcome}"
                            )
                        else:
                            logger.info(f"🔷 [{node_name}]   Assessment completed")

                    if "complexity" in result:
                        logger.info(
                            f"🔷 [{node_name}]   Complexity: {result['complexity']}"
                        )

                    if "current_step" in result:
                        logger.info(
                            f"🔷 [{node_name}]   Current step: {result['current_step']}"
                        )

                    if "retry_count" in result:
                        logger.info(
                            f"🔷 [{node_name}]   Retry count: {result['retry_count']}"
                        )

                return result
            except GraphInterrupt as e:
                # ✅ FIX: GraphInterrupt is expected for human-in-the-loop - don't log as error
                logger.info(f"🔷 [{node_name}] ===== PAUSED FOR APPROVAL =====")
                logger.info(f"🔷 [{node_name}]   Waiting for user approval")
                raise  # Re-raise to let LangGraph handle it
            except Exception as e:
                logger.error(f"🔷 [{node_name}] ===== FAILED =====")
                logger.error(f"🔷 [{node_name}]   Error: {str(e)}", exc_info=True)
                raise

        return wrapper

    return decorator


# Wrap all nodes with logging
classify = log_node_execution("CLASSIFY")(_classify)
direct_act = log_node_execution("DIRECT_ACT")(_direct_act)
simple_plan = log_node_execution("SIMPLE_PLAN")(_simple_plan)
plan = log_node_execution("PLAN")(_plan)
act = log_node_execution("ACT")(_act)
assess = log_node_execution("ASSESS")(_assess)
finish = log_node_execution("FINISH")(_finish)
advance_step = log_node_execution("ADVANCE_STEP")(_advance_step)
increment_retry = log_node_execution("INCREMENT_RETRY")(_increment_retry)
execute_error_recovery = log_node_execution("ERROR_RECOVERY")(_execute_error_recovery)
micro_retry = log_node_execution("MICRO_RETRY")(_micro_retry)
check_file_approval = log_node_execution("CHECK_FILE_APPROVAL")(_check_file_approval)


# Custom ToolNode wrapper with logging
class TimedToolNode:
    """Wrapper around ToolNode to add logging."""

    def __init__(self, tools):
        self.tool_node = ToolNode(tools)
        self.logger = logging.getLogger(__name__)

    async def __call__(self, state: State, **kwargs):
        """Execute tools with logging."""
        self.logger.info(f"🔷 [TOOLS] ===== STARTING =====")

        try:
            # Get tool calls from last message
            if state.messages and hasattr(state.messages[-1], "tool_calls"):
                tool_calls = state.messages[-1].tool_calls
                if tool_calls:
                    self.logger.info(
                        f"🔧 [TOOLS] Executing {len(tool_calls)} tool call(s)"
                    )
                    for tc in tool_calls:
                        self.logger.info(f"🔧 [TOOLS]   - {tc.get('name', 'unknown')}")

            # Invoke the ToolNode properly using ainvoke
            result = await self.tool_node.ainvoke(state, **kwargs)

            self.logger.info(f"🔷 [TOOLS] ===== COMPLETED =====")

            return result

        except Exception as e:
            self.logger.error(f"🔷 [TOOLS] ===== FAILED =====")
            self.logger.error(f"🔷 [TOOLS] Error: {str(e)}", exc_info=True)
            raise


def create_graph() -> StateGraph:
    """Construct the ReAct agent graph with file operation approval."""
    builder = StateGraph(State, input_schema=InputState, context_schema=Context)

    # Add all nodes
    builder.add_node("classify", classify)
    builder.add_node("direct_act", direct_act)
    builder.add_node("simple_plan", simple_plan)
    builder.add_node("plan", plan)
    builder.add_node("act", act)
    builder.add_node("tools", TimedToolNode(TOOLS))  # Use instrumented ToolNode
    builder.add_node("check_file_approval", check_file_approval)  # NEW NODE
    builder.add_node("assess", assess)
    builder.add_node("error_recovery", execute_error_recovery)
    builder.add_node("micro_retry", micro_retry)
    builder.add_node("advance_step", advance_step)
    builder.add_node("increment_retry", increment_retry)
    builder.add_node("finish", finish)

    # Add edges
    builder.add_edge("__start__", "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_conditional_edges("direct_act", route_after_direct_act)
    builder.add_conditional_edges("simple_plan", route_after_simple_plan)
    builder.add_conditional_edges("plan", route_after_plan)
    builder.add_conditional_edges("act", route_after_act)
    builder.add_conditional_edges("tools", route_after_tools)  # NEW ROUTING
    builder.add_edge("check_file_approval", "assess")  # NEW EDGE
    builder.add_conditional_edges("assess", route_after_assess)
    builder.add_conditional_edges("error_recovery", route_after_error_recovery)
    builder.add_conditional_edges("micro_retry", route_after_micro_retry)
    builder.add_edge("advance_step", "act")
    builder.add_edge("increment_retry", "act")
    builder.add_edge("finish", "__end__")

    return builder.compile(name="ReAct Agent with File Operation Approval")


# Export the compiled graph
graph = create_graph()

"""Pydantic schemas for ReAct agent tools.

This package contains input and response schemas for all production tools,
providing type safety, validation, and better LLM integration.
"""

from react_agent.tools.schemas.search_project_schemas import (
    SearchProjectInput,
    SearchProjectResponse,
    TableName,
    AssetResult,
    GameObjectResult,
    ComponentResult,
)

from react_agent.tools.schemas.code_snippets_schemas import (
    CodeSnippetsInput,
    CodeSnippetsResponse,
    CodeSnippet,
    CodeSnippetsFilter,
    CodeMetadata,
)

from react_agent.tools.schemas.web_search_schemas import (
    SearchConfig,
    WebSearchInput,
    WebSearchResponse,
    SearchResult,
)

from react_agent.tools.schemas.file_operation_schemas import (
    ModificationSpec,
    modification_spec_to_dict,
    ReplaceAllSpec,
    InsertAfterSpec,
    InsertBeforeSpec,
    AppendSpec,
    PrependSpec,
    LineRangeSpec,
    PatternReplacementSpec,
    LineRangeReplacement,
    PatternReplacement,
)

__all__ = [
    # search_project schemas
    "SearchProjectInput",
    "SearchProjectResponse",
    "TableName",
    "AssetResult",
    "GameObjectResult",
    "ComponentResult",
    # code_snippets schemas
    "CodeSnippetsInput",
    "CodeSnippetsResponse",
    "CodeSnippet",
    "CodeSnippetsFilter",
    "CodeMetadata",
    # web_search schemas
    "SearchConfig",
    "WebSearchInput",
    "WebSearchResponse",
    "SearchResult",
    # file_operation schemas
    "ModificationSpec",
    "modification_spec_to_dict",
    "ReplaceAllSpec",
    "InsertAfterSpec",
    "InsertBeforeSpec",
    "AppendSpec",
    "PrependSpec",
    "LineRangeSpec",
    "PatternReplacementSpec",
    "LineRangeReplacement",
    "PatternReplacement",
]

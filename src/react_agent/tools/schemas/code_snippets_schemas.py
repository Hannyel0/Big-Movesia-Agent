"""Pydantic schemas for code_snippets tool - provides input validation and type safety."""

from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class CodeSnippetsFilter(BaseModel):
    """Filter criteria for code search.
    
    Validates filter parameters to ensure they're within acceptable ranges
    and formats.
    """
    file_extension: Optional[str] = Field(
        default=None,
        description="File extension filter (e.g., '.cs', '.js', '.shader')"
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Filter by namespace"
    )
    min_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum file size in bytes"
    )
    max_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum file size in bytes"
    )
    
    @field_validator("file_extension")
    @classmethod
    def validate_extension(cls, v: Optional[str]) -> Optional[str]:
        """Ensure file extension starts with a dot."""
        if v is not None and not v.startswith('.'):
            return f'.{v}'
        return v
    
    @field_validator("max_size")
    @classmethod
    def validate_size_range(cls, v: Optional[int], info) -> Optional[int]:
        """Ensure max_size is greater than min_size if both are set."""
        min_size = info.data.get('min_size')
        if v is not None and min_size is not None and v < min_size:
            raise ValueError("max_size must be greater than or equal to min_size")
        return v
    
    class Config:
        """Pydantic config."""
        extra = "forbid"  # Reject unknown fields to catch typos


class CodeSnippetsInput(BaseModel):
    """Input schema for code_snippets tool.
    
    Validates all inputs before execution to catch errors early and
    provides clear schema to LLM for better tool usage.
    """
    query: str = Field(
        ...,
        min_length=1,
        description="Search query - can be a file name or semantic description. "
                    "Examples: 'PlayerController', 'movement scripts', 'Movesia events'"
    )
    filter_by: Optional[CodeSnippetsFilter] = Field(
        default=None,
        description="Optional filter criteria for narrowing search results"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of results to return (1-10, default 3 for token efficiency)"
    )
    include_full_code: bool = Field(
        default=False,
        description="Include full code in results? False returns only signatures/metadata (saves tokens)"
    )
    max_code_chars: int = Field(
        default=1500,
        ge=100,
        le=5000,
        description="Maximum characters of code to include when include_full_code=True"
    )
    score_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for semantic search (0.0-1.0)"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query isn't just whitespace."""
        if not v.strip():
            raise ValueError("query cannot be empty or whitespace")
        return v.strip()


class CodeMetadata(BaseModel):
    """Metadata extracted from code without including full implementation.
    
    This provides rich context about the code structure while minimizing
    token usage.
    """
    classes: List[str] = Field(
        default_factory=list,
        description="Class names found in the code"
    )
    public_methods: List[str] = Field(
        default_factory=list,
        description="Public method names"
    )
    properties: List[str] = Field(
        default_factory=list,
        description="Property names"
    )
    unity_callbacks: List[str] = Field(
        default_factory=list,
        description="Unity lifecycle callbacks (Start, Update, etc.)"
    )
    namespaces: List[str] = Field(
        default_factory=list,
        description="Namespace declarations"
    )
    using_directives: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Using directives (limited to 5 for token efficiency)"
    )


class CodeSnippet(BaseModel):
    """Single code snippet result with metadata and optional code.
    
    Designed for token efficiency - includes rich metadata by default,
    full code only when explicitly requested.
    """
    # File identification
    file_path: str = Field(
        ...,
        description="Relative path to the file in the project"
    )
    file_name: str = Field(
        ...,
        description="Name of the file"
    )
    line_range: str = Field(
        default="",
        description="Line range of the snippet (e.g., '1-50')"
    )
    
    # Relevance scoring
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from search (0.0-1.0)"
    )
    match_type: Literal["exact", "contains", "starts_with", "semantic"] = Field(
        ...,
        description="Type of match: exact file name, contains, starts_with, or semantic"
    )
    
    # File metadata
    file_hash: str = Field(
        default="",
        description="Hash of the file for change detection"
    )
    
    # Code structure metadata (always included - low token cost)
    classes: List[str] = Field(
        default_factory=list,
        description="Class names in this file"
    )
    public_methods: List[str] = Field(
        default_factory=list,
        description="Public method names"
    )
    properties: List[str] = Field(
        default_factory=list,
        description="Property names"
    )
    unity_callbacks: List[str] = Field(
        default_factory=list,
        description="Unity callbacks present"
    )
    namespaces: List[str] = Field(
        default_factory=list,
        description="Namespaces declared"
    )
    using_directives: List[str] = Field(
        default_factory=list,
        description="Using directives (limited to 5)"
    )
    
    # Code statistics
    total_lines: int = Field(
        ...,
        ge=0,
        description="Total number of lines in the file"
    )
    code_size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of the code in bytes"
    )
    
    # Code content (conditional - high token cost)
    code: Optional[str] = Field(
        default=None,
        description="Full code content (only if include_full_code=True)"
    )
    code_signature: Optional[str] = Field(
        default=None,
        description="Code signature with class/method declarations (when include_full_code=False)"
    )
    code_truncated: bool = Field(
        default=False,
        description="Whether the code was truncated due to length limits"
    )
    truncation_point: Optional[int] = Field(
        default=None,
        ge=0,
        description="Character position where code was truncated"
    )
    full_code_available: bool = Field(
        default=True,
        description="Whether full code can be retrieved if needed"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "examples": [
                {
                    "file_path": "Assets/Scripts/PlayerController.cs",
                    "file_name": "PlayerController.cs",
                    "line_range": "1-120",
                    "relevance_score": 0.95,
                    "match_type": "exact",
                    "file_hash": "abc123...",
                    "classes": ["PlayerController"],
                    "public_methods": ["Move", "Jump", "Attack"],
                    "properties": ["Speed", "Health"],
                    "unity_callbacks": ["Start", "Update"],
                    "namespaces": ["Game.Player"],
                    "using_directives": ["UnityEngine", "System.Collections"],
                    "total_lines": 120,
                    "code_size_bytes": 4521,
                    "code_signature": "public class PlayerController : MonoBehaviour { ... }",
                    "full_code_available": True
                }
            ]
        }


class CodeSnippetsResponse(BaseModel):
    """Response from code_snippets tool with type-safe results.
    
    Provides comprehensive search results with metadata about the search
    strategy and performance metrics.
    """
    success: bool = Field(
        ...,
        description="Whether the search completed successfully"
    )
    query: str = Field(
        ...,
        description="Original search query"
    )
    file_patterns_extracted: List[str] = Field(
        default_factory=list,
        description="File name patterns extracted from the query for exact matching"
    )
    search_strategy: str = Field(
        default="two_stage_hybrid_token_efficient",
        description="Search strategy used (e.g., 'two_stage_hybrid_token_efficient')"
    )
    snippets: List[CodeSnippet] = Field(
        default_factory=list,
        description="List of code snippets found, ordered by relevance"
    )
    total_found: int = Field(
        default=0,
        ge=0,
        description="Total number of snippets found"
    )
    score_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold used for semantic search"
    )
    project_id: str = Field(
        default="",
        description="Project ID that was searched"
    )
    timestamp: str = Field(
        default="",
        description="ISO timestamp of when search completed"
    )
    
    # Token usage information
    include_full_code: bool = Field(
        default=False,
        description="Whether full code was included in results"
    )
    estimated_tokens: int = Field(
        default=0,
        ge=0,
        description="Estimated token count for the results"
    )
    token_savings_mode: bool = Field(
        default=True,
        description="Whether token savings mode was active (signatures only)"
    )
    
    # Performance metrics
    execution_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Execution time in milliseconds"
    )
    
    # Error information
    error: Optional[str] = Field(
        default=None,
        description="Error message if success=False"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Type of error that occurred"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "examples": [
                {
                    "success": True,
                    "query": "PlayerController",
                    "file_patterns_extracted": ["PlayerController", "Player", "Controller"],
                    "search_strategy": "two_stage_hybrid_token_efficient",
                    "snippets": [
                        {
                            "file_path": "Assets/Scripts/PlayerController.cs",
                            "file_name": "PlayerController.cs",
                            "relevance_score": 1.0,
                            "match_type": "exact",
                            "classes": ["PlayerController"],
                            "public_methods": ["Move", "Jump"],
                            "total_lines": 120,
                            "code_size_bytes": 4521
                        }
                    ],
                    "total_found": 1,
                    "score_threshold": 0.30,
                    "project_id": "unity_project_123",
                    "timestamp": "2024-10-30T22:30:00Z",
                    "include_full_code": False,
                    "estimated_tokens": 150,
                    "token_savings_mode": True,
                    "execution_time_ms": 45.2
                }
            ]
        }

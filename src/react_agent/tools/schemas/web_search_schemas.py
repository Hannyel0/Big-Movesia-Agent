"""Pydantic schemas for web_search tool - provides input validation and type safety."""

from __future__ import annotations
from typing import Optional, List, Literal, Any, Dict
from pydantic import BaseModel, Field, field_validator, model_validator
import os


class SearchConfig(BaseModel):
    """Configuration for the enhanced search tool with automatic validation.
    
    Replaces the dataclass version with Pydantic for better validation,
    type coercion, and error messages.
    """
    
    # SearXNG Configuration
    searxng_url: str = Field(
        default_factory=lambda: os.getenv("SEARXNG_URL", "http://localhost:8888"),
        description="SearXNG instance URL"
    )
    searxng_timeout: int = Field(
        default_factory=lambda: int(os.getenv("SEARXNG_TIMEOUT", "15")),
        ge=1,
        le=60,
        description="Timeout for SearXNG requests in seconds (1-60)"
    )
    searxng_engines: Optional[str] = Field(
        default_factory=lambda: os.getenv("SEARXNG_ENGINES"),
        description="Comma-separated list of engines (e.g., 'google,duckduckgo,brave')"
    )
    searxng_categories: Optional[str] = Field(
        default_factory=lambda: os.getenv("SEARXNG_CATEGORIES"),
        description="Comma-separated list of categories (e.g., 'general,images')"
    )
    searxng_language: str = Field(
        default_factory=lambda: os.getenv("SEARXNG_LANGUAGE", "en"),
        description="Language code for search results"
    )
    
    # Brave Search Configuration
    brave_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("BRAVE_SEARCH_API_KEY"),
        description="Brave Search API key for fallback"
    )
    brave_timeout: int = Field(
        default_factory=lambda: int(os.getenv("BRAVE_TIMEOUT", "10")),
        ge=1,
        le=60,
        description="Timeout for Brave API requests in seconds (1-60)"
    )
    brave_base_url: str = Field(
        default="https://api.search.brave.com/res/v1/web/search",
        description="Brave Search API base URL"
    )
    
    # Retry Configuration
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts (1-10)"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Initial retry delay in seconds (0.1-10.0)"
    )
    retry_exponential_base: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier (1.0-5.0)"
    )
    
    # Result Configuration
    max_results: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "10")),
        ge=1,
        le=100,
        description="Maximum number of search results to return (1-100)"
    )
    
    # Cache Configuration
    enable_cache: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_SEARCH_CACHE", "true").lower() == "true",
        description="Enable result caching"
    )
    cache_ttl: int = Field(
        default_factory=lambda: int(os.getenv("SEARCH_CACHE_TTL", "3600")),
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds (60-86400)"
    )
    cache_max_size: int = Field(
        default_factory=lambda: int(os.getenv("SEARCH_CACHE_SIZE", "100")),
        ge=10,
        le=1000,
        description="Maximum number of cached items (10-1000)"
    )
    
    # Fallback Configuration
    enable_brave_fallback: bool = Field(
        default=True,
        description="Enable Brave Search as fallback when SearXNG fails"
    )
    fallback_on_empty_results: bool = Field(
        default=True,
        description="Use fallback when SearXNG returns no results"
    )
    
    @field_validator("searxng_url")
    @classmethod
    def validate_searxng_url(cls, v: str) -> str:
        """Ensure SearXNG URL is properly formatted."""
        if not v:
            raise ValueError("searxng_url cannot be empty")
        if not v.startswith(("http://", "https://")):
            raise ValueError("searxng_url must start with http:// or https://")
        return v.rstrip("/")
    
    @field_validator("brave_base_url")
    @classmethod
    def validate_brave_url(cls, v: str) -> str:
        """Ensure Brave URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("brave_base_url must start with http:// or https://")
        return v.rstrip("/")
    
    @field_validator("brave_api_key")
    @classmethod
    def validate_brave_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate Brave API key format."""
        if v and len(v) < 20:
            raise ValueError("brave_api_key appears invalid (too short)")
        return v
    
    @field_validator("searxng_language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Ensure language code is lowercase."""
        return v.lower()
    
    @model_validator(mode='after')
    def check_fallback_configuration(self) -> 'SearchConfig':
        """Validate fallback configuration consistency."""
        if self.enable_brave_fallback and not self.brave_api_key:
            # Automatically disable fallback if no API key
            print("Warning: Brave fallback enabled but no API key provided. Disabling fallback.")
            self.enable_brave_fallback = False
        
        return self
    
    class Config:
        """Pydantic config."""
        validate_assignment = True  # Validate on attribute assignment
        extra = "forbid"  # Reject unknown fields


class WebSearchInput(BaseModel):
    """Input schema for web_search tool.
    
    Validates search parameters before execution.
    """
    query: str = Field(
        ...,
        min_length=1,
        description="Search query string"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return (1-100)"
    )
    time_range: Optional[Literal["day", "month", "year"]] = Field(
        default=None,
        description="Time range filter for results"
    )
    safe_search: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Safe search level: 0 (off), 1 (moderate), 2 (strict)"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query isn't just whitespace."""
        if not v.strip():
            raise ValueError("query cannot be empty or whitespace")
        return v.strip()


class SearchResult(BaseModel):
    """Single search result with validation.
    
    Ensures all results have required fields and valid URLs.
    """
    title: str = Field(
        ...,
        description="Title of the search result"
    )
    url: str = Field(
        ...,
        description="URL of the search result"
    )
    content: str = Field(
        default="",
        description="Content snippet or description"
    )
    engine: str = Field(
        default="unknown",
        description="Search engine that provided this result"
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        description="Relevance score (if available)"
    )
    category: str = Field(
        default="general",
        description="Result category"
    )
    publishedDate: Optional[str] = Field(
        default=None,
        description="Publication date (if available)"
    )
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is properly formatted."""
        if not v:
            raise ValueError("url cannot be empty")
        if not v.startswith(("http://", "https://", "//")):
            raise ValueError(f"url must be a valid HTTP(S) URL, got: {v[:50]}")
        return v
    
    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow additional fields from different engines


class WebSearchResponse(BaseModel):
    """Response from web_search tool with type-safe results.
    
    Provides comprehensive search results with metadata about the search
    execution and any fallback usage.
    """
    success: bool = Field(
        ...,
        description="Whether the search completed successfully"
    )
    query: str = Field(
        ...,
        description="Original search query"
    )
    engine: str = Field(
        default="unknown",
        description="Search engine used (searxng or brave)"
    )
    results: List[SearchResult] = Field(
        default_factory=list,
        description="List of search results"
    )
    result_count: int = Field(
        default=0,
        ge=0,
        description="Number of results returned"
    )
    timestamp: str = Field(
        ...,
        description="ISO timestamp of when search completed"
    )
    
    # Additional metadata
    from_cache: bool = Field(
        default=False,
        description="Whether results were retrieved from cache"
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether fallback engine was used"
    )
    
    # Rich metadata from search engines
    infoboxes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Infobox results (SearXNG)"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Search suggestions"
    )
    answers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Direct answers (SearXNG)"
    )
    corrections: List[str] = Field(
        default_factory=list,
        description="Query corrections (SearXNG)"
    )
    
    # Error information
    error: Optional[str] = Field(
        default=None,
        description="Error message if success=False"
    )
    engines_tried: List[str] = Field(
        default_factory=list,
        description="List of engines attempted (on failure)"
    )
    
    @model_validator(mode='after')
    def validate_result_count(self) -> 'WebSearchResponse':
        """Ensure result_count matches actual results length."""
        if self.success and self.result_count != len(self.results):
            # Auto-correct result_count to match actual results
            self.result_count = len(self.results)
        return self
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "examples": [
                {
                    "success": True,
                    "query": "Unity game development",
                    "engine": "searxng",
                    "results": [
                        {
                            "title": "Unity - Game Development Platform",
                            "url": "https://unity.com",
                            "content": "Create and grow real-time 3D games...",
                            "engine": "google",
                            "score": 0.95,
                            "category": "general"
                        }
                    ],
                    "result_count": 1,
                    "timestamp": "2024-10-30T22:35:00Z",
                    "from_cache": False,
                    "fallback_used": False,
                    "suggestions": ["unity3d", "unreal engine"]
                }
            ]
        }

"""Pydantic schemas for search_project tool - provides input validation and type safety."""

from __future__ import annotations
from typing import Optional, List, Union, Any, Dict
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class TableName(str, Enum):
    """Valid database tables in Unity/Unreal project database.
    
    Using Enum ensures type safety and prevents typos in table names.
    """
    ASSETS = "assets"
    ASSET_DEPS = "asset_deps"
    SCENES = "scenes"
    HIERARCHY_SCENES = "hierarchy_scenes"
    HIERARCHY_GAMEOBJECTS = "hierarchy_gameobjects"
    HIERARCHY_COMPONENTS = "hierarchy_components"
    EVENTS = "events"


class SearchProjectInput(BaseModel):
    """Input schema for search_project tool.
    
    Validates all inputs before execution to catch errors early and
    provides clear schema to LLM for better tool usage.
    """
    query_description: str = Field(
        ...,
        min_length=3,
        description="Natural language description of what to search for in the Unity/Unreal project. "
                    "Examples: 'player scripts', 'GameObjects in main scene', 'Rigidbody components'"
    )
    tables: Optional[List[TableName]] = Field(
        default=None,
        description="Optional hint about which database tables to query. "
                    "Valid tables: assets, asset_deps, scenes, hierarchy_scenes, "
                    "hierarchy_gameobjects, hierarchy_components, events"
    )
    return_format: str = Field(
        default="structured",
        description="Response format: 'structured' returns raw data, 'natural_language' returns readable text"
    )
    
    @field_validator("query_description")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query isn't just whitespace."""
        if not v.strip():
            raise ValueError("query_description cannot be empty or whitespace")
        return v.strip()
    
    @field_validator("return_format")
    @classmethod
    def validate_return_format(cls, v: str) -> str:
        """Ensure return_format is valid."""
        valid_formats = ["structured", "natural_language"]
        if v not in valid_formats:
            raise ValueError(f"return_format must be one of {valid_formats}, got '{v}'")
        return v
    
    class Config:
        """Pydantic config."""
        use_enum_values = True  # Automatically convert TableName enum to string values


class AssetResult(BaseModel):
    """Single asset search result."""
    guid: Optional[str] = None
    path: Optional[str] = None
    kind: Optional[str] = Field(None, description="MonoScript, Prefab, Scene, Material, etc.")
    mtime: Optional[float] = None
    size: Optional[int] = Field(None, ge=0)
    hash: Optional[str] = None
    deleted: Optional[int] = Field(None, ge=0, le=1)
    updated_ts: Optional[str] = None
    project_id: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields from database


class GameObjectResult(BaseModel):
    """Single GameObject search result."""
    id: Optional[int] = None
    project_id: Optional[str] = None
    scene_path: Optional[str] = None
    instance_id: Optional[int] = None
    name: Optional[str] = None
    hierarchy_path: Optional[str] = None
    parent_path: Optional[str] = None
    tag: Optional[str] = None
    layer: Optional[int] = None
    active_self: Optional[int] = Field(None, ge=0, le=1)
    active_in_hierarchy: Optional[int] = Field(None, ge=0, le=1)
    is_static: Optional[int] = Field(None, ge=0, le=1)
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    pos_z: Optional[float] = None
    rot_x: Optional[float] = None
    rot_y: Optional[float] = None
    rot_z: Optional[float] = None
    rot_w: Optional[float] = None
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None
    scale_z: Optional[float] = None
    sibling_index: Optional[int] = None
    last_updated: Optional[str] = None
    is_deleted: Optional[int] = Field(None, ge=0, le=1)
    
    class Config:
        extra = "allow"


class ComponentResult(BaseModel):
    """Single component search result."""
    id: Optional[int] = None
    project_id: Optional[str] = None
    scene_path: Optional[str] = None
    gameobject_instance_id: Optional[int] = None
    type_name: Optional[str] = Field(None, description="Short type name like 'Rigidbody'")
    full_type_name: Optional[str] = None
    assembly_name: Optional[str] = None
    enabled: Optional[int] = Field(None, ge=0, le=1)
    properties_json: Optional[str] = None
    last_updated: Optional[str] = None
    is_deleted: Optional[int] = Field(None, ge=0, le=1)
    
    class Config:
        extra = "allow"


class SearchProjectResponse(BaseModel):
    """Structured response from search_project tool.
    
    Provides type-safe response with clear success/error states and
    both structured and natural language results.
    """
    success: bool = Field(
        ...,
        description="Whether the search completed successfully"
    )
    results: Union[str, List[Dict[str, Any]]] = Field(
        ...,
        description="Search results - natural language string or structured list depending on return_format"
    )
    results_structured: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Always contains structured data regardless of return_format"
    )
    result_count: int = Field(
        default=0,
        ge=0,
        description="Number of results found"
    )
    sql_query: str = Field(
        default="",
        description="Generated SQL query for transparency"
    )
    query_description: str = Field(
        ...,
        description="Original query description"
    )
    timestamp: Optional[str] = Field(
        None,
        description="ISO timestamp of when search completed"
    )
    execution_time_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Time spent executing SQL query"
    )
    total_time_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Total time including SQL generation and formatting"
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether the SQL query was retrieved from cache"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if success=False"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "examples": [
                {
                    "success": True,
                    "results": "Found 3 results for 'player scripts':\n1. PlayerController.cs...",
                    "results_structured": [
                        {"path": "Assets/Scripts/PlayerController.cs", "kind": "MonoScript", "size": 4521}
                    ],
                    "result_count": 3,
                    "sql_query": "SELECT path, kind, size FROM assets WHERE...",
                    "query_description": "player scripts",
                    "timestamp": "2024-10-30T22:23:45.123456Z",
                    "execution_time_seconds": 0.045,
                    "total_time_seconds": 1.234,
                    "cache_hit": False
                }
            ]
        }

"""Pydantic schemas for file operation tools with explicit JSON schema generation."""

from pydantic import BaseModel, Field
from typing import Union, Literal, List, Annotated


class ReplaceAllSpec(BaseModel):
    """Replace entire file content with new code."""
    
    type: Literal["replace_all"] = Field(
        default="replace_all",
        description="Operation type"
    )
    content: str = Field(
        ...,
        description="Complete new file content including using statements, class definition, and all methods",
        examples=[
            "using UnityEngine;\n\npublic class Test8527 : MonoBehaviour\n{\n    void Start() { }\n}"
        ]
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "type": "replace_all",
                    "content": "using UnityEngine;\n\npublic class PlayerController : MonoBehaviour\n{\n    void Update() {\n        // Movement logic\n    }\n}"
                }
            ]
        }


class InsertAfterSpec(BaseModel):
    """Insert content after a specific line."""
    
    type: Literal["insert_after"] = Field(
        default="insert_after",
        description="Operation type"
    )
    target_line: str = Field(
        ...,
        description="Exact line of code to insert after (e.g., 'void Start()')",
        examples=["void Start()", "public class MyClass"]
    )
    content: str = Field(
        ...,
        description="Content to insert (will be placed on new lines after target)",
        examples=["    Debug.Log(\"Initialized\");"]
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "type": "insert_after",
                    "target_line": "void Start()",
                    "content": "    controller = GetComponent<CharacterController>();\n    Debug.Log(\"Setup complete\");"
                }
            ]
        }


class InsertBeforeSpec(BaseModel):
    """Insert content before a specific line."""
    
    type: Literal["insert_before"] = Field(
        default="insert_before",
        description="Operation type"
    )
    target_line: str = Field(
        ...,
        description="Exact line of code to insert before",
        examples=["void Update()", "public class MyClass"]
    )
    content: str = Field(
        ...,
        description="Content to insert",
        examples=["    // Pre-update logic"]
    )


class AppendSpec(BaseModel):
    """Append content to the end of file."""
    
    type: Literal["append"] = Field(
        default="append",
        description="Operation type"
    )
    content: str = Field(
        ...,
        description="Content to append at end of file",
        examples=["    void OnDestroy() { Cleanup(); }"]
    )


class PrependSpec(BaseModel):
    """Prepend content to the start of file."""
    
    type: Literal["prepend"] = Field(
        default="prepend",
        description="Operation type"
    )
    content: str = Field(
        ...,
        description="Content to prepend at start of file",
        examples=["using System.Collections;"]
    )


class LineRangeReplacement(BaseModel):
    """Replace a range of lines."""
    
    start: int = Field(..., description="Start line number (1-indexed)", ge=1)
    end: int = Field(..., description="End line number (1-indexed)", ge=1)
    replacement: str = Field(..., description="New content to replace the range")


class LineRangeSpec(BaseModel):
    """Replace specific line ranges."""
    
    type: Literal["line_ranges"] = Field(
        default="line_ranges",
        description="Operation type"
    )
    ranges: List[LineRangeReplacement] = Field(
        ...,
        description="List of line ranges to replace"
    )


class PatternReplacement(BaseModel):
    """Replace text matching a pattern."""
    
    pattern: str = Field(..., description="Regex pattern to match")
    replacement: str = Field(..., description="Replacement text")


class PatternReplacementSpec(BaseModel):
    """Replace content using regex patterns."""
    
    type: Literal["pattern_replacements"] = Field(
        default="pattern_replacements",
        description="Operation type"
    )
    patterns: List[PatternReplacement] = Field(
        ...,
        description="List of pattern replacements to apply"
    )


# ⭐ Discriminated Union - LLM gets clear schema for each type
ModificationSpec = Annotated[
    Union[
        ReplaceAllSpec,
        InsertAfterSpec,
        InsertBeforeSpec,
        AppendSpec,
        PrependSpec,
        LineRangeSpec,
        PatternReplacementSpec
    ],
    Field(discriminator='type')
]


def modification_spec_to_dict(spec: ModificationSpec) -> dict:
    """
    Convert Pydantic model to the dict format expected by existing implementation.
    
    This adapter allows us to use Pydantic for validation while maintaining
    backward compatibility with the existing _file_modify_prepare implementation.
    """
    # ⚠️ Validate input type
    if not isinstance(spec, (ReplaceAllSpec, InsertAfterSpec, InsertBeforeSpec, 
                             AppendSpec, PrependSpec, LineRangeSpec, PatternReplacementSpec)):
        raise TypeError(f"Expected ModificationSpec, got {type(spec)}")
    
    if isinstance(spec, ReplaceAllSpec):
        return {"replace_all": spec.content}
    
    elif isinstance(spec, InsertAfterSpec):
        return {
            "insert_after": spec.target_line,
            "content_to_insert": spec.content
        }
    
    elif isinstance(spec, InsertBeforeSpec):
        return {
            "insert_before": spec.target_line,
            "content_to_insert": spec.content
        }
    
    elif isinstance(spec, AppendSpec):
        return {"append": spec.content}
    
    elif isinstance(spec, PrependSpec):
        return {"prepend": spec.content}
    
    elif isinstance(spec, LineRangeSpec):
        return {
            "line_ranges": [
                {
                    "start": r.start,
                    "end": r.end,
                    "replacement": r.replacement
                }
                for r in spec.ranges
            ]
        }
    
    elif isinstance(spec, PatternReplacementSpec):
        return {
            "pattern_replacements": [
                {
                    "pattern": p.pattern,
                    "replacement": p.replacement
                }
                for p in spec.patterns
            ]
        }
    
    else:
        raise ValueError(f"Unknown modification spec type: {type(spec)}")

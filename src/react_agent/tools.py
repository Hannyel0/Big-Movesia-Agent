"""Enhanced tools for the ReAct agent with sophisticated failure simulation and attempt tracking."""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any, List
import asyncio
import json
import os
from datetime import datetime, UTC
from collections import defaultdict

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from react_agent.context import Context


class EnhancedFailureTestConfig:
    """Enhanced configuration for controlled failure testing with attempt tracking."""

    def __init__(self):
        self.enabled = os.getenv("TEST_FAILURES_ENABLED", "false").lower() == "true"
        self.failure_patterns = self._load_failure_patterns()
        # Track attempts per tool per failure pattern
        self.attempt_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Track which tools have been "fixed" (exceeded max failures)
        self.fixed_tools: Dict[str, set] = defaultdict(set)

    def _load_failure_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load failure patterns with configurable failure counts."""
        return {
            # Configuration errors - fail first 2 times
            "config_error": {
                "tools": ["edit_project_config", "get_project_info"],
                "error": "Configuration file corrupted or inaccessible",
                "category": "configuration_error",
                "recoverable": True,
                "max_failures": 2,  # NEW: Max times this can fail before succeeding
                "recovery_message": "Configuration has been restored and is now accessible"
            },

            # Missing dependency errors - fail first time only
            "missing_dependency": {
                "tools": ["write_file", "compile_and_test"],
                "error": "Required Unity package 'com.unity.inputsystem' not found in project",
                "category": "dependency_missing",
                "recoverable": True,
                "max_failures": 1,  # Fail only once
                "recovery_message": "Required Unity package has been installed and is now available"
            },

            # Resource not found errors - fail first 3 times
            "resource_missing": {
                "tools": ["get_script_snippets", "scene_management"],
                "error": "Specified script file 'PlayerController.cs' does not exist in project",
                "category": "resource_not_found",
                "recoverable": True,
                "max_failures": 3,
                "recovery_message": "Missing script files have been located and are now accessible"
            },

            # Permission/access errors - always fail (unrecoverable)
            "permission_error": {
                "tools": ["write_file", "edit_project_config"],
                "error": "Access denied: insufficient permissions to modify project files",
                "category": "permission_error",
                "recoverable": False,
                "max_failures": -1,  # -1 means always fail
                "recovery_message": None
            },

            # Build/compilation errors - fail first 2 times
            "build_error": {
                "tools": ["compile_and_test", "write_file"],
                "error": "Compilation failed: 3 errors, 7 warnings in PlayerController.cs",
                "category": "build_error",
                "recoverable": True,
                "max_failures": 2,
                "recovery_message": "Compilation errors have been resolved - build is now clean"
            },

            # Network/external service errors - fail first time only
            "network_error": {
                "tools": ["search"],
                "error": "Network timeout: Unable to connect to search service",
                "category": "network_error",
                "recoverable": True,
                "max_failures": 1,
                "recovery_message": "Network connection has been restored"
            },

            # Invalid parameter errors - fail first time only
            "invalid_parameter": {
                "tools": ["create_asset", "scene_management"],
                "error": "Invalid asset type 'InvalidType' - must be one of: script, prefab, material, scene",
                "category": "invalid_parameter",
                "recoverable": True,
                "max_failures": 1,
                "recovery_message": "Parameter validation has been updated - invalid parameters are now handled correctly"
            },

            # Project state errors - fail first 2 times
            "project_state_error": {
                "tools": ["get_project_info", "scene_management", "compile_and_test"],
                "error": "Unity Editor is not running or project is not loaded",
                "category": "project_state_error",
                "recoverable": True,
                "max_failures": 2,
                "recovery_message": "Unity Editor has been started and project is now loaded"
            }
        }

    def should_fail(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Determine if a tool should fail based on attempt tracking."""
        if not self.enabled:
            return None

        # Check environment variable for specific tool failure
        specific_failure = os.getenv(f"FAIL_{tool_name.upper()}")
        if specific_failure:
            pattern_name = specific_failure
        else:
            # Check for general failure trigger
            failure_trigger = os.getenv("TRIGGER_FAILURE")
            if not failure_trigger:
                return None
            pattern_name = failure_trigger

        # Find matching failure pattern
        if pattern_name not in self.failure_patterns:
            return None
            
        pattern = self.failure_patterns[pattern_name]
        if tool_name not in pattern["tools"]:
            return None

        # Check if this tool+pattern combination has already been "fixed"
        if pattern_name in self.fixed_tools[tool_name]:
            return None  # Don't fail anymore - it's been "fixed"

        max_failures = pattern["max_failures"]
        
        # Always fail if max_failures is -1 (unrecoverable)
        if max_failures == -1:
            current_attempts = self.attempt_counters[tool_name][pattern_name]
            self.attempt_counters[tool_name][pattern_name] = current_attempts + 1
            return {
                "should_fail": True,
                "error_message": pattern["error"],
                "error_category": pattern["category"],
                "recoverable": pattern["recoverable"],
                "attempt_count": current_attempts + 1,
                "max_failures": max_failures,
                "recovery_message": pattern["recovery_message"]
            }

        # Check current attempt count for recoverable errors
        current_attempts = self.attempt_counters[tool_name][pattern_name]
        
        if current_attempts < max_failures:
            # Still within failure limit - fail this time
            self.attempt_counters[tool_name][pattern_name] = current_attempts + 1
            return {
                "should_fail": True,
                "error_message": pattern["error"],
                "error_category": pattern["category"],
                "recoverable": pattern["recoverable"],
                "attempt_count": current_attempts + 1,
                "max_failures": max_failures,
                "recovery_message": pattern["recovery_message"]
            }
        else:
            # Exceeded failure limit - mark as "fixed" and succeed
            self.fixed_tools[tool_name].add(pattern_name)
            return {
                "should_fail": False,  # SUCCESS after being "fixed"
                "was_fixed": True,
                "recovery_message": pattern["recovery_message"],
                "attempt_count": current_attempts + 1,
                "max_failures": max_failures
            }

    def reset_failures(self, tool_name: Optional[str] = None, pattern_name: Optional[str] = None):
        """Reset failure counters for testing purposes."""
        if tool_name and pattern_name:
            # Reset specific tool+pattern combination
            if tool_name in self.attempt_counters:
                self.attempt_counters[tool_name][pattern_name] = 0
            if tool_name in self.fixed_tools:
                self.fixed_tools[tool_name].discard(pattern_name)
        elif tool_name:
            # Reset all patterns for a specific tool
            self.attempt_counters[tool_name].clear()
            self.fixed_tools[tool_name].clear()
        else:
            # Reset everything
            self.attempt_counters.clear()
            self.fixed_tools.clear()

    def get_failure_stats(self) -> Dict[str, Any]:
        """Get current failure statistics for debugging."""
        return {
            "attempt_counters": dict(self.attempt_counters),
            "fixed_tools": {tool: list(patterns) for tool, patterns in self.fixed_tools.items()},
            "enabled_patterns": list(self.failure_patterns.keys()),
            "active_trigger": os.getenv("TRIGGER_FAILURE"),
            "tool_specific_triggers": {
                key: value for key, value in os.environ.items() 
                if key.startswith("FAIL_")
            }
        }

# Global enhanced failure config instance
failure_config = EnhancedFailureTestConfig()


def _simulate_failure(tool_name: str, failure_info: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate a tool failure with realistic error information."""
    base_result = {
        "success": False,
        "error": failure_info["error_message"],
        "error_category": failure_info["error_category"],
        "recoverable": failure_info["recoverable"],
        "tool_name": tool_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "attempt_count": failure_info["attempt_count"]
    }

    # Add tool-specific failure details
    if tool_name == "compile_and_test":
        base_result.update({
            "compilation_errors": [
                {
                    "file": "Assets/Scripts/PlayerController.cs",
                    "line": 23,
                    "error": "CS0103: The name 'InputSystem' does not exist in the current context"
                },
                {
                    "file": "Assets/Scripts/PlayerController.cs",
                    "line": 45,
                    "error": "CS0246: The type or namespace name 'CharacterController' could not be found"
                }
            ],
            "warnings": [
                {
                    "file": "Assets/Scripts/GameManager.cs",
                    "line": 12,
                    "warning": "CS0649: Field 'GameManager.playerScore' is never assigned to"
                }
            ]
        })

    elif tool_name == "write_file":
        base_result.update({
            "attempted_path": "Assets/Scripts/PlayerController.cs",
            "file_size_attempted": 1247,
            "partial_content_written": False
        })

    elif tool_name == "get_script_snippets":
        base_result.update({
            "searched_paths": [
                "Assets/Scripts/",
                "Assets/Standard Assets/",
                "Packages/com.unity.standardassets/"
            ],
            "files_checked": 0,
            "search_pattern": "movement, character controller"
        })

    elif tool_name == "scene_management":
        base_result.update({
            "attempted_scene": "MainScene.unity",
            "attempted_action": "load_scene",
            "current_scenes": []
        })

    elif tool_name == "search":
        base_result.update({
            "query": "Unity character controller tutorial",
            "timeout_duration": "30 seconds",
            "retry_attempts": 3
        })

    return base_result


@tool
async def search(query: str) -> Dict[str, Any]:
    """Search for general web results about game development, Unity, Unreal Engine, and related topics."""
    try:
        # Check for controlled failure with proper guard
        failure_info = failure_config.should_fail("search")
        if failure_info and failure_info.get("should_fail", False):
            return _simulate_failure("search", failure_info)

        runtime = get_runtime(Context)
        timeout = runtime.context.tool_timeout_seconds

        # Simulate realistic search results for testing
        simulated_results = [
            {
                "title": f"Unity Best Practices for {query}",
                "url": "https://docs.unity3d.com/best-practices",
                "content": f"Comprehensive guide covering {query} implementation in Unity with examples and code snippets.",
                "score": 0.95
            },
            {
                "title": f"Unity Forum Discussion: {query}",
                "url": "https://forum.unity.com/threads/character-controller",
                "content": f"Community discussion about {query} with solutions to common problems and debugging tips.",
                "score": 0.88
            },
            {
                "title": f"Unity Manual: {query}",
                "url": "https://docs.unity3d.com/manual/character-controller",
                "content": f"Official Unity documentation for {query} with API reference and implementation examples.",
                "score": 0.92
            },
            {
                "title": f"YouTube Tutorial: {query}",
                "url": "https://youtube.com/unity-tutorial",
                "content": f"Step-by-step video tutorial covering {query} implementation in Unity 2023.",
                "score": 0.85
            },
            {
                "title": f"Unity Learn: {query}",
                "url": "https://learn.unity.com/tutorial/character-movement",
                "content": f"Interactive Unity Learn tutorial for {query} with downloadable project files.",
                "score": 0.90
            }
        ]
        
        # Handle recovery case or normal success
        result = {
            "success": True,
            "result": simulated_results,
            "query": query,
            "total_results": len(simulated_results),
            "timestamp": datetime.now(UTC).isoformat(),
            "message": failure_info.get("recovery_message", f"Found {len(simulated_results)} relevant resources for '{query}'")
        } if failure_info else {
            "success": True,
            "result": simulated_results,
            "query": query,
            "total_results": len(simulated_results),
            "timestamp": datetime.now(UTC).isoformat(),
            "message": f"Found {len(simulated_results)} relevant resources for '{query}'"
        }
        return result
        
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Search timed out after {timeout} seconds",
            "timestamp": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "timestamp": datetime.now(UTC).isoformat()
        }


@tool
async def get_project_info() -> Dict[str, Any]:
    """Get information about the current Unity or Unreal Engine project."""
    # Check for controlled failure with proper guard
    failure_info = failure_config.should_fail("get_project_info")
    if failure_info and failure_info.get("should_fail", False):
        return _simulate_failure("get_project_info", failure_info)

    return {
        "success": True,
        "engine": "Unity",
        "version": "2023.3.15f1",
        "project_name": "MyGameProject",
        "current_scene": "MainScene.unity",
        "target_platform": "PC, Mac & Linux Standalone",
        "render_pipeline": "Universal Render Pipeline",
        "installed_packages": [
            "Unity UI",
            "Post Processing",
            "Cinemachine",
            "Input System",
            "Timeline",
            "TextMeshPro"
        ],
        "project_structure": {
            "Assets": {
                "Scripts": ["PlayerController.cs", "GameManager.cs", "UIManager.cs"],
                "Scenes": ["MainScene.unity", "MenuScene.unity"],
                "Prefabs": ["Player.prefab", "Enemy.prefab", "UI_Canvas.prefab"],
                "Materials": ["Ground.mat", "Player.mat", "Sky.mat"],
                "Textures": ["ground_texture.png", "player_sprite.png"]
            }
        },
        "build_settings": {
            "scenes_in_build": ["MenuScene", "MainScene"],
            "platform": "Windows x64"
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "message": failure_info.get("recovery_message", "Successfully retrieved project information") if failure_info else "Successfully retrieved project information"
    }


@tool
async def create_asset(asset_type: str, name: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new asset in the game project."""
    # Check for controlled failure with proper guard
    failure_info = failure_config.should_fail("create_asset")
    if failure_info and failure_info.get("should_fail", False):
        return _simulate_failure("create_asset", failure_info)

    if not properties:
        properties = {}

    asset_info = {
        "success": True,
        "asset_type": asset_type,
        "name": name,
        "path": f"Assets/{asset_type.title()}s/{name}",
        "created_at": datetime.now(UTC).isoformat(),
        "properties": properties,
        "status": "created",
        "message": failure_info.get("recovery_message", f"Successfully created {asset_type} '{name}'") if failure_info else f"Successfully created {asset_type} '{name}'"
    }
    
    # Add type-specific information
    if asset_type.lower() == "script":
        asset_info.update({
            "language": "C#",
            "template": "MonoBehaviour",
            "methods": ["Start", "Update"],
            "namespaces": ["UnityEngine"],
            "file_extension": ".cs"
        })
    elif asset_type.lower() == "prefab":
        asset_info.update({
            "components": ["Transform", "Renderer", "Collider"],
            "children": 0,
            "size": "1.2 MB",
            "file_extension": ".prefab"
        })
    elif asset_type.lower() == "material":
        asset_info.update({
            "shader": "Universal Render Pipeline/Lit",
            "textures": [],
            "color": "White",
            "file_extension": ".mat"
        })
    elif asset_type.lower() == "scene":
        asset_info.update({
            "objects": ["Main Camera", "Directional Light"],
            "lighting": "Realtime",
            "skybox": "Default",
            "file_extension": ".unity"
        })
    
    return asset_info


@tool
async def write_file(file_path: str, content: str, file_type: str = "script") -> Dict[str, Any]:
    """Write or create a file in the project (scripts, config files, etc.)."""
    # Check for controlled failure
    failure_info = failure_config.should_fail("write_file")
    if failure_info:
        return _simulate_failure("write_file", failure_info)

    return {
        "success": True,
        "file_path": file_path,
        "file_type": file_type,
        "size_bytes": len(content),
        "lines_written": len(content.split('\n')),
        "encoding": "UTF-8",
        "timestamp": datetime.now(UTC).isoformat(),
        "content_preview": content[:100] + ("..." if len(content) > 100 else ""),
        "message": f"Successfully wrote {file_type} file to {file_path}"
    }


@tool
async def edit_project_config(config_section: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Edit project configuration settings."""
    # Check for controlled failure
    failure_info = failure_config.should_fail("edit_project_config")
    if failure_info:
        return _simulate_failure("edit_project_config", failure_info)

    return {
        "success": True,
        "config_section": config_section,
        "updated_settings": settings,
        "previous_values": {
            # Simulated previous values
            key: f"previous_{key}_value" for key in settings.keys()
        },
        "requires_restart": config_section in ["player_settings", "graphics"],
        "timestamp": datetime.now(UTC).isoformat(),
        "message": f"Successfully updated {config_section} configuration"
    }


@tool
async def get_script_snippets(category: str, language: str = "csharp") -> Dict[str, Any]:
    """Get code snippets from the USER'S existing Unity project scripts."""
    # Check for controlled failure
    failure_info = failure_config.should_fail("get_script_snippets")
    if failure_info:
        return _simulate_failure("get_script_snippets", failure_info)

    # Simulate finding snippets in the user's existing scripts
    found_snippets = {
        "PlayerController.cs": {
            "movement_method": '''void HandleMovement()
{
    float horizontal = Input.GetAxis("Horizontal");
    float vertical = Input.GetAxis("Vertical");
    
    Vector3 movement = new Vector3(horizontal, 0, vertical);
    transform.Translate(movement * speed * Time.deltaTime);
}''',
            "jump_method": '''void HandleJump()
{
    if (Input.GetKeyDown(KeyCode.Space) && isGrounded)
    {
        rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
    }
}'''
        },
        "GameManager.cs": {
            "game_state": '''public enum GameState
{
    Menu,
    Playing,
    Paused,
    GameOver
}

private GameState currentState = GameState.Menu;''',
            "score_system": '''private int playerScore = 0;

public void AddScore(int points)
{
    playerScore += points;
    UpdateScoreUI();
}'''
        },
        "UIManager.cs": {
            "update_health": '''public void UpdateHealthBar(float currentHealth, float maxHealth)
{
    healthSlider.value = currentHealth / maxHealth;
    healthText.text = $"{currentHealth}/{maxHealth}";
}''',
            "show_menu": '''public void ShowMenu(GameObject menu)
{
    menu.SetActive(true);
    Time.timeScale = 0f;
}'''
        }
    }
    
    # Filter snippets based on category
    category_lower = category.lower()
    relevant_snippets = {}
    
    for script_name, snippets in found_snippets.items():
        script_relevant_snippets = {}
        
        for snippet_name, code in snippets.items():
            # Match category to snippet content
            if any(keyword in category_lower for keyword in ["movement", "player", "character", "control"]):
                if any(term in snippet_name.lower() for term in ["movement", "jump", "control"]):
                    script_relevant_snippets[snippet_name] = code
            elif any(keyword in category_lower for keyword in ["ui", "menu", "interface"]):
                if any(term in snippet_name.lower() for term in ["health", "menu", "ui"]):
                    script_relevant_snippets[snippet_name] = code
            elif any(keyword in category_lower for keyword in ["game", "manager", "state"]):
                if any(term in snippet_name.lower() for term in ["game", "score", "state"]):
                    script_relevant_snippets[snippet_name] = code
            elif any(keyword in category_lower for keyword in ["physics", "jump", "force"]):
                if any(term in snippet_name.lower() for term in ["jump", "force", "physics"]):
                    script_relevant_snippets[snippet_name] = code
        
        if script_relevant_snippets:
            relevant_snippets[script_name] = script_relevant_snippets
    
    total_snippets = sum(len(snippets) for snippets in relevant_snippets.values())
    
    return {
        "success": True,
        "category": category,
        "language": language,
        "scripts_searched": list(found_snippets.keys()),
        "relevant_scripts": list(relevant_snippets.keys()),
        "snippets_by_script": relevant_snippets,
        "total_snippets_found": total_snippets,
        "timestamp": datetime.now(UTC).isoformat(),
        "message": f"Found {total_snippets} code snippets in {len(relevant_snippets)} scripts matching '{category}'"
    }


@tool
async def compile_and_test(target: str = "editor") -> Dict[str, Any]:
    """Compile the project and run basic tests."""
    # Check for controlled failure with proper guard
    failure_info = failure_config.should_fail("compile_and_test")
    if failure_info and failure_info.get("should_fail", False):
        return _simulate_failure("compile_and_test", failure_info)

    # Handle recovery case or normal success
    result = {
        "success": True,
        "target": target,
        "compilation_time": "12.3 seconds",
        "warnings": 2,
        "errors": 0,
        "details": {
            "scripts_compiled": 47,
            "assets_processed": 156,
            "build_size": "45.2 MB",
            "warnings": [
                "Unused variable 'tempVar' in PlayerController.cs line 23",
                "Missing reference in UIManager prefab"
            ]
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "message": failure_info.get("recovery_message", f"Successfully compiled for {target} with 2 warnings and 0 errors") if failure_info else f"Successfully compiled for {target} with 2 warnings and 0 errors"
    }
    return result


@tool
async def scene_management(action: str, scene_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Manage scenes in the project (create, load, modify, etc.)."""
    # Check for controlled failure
    failure_info = failure_config.should_fail("scene_management")
    if failure_info:
        return _simulate_failure("scene_management", failure_info)

    if not parameters:
        parameters = {}

    result = {
        "success": True,
        "action": action,
        "scene_name": scene_name,
        "timestamp": datetime.now(UTC).isoformat()
    }
    
    if action == "create":
        result.update({
            "scene_path": f"Assets/Scenes/{scene_name}.unity",
            "default_objects": ["Main Camera", "Directional Light"],
            "lighting_settings": "Realtime",
            "message": f"Successfully created new scene: {scene_name}"
        })
    elif action == "load":
        result.update({
            "objects_in_scene": ["Player", "Ground", "UI Canvas", "Main Camera", "Directional Light"],
            "lighting": "Mixed",
            "active_objects": 12,
            "message": f"Successfully loaded scene: {scene_name}"
        })
    elif action == "add_object":
        obj_type = parameters.get("object_type", "GameObject")
        result.update({
            "object_added": obj_type,
            "position": parameters.get("position", [0, 0, 0]),
            "rotation": parameters.get("rotation", [0, 0, 0]),
            "message": f"Successfully added {obj_type} to {scene_name}"
        })
    elif action == "save":
        result.update({
            "changes_saved": True,
            "backup_created": True,
            "message": f"Successfully saved scene: {scene_name}"
        })
    
    return result


# Export tools - focused on game development
TOOLS = [
    search,
    get_project_info,
    create_asset,
    write_file,
    edit_project_config,
    get_script_snippets,
    compile_and_test,
    scene_management,
]


# Tool metadata for game development context - now includes type validation
TOOL_METADATA = {
    "search": {
        "category": "information_retrieval",
        "cost": "medium",
        "reliability": "high",
        "best_for": ["game dev tutorials", "Unity/Unreal documentation", "best practices", "troubleshooting"],
        "type_safe": True
    },
    "get_project_info": {
        "category": "project_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["project inspection", "understanding current setup", "debugging"],
        "type_safe": True
    },
    "create_asset": {
        "category": "content_creation",
        "cost": "low", 
        "reliability": "very_high",
        "best_for": ["creating scripts", "making prefabs", "new materials", "scenes"],
        "type_safe": True
    },
    "write_file": {
        "category": "file_management",
        "cost": "low",
        "reliability": "very_high", 
        "best_for": ["writing scripts", "config files", "documentation", "shaders"],
        "type_safe": True
    },
    "edit_project_config": {
        "category": "configuration",
        "cost": "low",
        "reliability": "high",
        "best_for": ["build settings", "player settings", "quality settings", "input configuration"],
        "type_safe": True
    },
    "get_script_snippets": {
        "category": "code_assistance",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["code templates", "common patterns", "best practices", "quick implementation"],
        "type_safe": True
    },
    "compile_and_test": {
        "category": "development",
        "cost": "medium",
        "reliability": "high", 
        "best_for": ["testing changes", "checking for errors", "build validation", "deployment prep"],
        "type_safe": True
    },
    "scene_management": {
        "category": "world_building",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["scene creation", "object placement", "level design", "world setup"],
        "type_safe": True
    }
}


def get_available_tool_names() -> List[str]:
    """Get list of available tool names for validation."""
    return [tool.name for tool in TOOLS]


# Testing utilities
def enable_failure_testing():
    """Enable failure testing mode."""
    failure_config.enabled = True
    os.environ["TEST_FAILURES_ENABLED"] = "true"


def disable_failure_testing():
    """Disable failure testing mode."""
    failure_config.enabled = False
    os.environ["TEST_FAILURES_ENABLED"] = "false"


def trigger_specific_failure(failure_type: str):
    """Trigger a specific failure type for testing."""
    os.environ["TRIGGER_FAILURE"] = failure_type


def trigger_tool_failure(tool_name: str, failure_type: str):
    """Trigger failure for a specific tool."""
    os.environ[f"FAIL_{tool_name.upper()}"] = failure_type


def clear_failure_triggers():
    """Clear all failure triggers."""
    failure_vars = [key for key in os.environ if key.startswith("FAIL_") or key == "TRIGGER_FAILURE"]
    for var in failure_vars:
        del os.environ[var]
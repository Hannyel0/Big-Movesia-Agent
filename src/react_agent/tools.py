from __future__ import annotations
from typing import Literal, Optional, Dict, Any, List
import asyncio
import json
from datetime import datetime, UTC

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from react_agent.context import Context


@tool
async def search(query: str) -> Dict[str, Any]:
    """Search for general web results about game development, Unity, Unreal Engine, and related topics.
    
    This function performs a search using the Tavily search engine, which provides
    comprehensive, accurate, and trusted results. Particularly useful for finding
    current game development tutorials, documentation, and best practices.
    
    Args:
        query: The search query string
    """
    try:
        runtime = get_runtime(Context)
        timeout = runtime.context.tool_timeout_seconds
        
        # For testing, simulate successful search results instead of actual web search
        # In production, you would use: wrapped = TavilySearch(max_results=runtime.context.max_search_results)
        
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
        
        return {
            "success": True,
            "result": simulated_results,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
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
    """Get information about the current Unity or Unreal Engine project.
    
    Returns details about the active project including engine version, project structure,
    installed packages, and current scene information.
    """
    # Simulated project info - in reality this would connect to the actual engine
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
        }
    }


@tool
async def create_asset(asset_type: str, name: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new asset in the game project.
    
    Args:
        asset_type: Type of asset to create (script, prefab, material, scene, etc.)
        name: Name for the new asset
        properties: Optional properties/configuration for the asset
    """
    if not properties:
        properties = {}
    
    # Simulated asset creation
    asset_info = {
        "success": True,
        "asset_type": asset_type,
        "name": name,
        "path": f"Assets/{asset_type}s/{name}",
        "created_at": datetime.now(UTC).isoformat(),
        "properties": properties
    }
    
    if asset_type.lower() == "script":
        asset_info.update({
            "language": "C#",
            "template": "MonoBehaviour",
            "methods": ["Start", "Update"],
            "namespaces": ["UnityEngine"]
        })
    elif asset_type.lower() == "prefab":
        asset_info.update({
            "components": ["Transform", "Renderer", "Collider"],
            "children": 0,
            "size": "1.2 MB"
        })
    elif asset_type.lower() == "material":
        asset_info.update({
            "shader": "Universal Render Pipeline/Lit",
            "textures": [],
            "color": "White"
        })
    elif asset_type.lower() == "scene":
        asset_info.update({
            "objects": ["Main Camera", "Directional Light"],
            "lighting": "Realtime",
            "skybox": "Default"
        })
    
    return asset_info


@tool
async def write_file(file_path: str, content: str, file_type: str = "script") -> Dict[str, Any]:
    """Write or create a file in the project (scripts, config files, etc.).
    
    Args:
        file_path: Path where to create/write the file
        content: Content to write to the file
        file_type: Type of file (script, config, text, etc.)
    """
    # Simulated file writing
    return {
        "success": True,
        "file_path": file_path,
        "file_type": file_type,
        "size_bytes": len(content),
        "lines_written": len(content.split('\n')),
        "encoding": "UTF-8",
        "timestamp": datetime.now(UTC).isoformat(),
        "message": f"Successfully wrote {file_type} file to {file_path}"
    }


@tool
async def edit_project_config(config_section: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Edit project configuration settings.
    
    Args:
        config_section: Section of config to modify (build_settings, player_settings, quality, etc.)
        settings: Dictionary of settings to update
    """
    # Simulated config editing
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
        "message": f"Updated {config_section} configuration"
    }


@tool
async def get_script_snippets(category: str, language: str = "csharp") -> Dict[str, Any]:
    """Get code snippets and templates for common game development tasks.
    
    Args:
        category: Category of snippets (player_movement, character_controller, ui, physics, audio, etc.)
        language: Programming language (csharp, javascript, blueprint)
    """
    # Enhanced simulated code snippets database with better category mapping
    snippets = {
        "player_movement": {
            "csharp": {
                "basic_movement": '''public class PlayerMovement : MonoBehaviour 
{
    public float speed = 5f;
    private Rigidbody rb;
    
    void Start() 
    {
        rb = GetComponent<Rigidbody>();
    }
    
    void Update() 
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        
        Vector3 movement = new Vector3(horizontal, 0, vertical) * speed * Time.deltaTime;
        rb.MovePosition(transform.position + movement);
    }
}''',
                "fps_controller": '''public class FPSController : MonoBehaviour 
{
    public float walkSpeed = 6f;
    public float runSpeed = 12f;
    public float mouseSensitivity = 100f;
    
    private CharacterController controller;
    private Camera playerCamera;
    private float xRotation = 0f;
    
    void Start() 
    {
        controller = GetComponent<CharacterController>();
        playerCamera = GetComponentInChildren<Camera>();
        Cursor.lockState = CursorLockMode.Locked;
    }
    
    void Update() 
    {
        HandleMovement();
        HandleMouseLook();
    }
    
    void HandleMovement() 
    {
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");
        bool isRunning = Input.GetKey(KeyCode.LeftShift);
        
        float currentSpeed = isRunning ? runSpeed : walkSpeed;
        Vector3 move = transform.right * x + transform.forward * z;
        controller.Move(move * currentSpeed * Time.deltaTime);
    }
    
    void HandleMouseLook() 
    {
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity * Time.deltaTime;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity * Time.deltaTime;
        
        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, -90f, 90f);
        
        playerCamera.transform.localRotation = Quaternion.Euler(xRotation, 0f, 0f);
        transform.Rotate(Vector3.up * mouseX);
    }
}'''
            }
        },
        # Add character_controller as an alias to player_movement
        "character_controller": {
            "csharp": {
                "basic_character_controller": '''public class CharacterController : MonoBehaviour 
{
    public float speed = 6f;
    public float jumpHeight = 2f;
    public float gravity = -9.81f;
    
    private CharacterController controller;
    private Vector3 velocity;
    private bool isGrounded;
    
    void Start() 
    {
        controller = GetComponent<CharacterController>();
    }
    
    void Update() 
    {
        isGrounded = controller.isGrounded;
        if (isGrounded && velocity.y < 0)
        {
            velocity.y = -2f;
        }
        
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");
        
        Vector3 move = transform.right * x + transform.forward * z;
        controller.Move(move * speed * Time.deltaTime);
        
        if (Input.GetButtonDown("Jump") && isGrounded)
        {
            velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
        }
        
        velocity.y += gravity * Time.deltaTime;
        controller.Move(velocity * Time.deltaTime);
    }
}''',
                "advanced_character_controller": '''public class AdvancedCharacterController : MonoBehaviour 
{
    [Header("Movement")]
    public float walkSpeed = 4f;
    public float runSpeed = 8f;
    public float crouchSpeed = 2f;
    public float jumpHeight = 2f;
    
    [Header("Physics")]
    public float gravity = -9.81f;
    public float groundDistance = 0.4f;
    public LayerMask groundMask;
    
    private CharacterController controller;
    private Vector3 velocity;
    private bool isGrounded;
    private bool isCrouching;
    private Transform groundCheck;
    
    void Start() 
    {
        controller = GetComponent<CharacterController>();
        groundCheck = transform.Find("GroundCheck");
    }
    
    void Update() 
    {
        isGrounded = Physics.CheckSphere(groundCheck.position, groundDistance, groundMask);
        
        if (isGrounded && velocity.y < 0)
        {
            velocity.y = -2f;
        }
        
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");
        
        Vector3 move = transform.right * x + transform.forward * z;
        
        float currentSpeed = GetCurrentSpeed();
        controller.Move(move * currentSpeed * Time.deltaTime);
        
        if (Input.GetButtonDown("Jump") && isGrounded)
        {
            velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
        }
        
        HandleCrouch();
        
        velocity.y += gravity * Time.deltaTime;
        controller.Move(velocity * Time.deltaTime);
    }
    
    float GetCurrentSpeed()
    {
        if (isCrouching) return crouchSpeed;
        if (Input.GetKey(KeyCode.LeftShift)) return runSpeed;
        return walkSpeed;
    }
    
    void HandleCrouch()
    {
        if (Input.GetKeyDown(KeyCode.LeftControl))
        {
            isCrouching = !isCrouching;
            float height = isCrouching ? 1f : 2f;
            controller.height = height;
        }
    }
}'''
            }
        },
        "ui": {
            "csharp": {
                "health_bar": '''public class HealthBar : MonoBehaviour 
{
    public Slider healthSlider;
    public Text healthText;
    public float maxHealth = 100f;
    private float currentHealth;
    
    void Start() 
    {
        currentHealth = maxHealth;
        UpdateHealthUI();
    }
    
    public void TakeDamage(float damage) 
    {
        currentHealth -= damage;
        currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);
        UpdateHealthUI();
    }
    
    public void Heal(float healing) 
    {
        currentHealth += healing;
        currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);
        UpdateHealthUI();
    }
    
    void UpdateHealthUI() 
    {
        healthSlider.value = currentHealth / maxHealth;
        healthText.text = $"{currentHealth:F0} / {maxHealth:F0}";
    }
}''',
                "menu_controller": '''public class MenuController : MonoBehaviour 
{
    public GameObject mainMenuPanel;
    public GameObject settingsPanel;
    public GameObject gameplayPanel;
    
    void Start() 
    {
        ShowMainMenu();
    }
    
    public void ShowMainMenu() 
    {
        mainMenuPanel.SetActive(true);
        settingsPanel.SetActive(false);
        gameplayPanel.SetActive(false);
    }
    
    public void ShowSettings() 
    {
        mainMenuPanel.SetActive(false);
        settingsPanel.SetActive(true);
        gameplayPanel.SetActive(false);
    }
    
    public void StartGame() 
    {
        mainMenuPanel.SetActive(false);
        settingsPanel.SetActive(false);
        gameplayPanel.SetActive(true);
        // Load gameplay scene or initialize game
    }
    
    public void QuitGame() 
    {
        Application.Quit();
    }
}'''
            }
        },
        "audio": {
            "csharp": {
                "audio_manager": '''public class AudioManager : MonoBehaviour 
{
    public AudioSource musicSource;
    public AudioSource sfxSource;
    
    public AudioClip[] musicTracks;
    public AudioClip[] soundEffects;
    
    public static AudioManager Instance;
    
    void Awake() 
    {
        if (Instance == null) 
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        } 
        else 
        {
            Destroy(gameObject);
        }
    }
    
    public void PlayMusic(int trackIndex) 
    {
        if (trackIndex >= 0 && trackIndex < musicTracks.Length) 
        {
            musicSource.clip = musicTracks[trackIndex];
            musicSource.Play();
        }
    }
    
    public void PlaySFX(int effectIndex) 
    {
        if (effectIndex >= 0 && effectIndex < soundEffects.Length) 
        {
            sfxSource.PlayOneShot(soundEffects[effectIndex]);
        }
    }
    
    public void SetMusicVolume(float volume) 
    {
        musicSource.volume = Mathf.Clamp01(volume);
    }
    
    public void SetSFXVolume(float volume) 
    {
        sfxSource.volume = Mathf.Clamp01(volume);
    }
}'''
            }
        },
        # Add physics category
        "physics": {
            "csharp": {
                "rigidbody_movement": '''public class RigidbodyMovement : MonoBehaviour 
{
    public float moveForce = 10f;
    public float maxSpeed = 5f;
    public float jumpForce = 300f;
    
    private Rigidbody rb;
    private bool isGrounded;
    
    void Start() 
    {
        rb = GetComponent<Rigidbody>();
    }
    
    void Update() 
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        
        Vector3 movement = new Vector3(horizontal, 0, vertical) * moveForce;
        
        if (rb.velocity.magnitude < maxSpeed)
        {
            rb.AddForce(movement);
        }
        
        if (Input.GetKeyDown(KeyCode.Space) && isGrounded)
        {
            rb.AddForce(Vector3.up * jumpForce);
        }
    }
    
    void OnCollisionStay(Collision collision)
    {
        isGrounded = true;
    }
    
    void OnCollisionExit(Collision collision)
    {
        isGrounded = false;
    }
}'''
            }
        }
    }
    
    # Normalize category input and check for aliases
    category_lower = category.lower().replace(" ", "_")
    
    # Map common aliases to existing categories
    category_aliases = {
        "character": "character_controller",
        "player": "player_movement",
        "movement": "player_movement",
        "controller": "character_controller",
        "fps": "player_movement",
        "first_person": "player_movement"
    }
    
    # Check if it's an alias
    if category_lower in category_aliases:
        category_lower = category_aliases[category_lower]
    
    category_snippets = snippets.get(category_lower, {})
    language_snippets = category_snippets.get(language, {})
    
    return {
        "success": True,
        "category": category,
        "language": language,
        "available_snippets": list(language_snippets.keys()),
        "snippets": language_snippets,
        "total_snippets": len(language_snippets),
        "message": f"Retrieved {len(language_snippets)} code snippets for {category} in {language}"
    }


@tool
async def compile_and_test(target: str = "editor") -> Dict[str, Any]:
    """Compile the project and run basic tests.
    
    Args:
        target: Compilation target (editor, standalone, mobile, etc.)
    """
    # Simulated compilation process - always successful for testing
    return {
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
        "message": f"Successfully compiled for {target} with {2} warnings and {0} errors"
    }


@tool
async def scene_management(action: str, scene_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Manage scenes in the project (create, load, modify, etc.).
    
    Args:
        action: Action to perform (create, load, save, add_object, remove_object, etc.)
        scene_name: Name of the scene to work with
        parameters: Additional parameters specific to the action
    """
    if not parameters:
        parameters = {}
    
    # Simulated scene management
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
            "message": f"Created new scene: {scene_name}"
        })
    elif action == "load":
        result.update({
            "objects_in_scene": ["Player", "Ground", "UI Canvas", "Main Camera", "Directional Light"],
            "lighting": "Mixed",
            "active_objects": 12,
            "message": f"Loaded scene: {scene_name}"
        })
    elif action == "add_object":
        obj_type = parameters.get("object_type", "GameObject")
        result.update({
            "object_added": obj_type,
            "position": parameters.get("position", [0, 0, 0]),
            "rotation": parameters.get("rotation", [0, 0, 0]),
            "message": f"Added {obj_type} to {scene_name}"
        })
    elif action == "save":
        result.update({
            "changes_saved": True,
            "backup_created": True,
            "message": f"Saved scene: {scene_name}"
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


# Utility function to get available tool names for type validation
def get_available_tool_names() -> List[str]:
    """Get list of available tool names for validation."""
    return [tool.name for tool in TOOLS]
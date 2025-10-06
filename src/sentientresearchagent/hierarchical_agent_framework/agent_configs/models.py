"""
Pydantic Models for Agent Configuration Validation (Pydantic v2)

This module provides comprehensive data models for agent configuration validation,
ensuring type safety and automatic validation for all agent-related configurations.
It integrates with existing types from the framework to avoid duplication.
"""

from typing import Dict, Any, Optional, List, Union, Literal, Type
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pathlib import Path
import os

# Import existing types from the framework
from sentientresearchagent.hierarchical_agent_framework.types import TaskType, TaskStatus, NodeType

# Re-export for convenience
__all__ = [
    "ModelConfig", "ToolConfig", "E2BToolsParams",
    "ToolkitConfig", "ActionKey", "RegistrationConfig",
    "AgnoParams", "AdapterParams", "AgentConfig", "AgentsYAMLConfig",
    "ProfileConfig", "ProfileYAMLConfig", "validate_agent_config",
    "validate_agents_yaml", "validate_profile_yaml", "validate_toolkit_config"
]
# Crypto toolkit params removed - not needed for Sentune (MeetingGenius)

# Model provider types
ModelProviderType = Literal["litellm", "openai", "fireworks", "fireworks_ai", "google", "gemini"]

# Agent types
AgentType = Literal["planner", "executor", "aggregator", "atomizer", "plan_modifier", "custom_search"]

# Market types for toolkits
MarketType = Literal["spot", "usdm", "coinm"]


class ModelConfig(BaseModel):
    """Configuration for LLM model instances."""
    
    model_config = ConfigDict(extra='allow')  # Allow additional model parameters
    
    provider: ModelProviderType = Field(
        ...,
        description="Model provider (litellm, openai, etc.)"
    )
    model_id: str = Field(
        ...,
        description="Model identifier (e.g., 'openai/gpt-4', 'anthropic/claude-3')"
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Model temperature for response randomness"
    )
    max_tokens: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum tokens for model response"
    )
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, gt=0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.0)
    min_p: Optional[float] = Field(None, ge=0.0)
    tfs: Optional[float] = Field(None, ge=0.0)
    typical_p: Optional[float] = Field(None, ge=0.0)
    epsilon_cutoff: Optional[float] = Field(None, ge=0.0)
    eta_cutoff: Optional[float] = Field(None, ge=0.0)
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v):
        """Validate and normalize provider."""
        return v.lower()
    
    @model_validator(mode='after')
    def validate_environment(self):
        """Validate that required environment variables are set."""
        provider = self.provider.lower() if self.provider else ""
        model_id = self.model_id or ""
        
        if provider == "litellm":
            # Check model-specific requirements
            if model_id.startswith("openrouter/"):
                if not os.getenv("OPENROUTER_API_KEY"):
                    raise ValueError(
                        f"OpenRouter model '{model_id}' requires OPENROUTER_API_KEY environment variable"
                    )
            elif model_id.startswith("anthropic/"):
                if not os.getenv("ANTHROPIC_API_KEY"):
                    raise ValueError(
                        f"Anthropic model '{model_id}' requires ANTHROPIC_API_KEY environment variable"
                    )
            elif model_id.startswith("openai/") or model_id.startswith("gpt-"):
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError(
                        f"OpenAI model '{model_id}' requires OPENAI_API_KEY environment variable"
                    )
            elif model_id.startswith("azure/"):
                if not os.getenv("AZURE_API_KEY"):
                    raise ValueError(
                        f"Azure model '{model_id}' requires AZURE_API_KEY environment variable"
                    )
            elif model_id.startswith("fireworks_ai/") or model_id.startswith("fireworks/"):
                if not os.getenv("FIREWORKS_AI_API_KEY"):
                    raise ValueError(
                        f"Fireworks AI model '{model_id}' requires FIREWORKS_AI_API_KEY environment variable"
                    )
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI provider requires OPENAI_API_KEY environment variable")
        elif provider in ["fireworks", "fireworks_ai"]:
            if not os.getenv("FIREWORKS_AI_API_KEY"):
                raise ValueError("Fireworks AI provider requires FIREWORKS_AI_API_KEY environment variable")
        elif provider in ["google", "gemini"]:
            if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                raise ValueError(
                    "Google/Gemini provider requires GOOGLE_API_KEY or GEMINI_API_KEY environment variable"
                )
        
        return self


class ToolConfig(BaseModel):
    """Configuration for individual tools."""
    
    name: str = Field(..., description="Tool name or identifier")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Tool-specific parameters"
    )
    
    @model_validator(mode='after')
    def validate_tool_params(self):
        """Validate tool-specific parameters."""
        tool_name = self.name
        params = self.params
        
        # Special validation for PythonTools
        if tool_name == "PythonTools" and params is not None:
            # Ensure save_and_run is boolean if present
            if "save_and_run" in params and not isinstance(params["save_and_run"], bool):
                raise ValueError("PythonTools 'save_and_run' parameter must be boolean")
        
        return self


# Crypto toolkit params removed - not needed for Sentune (MeetingGenius)
# BaseDataToolkitParams, BinanceToolkitParams, CoingeckoToolkitParams,
# ArkhamToolkitParams, DefiLlamaToolkitParams deleted (273 lines total)


class ToolkitConfig(BaseModel):
    """Configuration for toolkit instances."""
    
    name: str = Field(..., description="Toolkit class name")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Toolkit initialization parameters"
    )
    available_tools: Optional[List[str]] = Field(
        None,
        description="Specific tools to extract from toolkit"
    )

    @classmethod
    def get_toolkit_params_class(cls, name: str) -> Type[BaseModel]:
        """Get the appropriate parameter class for a toolkit name."""
        # Crypto toolkit params removed - return None for all (backward compatibility)
        toolkit_registry = {}
        return toolkit_registry.get(name)
    
    @model_validator(mode='after')
    def validate_toolkit_config(self):
        """Validate toolkit-specific parameters based on toolkit name."""
        name = self.name
        params = self.params or {}
        available_tools = self.available_tools or []
        
        param_class = self.get_toolkit_params_class(name)
        if param_class:
            # Validate using the appropriate toolkit params model
            validated_params = param_class(**params)
            self.params = validated_params.model_dump(exclude_none=True)
            
            # Validate available tools using the class method
            if available_tools:
                valid_tools = param_class.get_valid_tools()
                invalid_tools = [tool for tool in available_tools if tool not in valid_tools]
                if invalid_tools:
                    raise ValueError(
                        f"Invalid {name} tools: {invalid_tools}. "
                        f"Valid tools: {valid_tools}"
                    )
        
        # If toolkit not in registry, just pass through (for backward compatibility)
        
        return self


class ActionKey(BaseModel):
    """Action key configuration for agent registration."""
    
    action_verb: str = Field(..., description="Action verb (plan, execute, etc.)")
    task_type: Optional[str] = Field(None, description="Associated task type as string")
    
    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v):
        """Validate task type string against TaskType enum."""
        if v is not None:
            # Validate against TaskType enum values
            valid_types = [t.value for t in TaskType]
            if v.upper() not in valid_types:
                raise ValueError(f"Invalid task_type: {v}. Must be one of {valid_types}")
        return v


class RegistrationConfig(BaseModel):
    """Agent registration configuration."""
    
    action_keys: Optional[List[ActionKey]] = Field(
        default_factory=list,
        description="Action verb and task type mappings"
    )
    named_keys: Optional[List[str]] = Field(
        default_factory=list,
        description="Named keys for agent lookup"
    )


class AgnoParams(BaseModel):
    """Additional parameters for Agno agent configuration."""
    
    model_config = ConfigDict(extra='allow')  # Allow additional agno parameters
    
    reasoning: Optional[bool] = Field(None, description="Enable reasoning mode")


class AdapterParams(BaseModel):
    """Parameters for adapter configuration."""
    
    model_config = ConfigDict(extra='allow')  # Allow additional adapter-specific parameters
    
    model_id: Optional[str] = Field(None, description="Override model ID")
    search_context_size: Optional[Literal["low", "medium", "high"]] = Field(
        None,
        description="Search context size for custom searchers"
    )
    use_openrouter: Optional[bool] = Field(
        None,
        description="Use OpenRouter instead of direct API"
    )
    num_results: Optional[int] = Field(
        None,
        gt=0,
        description="Number of search results to retrieve"
    )
    include_domains: Optional[List[str]] = Field(
        None,
        description="Domains to include in search"
    )


class AgentConfig(BaseModel):
    """Complete agent configuration model."""
    
    name: str = Field(..., description="Unique agent name")
    type: AgentType = Field(..., description="Agent type classification")
    adapter_class: str = Field(..., description="Adapter class name")
    description: Optional[str] = Field(None, description="Agent description")
    enabled: bool = Field(True, description="Whether agent is enabled")
    
    # Model configuration
    model: Optional[ModelConfig] = Field(None, description="LLM model configuration")
    
    # Prompt configuration
    prompt_source: Optional[str] = Field(
        None,
        description="Prompt source path (e.g., 'prompts.planner_prompts.SYSTEM_MESSAGE')"
    )
    
    # Response model
    response_model: Optional[str] = Field(
        None,
        description="Pydantic model name for structured output"
    )
    
    # Tools configuration
    tools: Optional[List[Union[str, ToolConfig]]] = Field(
        default_factory=list,
        description="Tool configurations"
    )
    toolkits: Optional[List[ToolkitConfig]] = Field(
        default_factory=list,
        description="Toolkit configurations"
    )
    
    # Registration configuration
    registration: Optional[RegistrationConfig] = Field(
        None,
        description="Agent registration configuration"
    )
    
    # Additional parameters
    agno_params: Optional[AgnoParams] = Field(
        None,
        description="Additional Agno agent parameters"
    )
    adapter_params: Optional[AdapterParams] = Field(
        None,
        description="Additional adapter parameters"
    )
    
    @field_validator("adapter_class")
    @classmethod
    def validate_adapter_class(cls, v):
        """Validate adapter class name."""
        valid_adapters = [
            "PlannerAdapter", "ExecutorAdapter", "AtomizerAdapter",
            "AggregatorAdapter", "PlanModifierAdapter",
            "OpenAICustomSearchAdapter", "GeminiCustomSearchAdapter",
            "ExaCustomSearchAdapter",
            "WhisperTranscriptionAdapter", "StandupActionExtractor",
            "StandupBlockerExtractor", "StandupSummaryWriter"
        ]
        if v not in valid_adapters:
            raise ValueError(f"Invalid adapter class: {v}. Must be one of {valid_adapters}")
        return v
    
    @field_validator("tools", mode='before')
    @classmethod
    def normalize_tools(cls, v):
        """Normalize tool configurations to ToolConfig objects."""
        if v is None:
            return []
        
        normalized = []
        for tool in v:
            if isinstance(tool, str):
                # Handle special case for web_search
                normalized.append(tool if tool == "web_search" else ToolConfig(name=tool))
            elif isinstance(tool, dict):
                normalized.append(ToolConfig(**tool))
            else:
                normalized.append(tool)
        return normalized
    
    @model_validator(mode='after')
    def validate_agent_config(self):
        """Cross-field validation for agent configuration."""
        agent_type = self.type
        adapter_class = self.adapter_class
        
        # Validate adapter matches agent type
        type_adapter_map = {
            "planner": ["PlannerAdapter"],
            "executor": ["ExecutorAdapter", "WhisperTranscriptionAdapter", "StandupActionExtractor",
                        "StandupBlockerExtractor", "StandupSummaryWriter"],
            "aggregator": ["AggregatorAdapter"],
            "atomizer": ["AtomizerAdapter"],
            "plan_modifier": ["PlanModifierAdapter"],
            "custom_search": ["OpenAICustomSearchAdapter", "GeminiCustomSearchAdapter", "ExaCustomSearchAdapter"]
        }
        
        valid_adapters = type_adapter_map.get(agent_type, [])
        if adapter_class not in valid_adapters:
            raise ValueError(
                f"Adapter class '{adapter_class}' is not valid for agent type '{agent_type}'. "
                f"Valid adapters: {valid_adapters}"
            )
        
        # Validate custom search agents and meeting adapters don't have model/prompt_source
        custom_adapters_without_model = [
            "OpenAICustomSearchAdapter", "GeminiCustomSearchAdapter", "ExaCustomSearchAdapter",
            "WhisperTranscriptionAdapter"
        ]
        if agent_type == "custom_search" or adapter_class in custom_adapters_without_model:
            if self.model or self.prompt_source:
                raise ValueError(
                    f"Agent '{self.name}' with adapter '{adapter_class}' should not have 'model' or 'prompt_source' fields"
                )
        else:
            # Non-custom search agents should have model and prompt_source
            if not self.model:
                raise ValueError(f"Agent type '{agent_type}' requires 'model' configuration")
            if not self.prompt_source:
                raise ValueError(f"Agent type '{agent_type}' requires 'prompt_source' configuration")
        
        # Validate response models for specific agent types
        if self.response_model:
            response_model_requirements = {
                "planner": ["PlanOutput"],
                "atomizer": ["AtomizerOutput"],
                "plan_modifier": ["PlanOutput"],
                "executor": ["WebSearchResultsOutput", "CustomSearcherOutput", None],  # Various options
                "aggregator": [None],  # Usually no structured output
                "custom_search": ["CustomSearcherOutput", None]
            }
            
            required_models = response_model_requirements.get(agent_type, [None])
            response_model = self.response_model
            if required_models and response_model not in required_models and None not in required_models:
                raise ValueError(
                    f"Agent type '{agent_type}' expects response_model to be one of {required_models}, "
                    f"got '{response_model}'"
                )
        
        return self


class AgentsYAMLConfig(BaseModel):
    """Root configuration for agents.yaml file."""
    
    agents: List[AgentConfig] = Field(..., description="List of agent configurations")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration metadata"
    )
    
    @field_validator("agents")
    @classmethod
    def validate_unique_names(cls, v):
        """Ensure all agent names are unique."""
        names = [agent.name for agent in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate agent names found: {set(duplicates)}")
        return v


class ProfileConfig(BaseModel):
    """Configuration for agent profiles (from profiles/*.yaml)."""
    
    name: str = Field(..., description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    
    # Root-specific agents
    root_planner_adapter_name: Optional[str] = Field(
        None, 
        description="Root planner agent name for initial decomposition"
    )
    root_aggregator_adapter_name: Optional[str] = Field(
        None,
        description="Root aggregator agent name for final synthesis"
    )
    
    # Task type to agent mappings - using strings that will be converted to TaskType
    planner_adapter_names: Optional[Dict[str, str]] = Field(
        None,
        description="Task type to planner agent mappings"
    )
    executor_adapter_names: Optional[Dict[str, str]] = Field(
        None,
        description="Task type to executor agent mappings"
    )
    aggregator_adapter_names: Optional[Dict[str, str]] = Field(
        None,
        description="Task type to aggregator agent mappings"
    )
    
    # Special agents
    atomizer_adapter_name: Optional[str] = Field(None, description="Atomizer agent name")
    aggregator_adapter_name: Optional[str] = Field(
        None, 
        description="Fallback aggregator agent name"
    )
    plan_modifier_adapter_name: Optional[str] = Field(None, description="Plan modifier agent name")
    
    # Default agents
    default_planner_adapter_name: Optional[str] = Field(None, description="Default planner agent name")
    default_executor_adapter_name: Optional[str] = Field(None, description="Default executor agent name")
    default_node_agent_name_prefix: Optional[str] = Field(
        None,
        description="Prefix for node agent names"
    )
    
    @field_validator("planner_adapter_names", "executor_adapter_names", "aggregator_adapter_names")
    @classmethod
    def validate_task_type_keys(cls, v):
        """Validate that task type keys are valid TaskType values."""
        if v is not None:
            valid_types = [t.value for t in TaskType]
            for task_type_str in v.keys():
                if task_type_str.upper() not in valid_types:
                    raise ValueError(
                        f"Invalid task type key '{task_type_str}'. "
                        f"Must be one of {valid_types}"
                    )
        return v


class ProfileYAMLConfig(BaseModel):
    """Root configuration for profile YAML files."""
    
    profile: ProfileConfig = Field(..., description="Profile configuration")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Profile metadata"
    )


# Factory helper functions
def validate_agent_config(config_dict: Dict[str, Any]) -> AgentConfig:
    """Validate and parse agent configuration dictionary."""
    return AgentConfig(**config_dict)


def validate_agents_yaml(config_dict: Dict[str, Any]) -> AgentsYAMLConfig:
    """Validate and parse agents.yaml configuration."""
    return AgentsYAMLConfig(**config_dict)


def validate_profile_yaml(config_dict: Dict[str, Any]) -> ProfileYAMLConfig:
    """Validate and parse profile YAML configuration."""
    return ProfileYAMLConfig(**config_dict)


def validate_toolkit_config(config_dict: Dict[str, Any]) -> ToolkitConfig:
    """Validate and parse toolkit configuration."""
    return ToolkitConfig(**config_dict)
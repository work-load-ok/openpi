"""YAML configuration loader for TrainConfig.

This module provides utilities for loading TrainConfig from YAML files,
with support for nested class instantiation and automatic class registration.

Example YAML format:
```yaml
_target_: TrainConfig
name: my_config
model:
  _target_: Pi0Config
  action_dim: 32
  action_horizon: 50
data:
  _target_: LeRobotAlohaDataConfig
  repo_id: my/dataset
  assets:
    _target_: AssetsConfig
    asset_id: trossen
```

Automatic Registration:
Classes that inherit from registered base classes are automatically discoverable.
You can also use `auto_register_subclasses()` to scan modules and register all
subclasses of specified base classes.
"""

import dataclasses
import enum
import importlib
import inspect
import logging
import pathlib
import pkgutil
from typing import Any, TypeVar, get_type_hints, get_origin, get_args, Union

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Registry mapping short names to fully qualified class paths
_CLASS_REGISTRY: dict[str, str] = {}

# Registry of base classes for automatic subclass discovery
_BASE_CLASS_REGISTRY: dict[str, type] = {}


def register_class(name: str, cls_or_path: type | str) -> None:
    """Register a class with a short name for YAML instantiation.
    
    Args:
        name: Short name to use in YAML (e.g., "Pi0Config")
        cls_or_path: Either a class object or a fully qualified path string
                    (e.g., "openpi.models.pi0_config.Pi0Config")
    """
    if isinstance(cls_or_path, str):
        _CLASS_REGISTRY[name] = cls_or_path
    else:
        _CLASS_REGISTRY[name] = f"{cls_or_path.__module__}.{cls_or_path.__qualname__}"


def register_base_class(name: str, cls: type) -> None:
    """Register a base class for automatic subclass discovery.
    
    When a class name is not found in the registry, the system will search
    for subclasses of registered base classes.
    
    Args:
        name: Name identifier for the base class
        cls: The base class type
    """
    _BASE_CLASS_REGISTRY[name] = cls
    # Also register the base class itself
    register_class(cls.__name__, cls)


def unregister_class(name: str) -> None:
    """Remove a class from the registry.
    
    Args:
        name: The registered name to remove
    """
    _CLASS_REGISTRY.pop(name, None)


def get_registered_classes() -> dict[str, str]:
    """Get a copy of the current class registry.
    
    Returns:
        Dictionary mapping class names to their full module paths
    """
    return dict(_CLASS_REGISTRY)


def is_class_registered(name: str) -> bool:
    """Check if a class name is registered.
    
    Args:
        name: Class name to check
        
    Returns:
        True if registered, False otherwise
    """
    return name in _CLASS_REGISTRY


def _find_subclass_in_modules(
    class_name: str,
    base_classes: list[type],
    search_modules: list[str] | None = None,
) -> type | None:
    """Search for a subclass by name in specified modules.
    
    Args:
        class_name: Name of the class to find
        base_classes: List of base classes to check inheritance against
        search_modules: List of module paths to search. If None, searches common modules.
        
    Returns:
        The found class or None
    """
    if search_modules is None:
        search_modules = [
            "openpi.training.config",
            "openpi.models.pi0_config",
            "openpi.models.pi0_fast",
            "openpi.training.weight_loaders",
            "openpi.training.optimizer",
            "openpi.transforms",
            "openpi.training.droid_rlds_dataset",
        ]
    
    for module_path in search_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                if isinstance(cls, type):
                    # Check if it's a subclass of any base class
                    for base in base_classes:
                        if issubclass(cls, base) and cls is not base:
                            return cls
                    # If no base classes specified or it's a direct match
                    if not base_classes:
                        return cls
        except (ImportError, AttributeError):
            continue
    
    return None


def get_class(name: str, auto_discover: bool = True) -> type:
    """Get a class from its registered name or fully qualified path.
    
    If the class is not in the registry and auto_discover is True,
    attempts to find it as a subclass of registered base classes.
    
    Args:
        name: Either a registered short name or a fully qualified path
        auto_discover: If True, attempt to find unregistered classes
                      by searching for subclasses of known base classes
        
    Returns:
        The class object
        
    Raises:
        ValueError: If the class cannot be found
    """
    # First check registry
    if name in _CLASS_REGISTRY:
        full_path = _CLASS_REGISTRY[name]
        parts = full_path.rsplit(".", 1)
        if len(parts) == 2:
            module_path, class_name = parts
            try:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Cannot load class '{class_name}' from module '{module_path}': {e}") from e
    
    # Check if it's a fully qualified path
    if "." in name:
        parts = name.rsplit(".", 1)
        module_path, class_name = parts
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            # Auto-register for future use
            register_class(name, cls)
            return cls
        except (ImportError, AttributeError):
            pass  # Fall through to auto-discovery
    
    # Try auto-discovery if enabled
    if auto_discover and _BASE_CLASS_REGISTRY:
        base_classes = list(_BASE_CLASS_REGISTRY.values())
        cls = _find_subclass_in_modules(name, base_classes)
        if cls is not None:
            # Auto-register for future use
            register_class(name, cls)
            logger.debug(f"Auto-registered class '{name}' from {cls.__module__}")
            return cls
    
    # Last resort: try to find in common modules without base class check
    if auto_discover:
        cls = _find_subclass_in_modules(name, [])
        if cls is not None:
            register_class(name, cls)
            logger.debug(f"Auto-registered class '{name}' from {cls.__module__}")
            return cls
    
    raise ValueError(
        f"Class '{name}' not found in registry. "
        f"Use register_class('{name}', 'module.path.{name}') to register it, "
        f"or use the full module path in YAML."
    )


def _is_instantiatable(obj: Any) -> bool:
    """Check if an object is a dict with _target_ key (instantiatable config)."""
    return isinstance(obj, dict) and "_target_" in obj


def _unwrap_type(type_hint: Any) -> Any:
    """Unwrap type hints to get the base type.
    
    Handles tyro.conf.Suppress[T], Optional[T], Annotated[T, ...], etc.
    
    Args:
        type_hint: A type hint that may be wrapped
        
    Returns:
        The unwrapped base type
    """
    origin = get_origin(type_hint)
    
    # Handle Optional[T] (which is Union[T, None])
    if origin is Union:
        args = get_args(type_hint)
        # Filter out None type
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _unwrap_type(non_none_args[0])
        return type_hint
    
    # Handle Annotated[T, ...] (which includes tyro.conf.Suppress[T])
    if hasattr(type_hint, "__origin__") and hasattr(type_hint, "__metadata__"):
        # This is an Annotated type
        args = get_args(type_hint)
        if args:
            return _unwrap_type(args[0])
    
    return type_hint


def _resolve_enum_value(value: str, enum_class: type) -> Any:
    """Resolve a string to an enum value.
    
    Handles formats like:
    - "VALUE" -> EnumClass.VALUE
    - "EnumClass.VALUE" -> EnumClass.VALUE
    
    Args:
        value: String representation of the enum value
        enum_class: The enum class to resolve to
        
    Returns:
        The resolved enum value
        
    Raises:
        ValueError: If the value cannot be resolved
    """
    if not isinstance(value, str):
        return value
    
    # Handle "EnumClass.VALUE" format
    if "." in value:
        parts = value.split(".")
        enum_name = parts[-1]
    else:
        enum_name = value
    
    # Try to find the enum value
    try:
        return enum_class[enum_name]
    except KeyError:
        # Try case-insensitive match
        for member in enum_class:
            if member.name.upper() == enum_name.upper():
                return member
        raise ValueError(f"Cannot resolve '{value}' to {enum_class.__name__}. "
                        f"Valid values: {[m.name for m in enum_class]}")


def _convert_value_to_type(value: Any, type_hint: Any) -> Any:
    """Convert a value to match the expected type hint.
    
    Args:
        value: The value to convert
        type_hint: The expected type
        
    Returns:
        The converted value
    """
    if value is None:
        return None
    
    # Unwrap the type first
    base_type = _unwrap_type(type_hint)
    
    # Handle enum types
    if isinstance(base_type, type) and issubclass(base_type, enum.Enum):
        return _resolve_enum_value(value, base_type)
    
    return value


def _get_field_types(cls: type) -> dict[str, Any]:
    """Get type hints for a class's fields, handling dataclasses.
    
    Args:
        cls: The class to get field types for
        
    Returns:
        Dictionary mapping field names to their type hints
    """
    try:
        return get_type_hints(cls)
    except Exception:
        # Fallback for classes where get_type_hints fails
        if dataclasses.is_dataclass(cls):
            return {f.name: f.type for f in dataclasses.fields(cls)}
        return {}


def _instantiate_recursive(config: Any) -> Any:
    """Recursively instantiate objects from config dicts.
    
    Args:
        config: A config value (dict, list, or primitive)
        
    Returns:
        The instantiated object or the original value
    """
    if _is_instantiatable(config):
        # Get the target class
        target = config.pop("_target_")
        cls = get_class(target)
        
        # Get type hints for the class fields
        field_types = _get_field_types(cls)
        
        # Recursively process all arguments
        processed_args = {}
        for key, value in config.items():
            processed_value = _instantiate_recursive(value)
            
            # Try to convert the value to match the expected type
            if key in field_types:
                try:
                    processed_value = _convert_value_to_type(processed_value, field_types[key])
                except Exception as e:
                    logger.warning(f"Could not convert field '{key}' for {cls.__name__}: {e}")
            
            processed_args[key] = processed_value
        
        # Handle special cases for callable fields (like lambdas)
        if "_callable_" in processed_args:
            # This is a placeholder - callable fields need special handling
            callable_config = processed_args.pop("_callable_")
            # For now, we'll skip callable fields
            logger.warning(f"Callable field detected but not supported in YAML: {callable_config}")
        
        # Instantiate the class
        try:
            return cls(**processed_args)
        except TypeError as e:
            raise TypeError(f"Error instantiating {cls.__name__}: {e}") from e
    
    elif isinstance(config, dict):
        # Regular dict - process values recursively
        return {key: _instantiate_recursive(value) for key, value in config.items()}
    
    elif isinstance(config, list):
        # List - process items recursively
        return [_instantiate_recursive(item) for item in config]
    
    else:
        # Primitive value - return as is
        return config


def instantiate(config: dict[str, Any]) -> Any:
    """Instantiate an object from a config dict.
    
    The config dict should have a "_target_" key specifying the class to instantiate.
    Nested dicts with "_target_" keys will be recursively instantiated.
    
    Args:
        config: Configuration dict with "_target_" and other parameters
        
    Returns:
        The instantiated object
    """
    # Make a deep copy to avoid modifying the original
    import copy
    config_copy = copy.deepcopy(config)
    return _instantiate_recursive(config_copy)


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        The parsed YAML content as a dict
    """
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | pathlib.Path) -> Any:
    """Load and instantiate a configuration from a YAML file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        The instantiated configuration object
    """
    config = load_yaml(path)
    return instantiate(config)


def load_configs_from_dir(
    config_dir: str | pathlib.Path,
    pattern: str = "*.yaml",
) -> list[Any]:
    """Load all configuration files from a directory.
    
    Args:
        config_dir: Directory containing YAML config files
        pattern: Glob pattern for config files (default: "*.yaml")
        
    Returns:
        List of instantiated configuration objects
    """
    config_dir = pathlib.Path(config_dir)
    configs = []
    
    for config_path in sorted(config_dir.glob(pattern)):
        try:
            config = load_config(config_path)
            configs.append(config)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    return configs


def save_config_to_yaml(config: Any, path: str | pathlib.Path) -> None:
    """Save a dataclass config to a YAML file.
    
    This converts a config object to YAML format with _target_ annotations.
    
    Args:
        config: A dataclass configuration object
        path: Path to save the YAML file
    """
    yaml_dict = config_to_dict(config)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a dataclass config to a dict with _target_ annotations.
    
    Args:
        config: A dataclass or primitive value
        
    Returns:
        A dict representation suitable for YAML serialization
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        # It's a dataclass instance
        result = {"_target_": type(config).__name__}
        
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            
            # Skip None values and default factory sentinels
            if value is None:
                continue
            
            # Handle special cases
            if callable(value) and not dataclasses.is_dataclass(value):
                # Skip lambda/callable fields - they can't be serialized to YAML
                result[f"# {field.name}"] = "<callable - not serializable>"
                continue
            
            result[field.name] = config_to_dict(value)
        
        return result
    
    elif isinstance(config, dict):
        return {key: config_to_dict(value) for key, value in config.items()}
    
    elif isinstance(config, (list, tuple)):
        return [config_to_dict(item) for item in config]
    
    elif isinstance(config, (str, int, float, bool, type(None))):
        return config
    
    elif hasattr(config, "__class__") and hasattr(config.__class__, "__name__"):
        # For other objects, try to represent them as strings
        return f"<{config.__class__.__name__}>"
    
    else:
        return config


# =============================================================================
# Auto-registration utilities
# =============================================================================

def auto_register_subclasses(
    base_class: type,
    module_paths: list[str] | None = None,
    recursive: bool = True,
) -> list[type]:
    """Automatically discover and register all subclasses of a base class.
    
    Scans specified modules for classes that inherit from the base class
    and registers them automatically.
    
    Args:
        base_class: The base class to find subclasses of
        module_paths: List of module paths to scan. If None, uses default modules.
        recursive: If True, recursively scan submodules
        
    Returns:
        List of registered subclass types
    """
    if module_paths is None:
        module_paths = [
            "openpi.training.config",
            "openpi.models.pi0_config",
            "openpi.models.pi0_fast",
            "openpi.training.weight_loaders",
            "openpi.training.optimizer",
            "openpi.transforms",
            "openpi.training.droid_rlds_dataset",
        ]
    
    registered = []
    
    for module_path in module_paths:
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.debug(f"Could not import {module_path}: {e}")
            continue
        
        # Scan all classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a subclass (but not the base class itself)
            if issubclass(obj, base_class) and obj is not base_class:
                # Only register if defined in this module (not imported)
                if obj.__module__ == module_path or obj.__module__.startswith(module_path + "."):
                    if not is_class_registered(name):
                        register_class(name, obj)
                        registered.append(obj)
                        logger.debug(f"Auto-registered {name} from {obj.__module__}")
        
        # Recursively scan submodules if requested
        if recursive and hasattr(module, "__path__"):
            for _, submodule_name, _ in pkgutil.iter_modules(module.__path__):
                sub_path = f"{module_path}.{submodule_name}"
                registered.extend(
                    auto_register_subclasses(base_class, [sub_path], recursive=True)
                )
    
    return registered


def auto_register_from_base_classes() -> dict[str, list[type]]:
    """Auto-register subclasses of all registered base classes.
    
    Returns:
        Dictionary mapping base class names to lists of registered subclasses
    """
    results = {}
    for name, base_class in _BASE_CLASS_REGISTRY.items():
        registered = auto_register_subclasses(base_class)
        results[name] = registered
        if registered:
            logger.debug(f"Auto-registered {len(registered)} subclasses of {name}")
    return results


# =============================================================================
# Default class registrations
# =============================================================================

def register_default_classes() -> None:
    """Register all default classes used in TrainConfig.
    
    This registers:
    1. Core classes (TrainConfig, DataConfig, AssetsConfig)
    2. Base classes for auto-discovery
    3. Common model, data, optimizer, and transform classes
    """
    # Import base classes for registration
    from openpi.training.config import (
        TrainConfig,
        DataConfig,
        AssetsConfig,
        DataConfigFactory,
    )
    from openpi.models.model import BaseModelConfig
    from openpi.training.weight_loaders import WeightLoader
    from openpi.training.optimizer import LRScheduleConfig, OptimizerConfig
    
    # Register core classes
    register_class("TrainConfig", TrainConfig)
    register_class("DataConfig", DataConfig)
    register_class("AssetsConfig", AssetsConfig)
    
    # Register base classes for auto-discovery
    register_base_class("DataConfigFactory", DataConfigFactory)
    register_base_class("BaseModelConfig", BaseModelConfig)
    register_base_class("WeightLoader", WeightLoader)
    register_base_class("LRScheduleConfig", LRScheduleConfig)
    register_base_class("OptimizerConfig", OptimizerConfig)
    
    # Auto-register all subclasses of base classes
    auto_register_from_base_classes()
    
    # Register additional classes that might not be caught by auto-discovery
    # (e.g., classes from modules not in default search path)
    register_class("Group", "openpi.transforms.Group")
    register_class("RepackTransform", "openpi.transforms.RepackTransform")
    register_class("RLDSDataset", "openpi.training.droid_rlds_dataset.RLDSDataset")

# NOTE: Do NOT call register_default_classes() at module load time!
# This causes issues with tyro CLI parsing because it triggers imports
# and class registrations before tyro can properly analyze the type hints.
# Call register_default_classes() explicitly when needed for YAML loading.

# register_default_classes()  # Disabled to avoid conflicts with tyro

# def register_default_classes_lazy() -> None:
#     """Lazily register default classes using string paths (no imports).
    
#     Use this if you need to avoid importing all modules at startup.
#     Classes will be imported on first use.
#     """
#     # Model configs
#     register_class("Pi0Config", "openpi.models.pi0_config.Pi0Config")
#     register_class("Pi0FASTConfig", "openpi.models.pi0_fast.Pi0FASTConfig")
    
#     # Data configs
#     register_class("TrainConfig", "openpi.training.config.TrainConfig")
#     register_class("DataConfig", "openpi.training.config.DataConfig")
#     register_class("AssetsConfig", "openpi.training.config.AssetsConfig")
#     register_class("FakeDataConfig", "openpi.training.config.FakeDataConfig")
#     register_class("SimpleDataConfig", "openpi.training.config.SimpleDataConfig")
#     register_class("LeRobotAlohaDataConfig", "openpi.training.config.LeRobotAlohaDataConfig")
#     register_class("LeRobotLiberoDataConfig", "openpi.training.config.LeRobotLiberoDataConfig")
#     register_class("LeRobotDROIDDataConfig", "openpi.training.config.LeRobotDROIDDataConfig")
#     register_class("RLDSDroidDataConfig", "openpi.training.config.RLDSDroidDataConfig")
    
#     # Weight loaders
#     register_class("NoOpWeightLoader", "openpi.training.weight_loaders.NoOpWeightLoader")
#     register_class("CheckpointWeightLoader", "openpi.training.weight_loaders.CheckpointWeightLoader")
#     register_class("PaliGemmaWeightLoader", "openpi.training.weight_loaders.PaliGemmaWeightLoader")
    
#     # Optimizer configs
#     register_class("CosineDecaySchedule", "openpi.training.optimizer.CosineDecaySchedule")
#     register_class("RsqrtDecaySchedule", "openpi.training.optimizer.RsqrtDecaySchedule")
#     register_class("AdamW", "openpi.training.optimizer.AdamW")
    
#     # Transform classes
#     register_class("Group", "openpi.transforms.Group")
#     register_class("RepackTransform", "openpi.transforms.RepackTransform")
    
#     # DROID related
#     register_class("RLDSDataset", "openpi.training.droid_rlds_dataset.RLDSDataset")


# Use lazy registration by default to avoid import errors at module load time
# Call register_default_classes() explicitly if you need full auto-discovery
# register_default_classes_lazy()


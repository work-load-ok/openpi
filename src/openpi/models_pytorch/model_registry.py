"""PyTorch Model Registry for dynamic model instantiation.

This module provides a registry system for PyTorch models that allows
dynamic model class selection through configuration files (YAML) or TrainConfig.

Usage:
    1. Register a model class using the decorator:
    
        from openpi.models_pytorch.model_registry import register_pytorch_model
        
        @register_pytorch_model()
        class MyModel(nn.Module):
            def __init__(self, config):
                ...
    
    2. Or register directly:
    
        from openpi.models_pytorch.model_registry import register_pytorch_model_class
        
        register_pytorch_model_class("MyModel", MyModel)
    
    3. Create model instance by name:
    
        from openpi.models_pytorch.model_registry import create_pytorch_model
        
        model = create_pytorch_model("MyModel", config)
    
    4. In YAML config, specify the model class:
    
        pytorch_model_class: "MyModel"
"""

import logging
from typing import Type

from torch import nn

logger = logging.getLogger(__name__)

# =============================================================================
# PyTorch Model Registry
# =============================================================================

# Registry mapping model class names to model classes
_PYTORCH_MODEL_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_pytorch_model(name: str | None = None):
    """Decorator to register a PyTorch model class.
    
    Usage:
        @register_pytorch_model()
        class MyModel(nn.Module):
            ...
            
        # Or with a custom name:
        @register_pytorch_model("CustomName")
        class MyModel(nn.Module):
            ...
    
    Args:
        name: Optional custom name for the model. If not provided,
              the class name will be used.
    
    Returns:
        Decorator function that registers the class.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        model_name = name if name is not None else cls.__name__
        _PYTORCH_MODEL_REGISTRY[model_name] = cls
        logger.debug(f"Registered PyTorch model: {model_name}")
        return cls
    return decorator


def register_pytorch_model_class(name: str, cls: Type[nn.Module]) -> None:
    """Register a PyTorch model class directly.
    
    Args:
        name: Name to register the model under
        cls: The model class
    
    Example:
        from my_models import CustomModel
        register_pytorch_model_class("CustomModel", CustomModel)
    """
    _PYTORCH_MODEL_REGISTRY[name] = cls
    logger.debug(f"Registered PyTorch model: {name}")


def unregister_pytorch_model(name: str) -> None:
    """Remove a model from the registry.
    
    Args:
        name: The registered name to remove
    """
    _PYTORCH_MODEL_REGISTRY.pop(name, None)
    logger.debug(f"Unregistered PyTorch model: {name}")


def get_pytorch_model_class(name: str) -> Type[nn.Module]:
    """Get a PyTorch model class by name.
    
    Args:
        name: The registered model name
        
    Returns:
        The model class
        
    Raises:
        ValueError: If the model is not found in the registry
    """
    if name not in _PYTORCH_MODEL_REGISTRY:
        available = list(_PYTORCH_MODEL_REGISTRY.keys())
        raise ValueError(
            f"PyTorch model '{name}' not found in registry. "
            f"Available models: {available}"
        )
    return _PYTORCH_MODEL_REGISTRY[name]


def get_registered_pytorch_models() -> dict[str, Type[nn.Module]]:
    """Get a copy of the current model registry.
    
    Returns:
        Dictionary mapping model names to model classes
    """
    return dict(_PYTORCH_MODEL_REGISTRY)


def is_pytorch_model_registered(name: str) -> bool:
    """Check if a model name is registered.
    
    Args:
        name: Model name to check
        
    Returns:
        True if registered, False otherwise
    """
    return name in _PYTORCH_MODEL_REGISTRY


def create_pytorch_model(name: str, config, **kwargs) -> nn.Module:
    """Create a PyTorch model instance by name.
    
    Args:
        name: The registered model name
        config: Model configuration object
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        The instantiated model
        
    Example:
        model = create_pytorch_model("PI0Pytorch", model_config)
        model = model.to(device)
    """
    model_cls = get_pytorch_model_class(name)
    return model_cls(config, **kwargs)


def list_pytorch_models() -> list[str]:
    """List all registered model names.
    
    Returns:
        List of registered model names
    """
    return list(_PYTORCH_MODEL_REGISTRY.keys())


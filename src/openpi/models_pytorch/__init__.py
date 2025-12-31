"""PyTorch models for OpenPI.

This module provides PyTorch implementations of PI0 models and a registry
system for dynamic model instantiation.

Model Registry:
    The model registry allows you to register and instantiate models by name,
    which is useful for configuration-driven training.
    
    Example:
        from openpi.models_pytorch import create_pytorch_model, register_pytorch_model
        
        # Create a model by name
        model = create_pytorch_model("PI0Pytorch", config)
        
        # Register a custom model
        @register_pytorch_model()
        class MyCustomModel(PI0Pytorch):
            ...

Available Models:
    - PI0Pytorch: Base PI0 PyTorch implementation
    - PI0Pytorch_Custom: Extended PI0 with value head and custom features
"""

# Re-export model registry functions for convenience
from openpi.models_pytorch.model_registry import (
    register_pytorch_model,
    register_pytorch_model_class,
    unregister_pytorch_model,
    get_pytorch_model_class,
    get_registered_pytorch_models,
    is_pytorch_model_registered,
    create_pytorch_model,
    list_pytorch_models,
)

# Import models to trigger registration
from openpi.models_pytorch import pi0_pytorch  # noqa: F401

__all__ = [
    # Registry functions
    "register_pytorch_model",
    "register_pytorch_model_class",
    "unregister_pytorch_model",
    "get_pytorch_model_class",
    "get_registered_pytorch_models",
    "is_pytorch_model_registered",
    "create_pytorch_model",
    "list_pytorch_models",
    # Model classes (imported from pi0_pytorch)
    "PI0Pytorch",
    "PI0Pytorch_Custom",
]

# Expose model classes directly
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, PI0Pytorch_Custom


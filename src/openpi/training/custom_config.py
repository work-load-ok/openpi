"""Custom configuration module for loading TrainConfig from YAML files.

This module provides utilities for loading TrainConfig from YAML files,
without modifying the original config.py file.

Features:
- Automatic class registration: Subclasses of registered base classes are
  automatically discoverable without explicit registration.
- YAML configuration loading with nested class instantiation
- Mixed Python/YAML config support
- Load specific YAML files without loading entire directories

Usage:
    # Load a single config from YAML
    from openpi.training.custom_config import load_yaml_config
    config = load_yaml_config("configs/train/pi0_aloha.yaml")

    # Load specific YAML files only (not the entire directory)
    from openpi.training.custom_config import load_yaml_configs
    configs = load_yaml_configs(yaml_files=[
        "configs/train/pi0_aloha.yaml",
        "configs/train/pi0_libero.yaml",
    ])

    # Load all configs from a directory
    configs = load_yaml_configs("configs/train")

    # Get config by name from specific file
    from openpi.training.custom_config import get_config
    config = get_config("pi0_aloha", yaml_files="configs/train/pi0_aloha.yaml")

    # Get config from directory
    config = get_config("pi0_aloha", yaml_config_dir="configs/train")

    # CLI with specific YAML file only
    from openpi.training.custom_config import cli
    config = cli(yaml_files="configs/train/my_config.yaml", include_python=False)

    # Enable full auto-discovery of all subclasses
    from openpi.training.custom_config import enable_auto_discovery
    enable_auto_discovery()

Example YAML config:
```yaml
_target_: TrainConfig
name: my_config
model:
  _target_: Pi0Config
  action_dim: 32
data:
  _target_: LeRobotAlohaDataConfig
  repo_id: my/dataset
```

Auto-Registration:
    Classes that inherit from these base classes are automatically registered:
    - DataConfigFactory (for data configs like LeRobotAlohaDataConfig)
    - BaseModelConfig (for model configs like Pi0Config)
    - WeightLoader (for weight loaders like CheckpointWeightLoader)
    - LRScheduleConfig (for learning rate schedules)
    - OptimizerConfig (for optimizers like AdamW)

    Custom classes can be registered using:
    ```python
    from openpi.training.custom_config import register_custom_class
    register_custom_class("MyDataConfig", MyDataConfig)
    ```

    Or by inheriting from a registered base class and enabling auto-discovery:
    ```python
    from openpi.training.custom_config import enable_auto_discovery
    enable_auto_discovery(module_paths=["my_module"])
    ```
"""

import difflib
import logging
import os
import pathlib
from typing import Any

import tyro

# Import from original config module
from openpi.training.config import TrainConfig
from openpi.training.config import _CONFIGS_DICT as PYTHON_CONFIGS

logger = logging.getLogger(__name__)

# Flag to track if full auto-discovery has been enabled
_AUTO_DISCOVERY_ENABLED = False


def load_yaml_configs(
    config_dir: str | pathlib.Path | None = None,
    yaml_files: str | pathlib.Path | list[str | pathlib.Path] | None = None,
) -> list[TrainConfig]:
    """Load TrainConfig objects from YAML files.
    
    You can either:
    1. Load all YAML files from a directory (using config_dir)
    2. Load specific YAML file(s) (using yaml_files)
    3. Load from environment variable OPENPI_CONFIG_DIR (if neither provided)
    
    Args:
        config_dir: Directory containing YAML config files. If None and yaml_files
                   is also None, uses OPENPI_CONFIG_DIR environment variable.
        yaml_files: Specific YAML file(s) to load. Can be:
                   - A single file path (str or Path)
                   - A list of file paths
                   If provided, config_dir is ignored.
    
    Returns:
        List of TrainConfig objects loaded from YAML files.
    
    Example:
        >>> # Load all from directory
        >>> configs = load_yaml_configs("./configs/train")
        
        >>> # Load specific file
        >>> configs = load_yaml_configs(yaml_files="./configs/train/pi0_aloha.yaml")
        
        >>> # Load multiple specific files
        >>> configs = load_yaml_configs(yaml_files=[
        ...     "./configs/train/pi0_aloha.yaml",
        ...     "./configs/train/pi0_libero.yaml",
        ... ])
    """
    from openpi.training import config_loader
    
    # If specific files are provided, load only those
    if yaml_files is not None:
        # Normalize to list
        if isinstance(yaml_files, (str, pathlib.Path)):
            yaml_files = [yaml_files]
        
        configs = []
        for yaml_file in yaml_files:
            yaml_path = pathlib.Path(yaml_file)
            if not yaml_path.exists():
                logger.warning(f"YAML file does not exist: {yaml_path}")
                continue
            try:
                config = config_loader.load_config(yaml_path)
                configs.append(config)
                logger.debug(f"Loaded config from {yaml_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {yaml_path}: {e}")
                raise
        return configs
    
    # Otherwise, load from directory
    if config_dir is None:
        config_dir = os.environ.get("OPENPI_CONFIG_DIR")
        if config_dir is None:
            return []
    
    config_dir = pathlib.Path(config_dir)
    if not config_dir.exists():
        logger.warning(f"Config directory does not exist: {config_dir}")
        return []
    
    return config_loader.load_configs_from_dir(config_dir)


def load_yaml_config(config_path: str | pathlib.Path) -> TrainConfig:
    """Load a single TrainConfig from a YAML file.
    
    Args:
        config_path: Path to the YAML config file.
    
    Returns:
        TrainConfig object loaded from the YAML file.
    
    Example:
        >>> config = load_yaml_config("./configs/train/pi0_aloha.yaml")
        >>> print(config.name)
        'pi0_aloha'
    """
    from openpi.training import config_loader
    return config_loader.load_config(config_path)


def save_config_to_yaml(config: TrainConfig, path: str | pathlib.Path) -> None:
    """Save a TrainConfig to a YAML file.
    
    This is useful for converting existing Python-defined configs to YAML format.
    Note that some fields (like lambdas and complex transforms) cannot be fully
    serialized and will need manual adjustment.
    
    Args:
        config: TrainConfig to save.
        path: Path where the YAML file will be saved.
    
    Example:
        >>> from openpi.training.config import get_config as get_python_config
        >>> config = get_python_config("pi0_aloha")
        >>> save_config_to_yaml(config, "./my_config.yaml")
    """
    from openpi.training import config_loader
    config_loader.save_config_to_yaml(config, path)


def get_yaml_configs_dict(
    yaml_config_dir: str | pathlib.Path | None = None,
    yaml_files: str | pathlib.Path | list[str | pathlib.Path] | None = None,
) -> dict[str, TrainConfig]:
    """Get YAML configs as a dictionary.
    
    Args:
        yaml_config_dir: Directory containing YAML config files. If None,
                        uses OPENPI_CONFIG_DIR env var.
        yaml_files: Specific YAML file(s) to load. If provided, yaml_config_dir
                   is ignored.
    
    Returns:
        Dictionary mapping config names to TrainConfig objects.
    """
    yaml_configs = load_yaml_configs(yaml_config_dir, yaml_files=yaml_files)
    return {config.name: config for config in yaml_configs}


def get_all_configs(
    include_python: bool = True,
    include_yaml: bool = True,
    yaml_config_dir: str | pathlib.Path | None = None,
    yaml_files: str | pathlib.Path | list[str | pathlib.Path] | None = None,
) -> dict[str, TrainConfig]:
    """Get available configs (both Python-defined and YAML-defined).
    
    Args:
        include_python: Whether to include Python-defined configs from config.py.
        include_yaml: Whether to include configs from YAML files.
        yaml_config_dir: Directory containing YAML config files. If None and
                        include_yaml is True, uses OPENPI_CONFIG_DIR env var.
        yaml_files: Specific YAML file(s) to load. If provided, yaml_config_dir
                   is ignored. Can be a single path or list of paths.
    
    Returns:
        Dictionary mapping config names to TrainConfig objects.
    
    Note:
        If a config name exists in both Python and YAML, the YAML version
        takes precedence (with a warning logged).
    
    Example:
        >>> # Load only specific YAML files
        >>> configs = get_all_configs(
        ...     include_python=False,
        ...     yaml_files="configs/train/pi0_aloha.yaml"
        ... )
        
        >>> # Load from directory
        >>> configs = get_all_configs(yaml_config_dir="configs/train")
    """
    configs: dict[str, TrainConfig] = {}
    
    if include_python:
        configs.update(PYTHON_CONFIGS)
    
    if include_yaml:
        yaml_configs = load_yaml_configs(yaml_config_dir, yaml_files=yaml_files)
        for config in yaml_configs:
            if config.name in configs:
                logger.warning(
                    f"YAML config '{config.name}' overrides Python-defined config"
                )
            configs[config.name] = config
    
    return configs


def get_config(
    config_name: str,
    yaml_config_dir: str | pathlib.Path | None = None,
    yaml_files: str | pathlib.Path | list[str | pathlib.Path] | None = None,
    yaml_only: bool = False,
) -> TrainConfig:
    """Get a config by name.
    
    Searches both Python-defined configs (from config.py) and YAML configs.
    YAML configs take precedence if there are name conflicts.
    
    Args:
        config_name: Name of the config to retrieve.
        yaml_config_dir: Optional directory to search for YAML configs.
                        If None, uses OPENPI_CONFIG_DIR env var.
        yaml_files: Specific YAML file(s) to load. If provided, yaml_config_dir
                   is ignored. Can be a single path or list of paths.
        yaml_only: If True, only search YAML configs (ignore Python configs).
    
    Returns:
        The TrainConfig with the specified name.
    
    Raises:
        ValueError: If config_name is not found.
    
    Example:
        >>> # Search in directory
        >>> config = get_config("pi0_aloha", yaml_config_dir="configs/train")
        
        >>> # Load from specific file
        >>> config = get_config("pi0_aloha", yaml_files="configs/train/pi0_aloha.yaml")
        
        >>> # Load from specific files (YAML only, no Python configs)
        >>> config = get_config(
        ...     "pi0_libero",
        ...     yaml_files=["configs/train/pi0_libero.yaml"],
        ...     yaml_only=True
        ... )
    """
    all_configs = get_all_configs(
        include_python=not yaml_only,
        include_yaml=True,
        yaml_config_dir=yaml_config_dir,
        yaml_files=yaml_files,
    )
    
    if config_name not in all_configs:
        closest = difflib.get_close_matches(config_name, all_configs.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return all_configs[config_name]


def cli(
    yaml_config_dir: str | pathlib.Path | None = None,
    yaml_files: str | pathlib.Path | list[str | pathlib.Path] | None = None,
    include_python: bool = True,
) -> TrainConfig:
    """Command-line interface for selecting and configuring TrainConfig.
    
    Supports both Python-defined configs and YAML configs.
    
    Args:
        yaml_config_dir: Optional directory containing YAML configs.
                        If None, uses OPENPI_CONFIG_DIR env var.
        yaml_files: Specific YAML file(s) to load. If provided, yaml_config_dir
                   is ignored.
        include_python: Whether to include Python-defined configs.
    
    Returns:
        The selected and configured TrainConfig.
    
    Example:
        >>> # In your training script:
        >>> from openpi.training.custom_config import cli
        >>> config = cli()
        
        >>> # Load only specific YAML file
        >>> config = cli(yaml_files="configs/train/my_config.yaml", include_python=False)
    """
    all_configs = get_all_configs(
        include_python=include_python,
        yaml_config_dir=yaml_config_dir,
        yaml_files=yaml_files,
    )
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in all_configs.items()})


def list_configs(
    yaml_config_dir: str | pathlib.Path | None = None,
    yaml_files: str | pathlib.Path | list[str | pathlib.Path] | None = None,
    show_source: bool = True,
    include_python: bool = True,
) -> None:
    """Print a list of available configs.
    
    Args:
        yaml_config_dir: Optional directory containing YAML configs.
        yaml_files: Specific YAML file(s) to include. If provided, yaml_config_dir
                   is ignored.
        show_source: If True, show whether each config is from Python or YAML.
        include_python: Whether to include Python-defined configs.
    """
    python_configs = set(PYTHON_CONFIGS.keys()) if include_python else set()
    yaml_configs_dict = get_yaml_configs_dict(yaml_config_dir, yaml_files=yaml_files)
    yaml_configs = set(yaml_configs_dict.keys())
    
    all_names = sorted(python_configs | yaml_configs)
    
    print("Available configs:")
    print("-" * 50)
    
    for name in all_names:
        if show_source:
            sources = []
            if name in python_configs:
                sources.append("Python")
            if name in yaml_configs:
                sources.append("YAML")
            source_str = f" [{', '.join(sources)}]"
        else:
            source_str = ""
        
        print(f"  {name}{source_str}")
    
    print("-" * 50)
    print(f"Total: {len(all_names)} configs")


# =============================================================================
# Convenience functions for working with YAML configs
# =============================================================================

def create_config_from_dict(config_dict: dict[str, Any]) -> TrainConfig:
    """Create a TrainConfig from a dictionary.
    
    The dictionary should have the same structure as a YAML config file,
    with _target_ keys specifying class names.
    
    Args:
        config_dict: Configuration dictionary with _target_ keys.
    
    Returns:
        Instantiated TrainConfig.
    
    Example:
        >>> config_dict = {
        ...     "_target_": "TrainConfig",
        ...     "name": "my_config",
        ...     "model": {
        ...         "_target_": "Pi0Config",
        ...         "action_dim": 32,
        ...     },
        ... }
        >>> config = create_config_from_dict(config_dict)
    """
    from openpi.training import config_loader
    return config_loader.instantiate(config_dict)


def register_custom_class(name: str, cls_or_path: type | str) -> None:
    """Register a custom class for YAML instantiation.
    
    Use this to register your own classes that can be instantiated from YAML.
    
    Args:
        name: Short name to use in YAML (e.g., "MyCustomConfig")
        cls_or_path: Either a class object or a fully qualified path string
                    (e.g., "my_module.MyCustomConfig")
    
    Example:
        >>> from my_module import MyDataConfig
        >>> register_custom_class("MyDataConfig", MyDataConfig)
        
        >>> # Or use a string path:
        >>> register_custom_class("MyDataConfig", "my_module.MyDataConfig")
    """
    from openpi.training import config_loader
    config_loader.register_class(name, cls_or_path)


def register_base_class(name: str, cls: type) -> None:
    """Register a base class for automatic subclass discovery.
    
    When auto-discovery is enabled, all subclasses of this base class
    will be automatically registered.
    
    Args:
        name: Name identifier for the base class
        cls: The base class type
    
    Example:
        >>> from my_module import MyBaseDataConfig
        >>> register_base_class("MyBaseDataConfig", MyBaseDataConfig)
        >>> 
        >>> # Now enable auto-discovery to find all subclasses
        >>> enable_auto_discovery(module_paths=["my_module"])
    """
    from openpi.training import config_loader
    config_loader.register_base_class(name, cls)


def enable_auto_discovery(
    module_paths: list[str] | None = None,
    recursive: bool = True,
) -> dict[str, list[type]]:
    """Enable full auto-discovery of subclasses for all registered base classes.
    
    This scans specified modules and automatically registers all classes that
    inherit from registered base classes (DataConfigFactory, BaseModelConfig, etc.)
    
    Args:
        module_paths: Additional module paths to scan. If None, only scans
                     default openpi modules. Pass your own module paths to
                     include custom classes.
        recursive: If True, recursively scan submodules
    
    Returns:
        Dictionary mapping base class names to lists of newly registered subclasses
    
    Example:
        >>> # Enable with default modules only
        >>> enable_auto_discovery()
        
        >>> # Include custom modules
        >>> enable_auto_discovery(module_paths=["my_project.configs"])
        
        >>> # After enabling, custom subclasses can be used in YAML:
        >>> # _target_: MyCustomDataConfig
    """
    global _AUTO_DISCOVERY_ENABLED
    from openpi.training import config_loader
    
    # First, run the full registration (imports base classes)
    config_loader.register_default_classes()
    
    results = {}
    
    # If additional module paths provided, scan them too
    if module_paths:
        for base_name, base_class in config_loader._BASE_CLASS_REGISTRY.items():
            registered = config_loader.auto_register_subclasses(
                base_class, module_paths, recursive
            )
            if registered:
                results[base_name] = registered
                logger.info(
                    f"Auto-registered {len(registered)} subclasses of {base_name} "
                    f"from {module_paths}"
                )
    
    _AUTO_DISCOVERY_ENABLED = True
    return results


def auto_register_subclasses_of(
    base_class: type,
    module_paths: list[str],
    recursive: bool = True,
) -> list[type]:
    """Automatically register all subclasses of a specific base class.
    
    Args:
        base_class: The base class to find subclasses of
        module_paths: List of module paths to scan
        recursive: If True, recursively scan submodules
    
    Returns:
        List of newly registered subclass types
    
    Example:
        >>> from openpi.training.config import DataConfigFactory
        >>> 
        >>> # Register all DataConfigFactory subclasses in your module
        >>> registered = auto_register_subclasses_of(
        ...     DataConfigFactory,
        ...     ["my_project.data_configs"]
        ... )
        >>> print(f"Registered {len(registered)} data config classes")
    """
    from openpi.training import config_loader
    return config_loader.auto_register_subclasses(base_class, module_paths, recursive)


def get_registered_classes() -> dict[str, str]:
    """Get all currently registered classes.
    
    Returns:
        Dictionary mapping class names to their full module paths
    
    Example:
        >>> classes = get_registered_classes()
        >>> for name, path in sorted(classes.items()):
        ...     print(f"{name}: {path}")
    """
    from openpi.training import config_loader
    return config_loader.get_registered_classes()


def is_auto_discovery_enabled() -> bool:
    """Check if full auto-discovery has been enabled.
    
    Returns:
        True if enable_auto_discovery() has been called
    """
    return _AUTO_DISCOVERY_ENABLED


def export_all_python_configs_to_yaml(
    output_dir: str | pathlib.Path,
    overwrite: bool = False,
) -> None:
    """Export all Python-defined configs to YAML files.
    
    This is useful for migrating from Python configs to YAML configs.
    
    Args:
        output_dir: Directory where YAML files will be saved.
        overwrite: If True, overwrite existing files.
    
    Example:
        >>> export_all_python_configs_to_yaml("configs/exported")
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, config in PYTHON_CONFIGS.items():
        output_path = output_dir / f"{name}.yaml"
        
        if output_path.exists() and not overwrite:
            logger.warning(f"Skipping {output_path} (already exists)")
            continue
        
        try:
            save_config_to_yaml(config, output_path)
            logger.info(f"Exported {name} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export {name}: {e}")


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom config utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available configs")
    list_parser.add_argument("--yaml-dir", type=str, help="YAML config directory")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export Python configs to YAML")
    export_parser.add_argument("output_dir", type=str, help="Output directory")
    export_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get and display a config")
    get_parser.add_argument("name", type=str, help="Config name")
    get_parser.add_argument("--yaml-dir", type=str, help="YAML config directory")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_configs(yaml_config_dir=args.yaml_dir)
    
    elif args.command == "export":
        export_all_python_configs_to_yaml(args.output_dir, overwrite=args.overwrite)
    
    elif args.command == "get":
        config = get_config(args.name, yaml_config_dir=args.yaml_dir)
        print(f"Config: {config.name}")
        print(f"  Model: {type(config.model).__name__}")
        print(f"  Data: {type(config.data).__name__}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Train steps: {config.num_train_steps}")
    
    else:
        parser.print_help()


# YAML Configuration System

This directory contains YAML configuration files for defining `TrainConfig` objects.

## Overview

The YAML configuration system allows you to define training configurations in YAML format
instead of (or in addition to) Python code. This provides several benefits:

- **Easier modification**: Change hyperparameters without editing Python code
- **Better version control**: YAML files are easier to diff and merge
- **Flexibility**: Override specific parameters while keeping defaults
- **Portability**: Share configurations as simple text files

## Directory Structure

```
configs/
├── README.md           # This file
└── train/              # Training configurations
    ├── pi0_aloha.yaml
    ├── pi0_libero.yaml
    ├── pi0_fast_libero.yaml
    ├── pi0_libero_low_mem.yaml
    ├── pi05_libero.yaml
    └── debug.yaml
```

## YAML Format

Each YAML file represents a `TrainConfig` object. Use the `_target_` key to specify
which class to instantiate. If the class is not registered, you need to use the full
module path (e.g., `openpi.training.config.TrainConfig` instead of `TrainConfig`):

```yaml
_target_: TrainConfig
name: my_config

model:
  _target_: Pi0Config
  action_dim: 32
  action_horizon: 50

data:
  _target_: LeRobotAlohaDataConfig
  repo_id: my-org/my-dataset
  assets:
    _target_: AssetsConfig
    asset_id: trossen

weight_loader:
  _target_: CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi0_base/params

num_train_steps: 30000
batch_size: 32
```

## Available Classes

### Model Configs
- `Pi0Config`: Standard Pi0 model configuration
- `Pi0FASTConfig`: Pi0-FAST model configuration

### Data Configs
- `FakeDataConfig`: Fake data for testing
- `LeRobotAlohaDataConfig`: ALOHA robot data
- `LeRobotLiberoDataConfig`: Libero dataset
- `LeRobotDROIDDataConfig`: DROID dataset (LeRobot format)
- `RLDSDroidDataConfig`: DROID dataset (RLDS format)
- `SimpleDataConfig`: Simple data configuration

### Weight Loaders
- `NoOpWeightLoader`: No weight loading (use random initialization)
- `CheckpointWeightLoader`: Load from a checkpoint path
- `PaliGemmaWeightLoader`: Load from official PaliGemma checkpoint

### Optimizer Configs
- `AdamW`: AdamW optimizer
- `CosineDecaySchedule`: Cosine decay learning rate schedule
- `RsqrtDecaySchedule`: Inverse square root decay schedule

### Other
- `AssetsConfig`: Asset configuration (norm stats, etc.)
- `DataConfig`: Base data configuration

## Usage

### Method 1: Environment Variable

Set the `OPENPI_CONFIG_DIR` environment variable:

```bash
export OPENPI_CONFIG_DIR=/path/to/configs/train
python -m openpi.training.train --config my_config
```

### Method 2: Python API (using custom_config module)

```python
from openpi.training.custom_config import load_yaml_config, load_yaml_configs

# Load a single config
config = load_yaml_config("configs/train/pi0_aloha.yaml")

# Load all configs from a directory
configs = load_yaml_configs("configs/train")

# Get config by name (searches both Python and YAML configs)
from openpi.training.custom_config import get_config
config = get_config("pi0_aloha", yaml_config_dir="configs/train")

# Get all available configs
from openpi.training.custom_config import get_all_configs
all_configs = get_all_configs(yaml_config_dir="configs/train")

# List all configs
from openpi.training.custom_config import list_configs
list_configs(yaml_config_dir="configs/train")

# Use CLI with YAML support
from openpi.training.custom_config import cli
config = cli(yaml_config_dir="configs/train")
```

### Method 3: Convert Existing Configs to YAML

```python
from openpi.training.config import get_config as get_python_config
from openpi.training.custom_config import save_config_to_yaml

# Get an existing Python-defined config
config = get_python_config("pi0_aloha")

# Save it as YAML
save_config_to_yaml(config, "my_config.yaml")

# Or export all Python configs at once
from openpi.training.custom_config import export_all_python_configs_to_yaml
export_all_python_configs_to_yaml("configs/exported")
```

### Method 4: Command-line utilities

```bash
# List all available configs
python -m openpi.training.custom_config list --yaml-dir configs/train

# Export Python configs to YAML
python -m openpi.training.custom_config export configs/exported

# Get info about a specific config
python -m openpi.training.custom_config get pi0_aloha --yaml-dir configs/train
```

## Limitations

Some features cannot be fully represented in YAML:

1. **Lambda functions**: Fields like `data_transforms` that use lambda functions
   need to be defined in Python code or using predefined factory classes.

2. **Complex transforms**: Custom transform classes with callable logic should
   be defined in Python and registered with the config loader.

3. **Freeze filters**: The `freeze_filter` field requires NNX filter objects
   which cannot be serialized to YAML directly.

For these cases, you can:
- Define a base config in YAML and override specific fields in Python
- Create custom DataConfigFactory subclasses that handle the complexity
- Use the existing Python configs as templates

## Registering Custom Classes

### Automatic Registration (Recommended)

Classes that inherit from registered base classes are automatically discoverable:

```python
from openpi.training.config import DataConfigFactory

# Your custom class - automatically discoverable!
class MyCustomDataConfig(DataConfigFactory):
    my_param: str = "default"
    
    def create(self, assets_dirs, model_config):
        # ... your implementation
        pass
```

To enable auto-discovery for your custom modules:

```python
from openpi.training.custom_config import enable_auto_discovery

# Enable auto-discovery including your modules
enable_auto_discovery(module_paths=["my_project.configs"])

# Now you can use your class in YAML:
# _target_: MyCustomDataConfig
# my_param: "custom_value"
```

Supported base classes for auto-discovery:
- `DataConfigFactory` - for data configuration classes
- `BaseModelConfig` - for model configuration classes
- `WeightLoader` - for weight loading classes
- `LRScheduleConfig` - for learning rate schedule classes
- `OptimizerConfig` - for optimizer classes

### Manual Registration

For classes that don't inherit from base classes, register them manually:

```python
from openpi.training.custom_config import register_custom_class

# Register with a short name
register_custom_class("MyCustomConfig", "my_module.MyCustomConfig")

# Or register the class directly
from my_module import MyCustomConfig
register_custom_class("MyCustomConfig", MyCustomConfig)
```

### Register Custom Base Classes

To enable auto-discovery for your own base classes:

```python
from openpi.training.custom_config import register_base_class, enable_auto_discovery

# Register your base class
register_base_class("MyBaseConfig", MyBaseConfig)

# Enable auto-discovery for your modules
enable_auto_discovery(module_paths=["my_project"])
```

## Examples

### Basic Training Config

```yaml
_target_: TrainConfig
name: my_training_run

model:
  _target_: Pi0Config

data:
  _target_: LeRobotAlohaDataConfig
  repo_id: my-org/my-dataset

num_train_steps: 10000
batch_size: 16
```

### Fine-tuning with Custom Learning Rate

```yaml
_target_: TrainConfig
name: custom_finetune

model:
  _target_: Pi0Config

data:
  _target_: LeRobotLiberoDataConfig
  repo_id: physical-intelligence/libero

weight_loader:
  _target_: CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi0_base/params

lr_schedule:
  _target_: CosineDecaySchedule
  warmup_steps: 500
  peak_lr: 0.0001
  decay_steps: 20000
  decay_lr: 0.00001

optimizer:
  _target_: AdamW
  weight_decay: 0.01
  clip_gradient_norm: 1.0

num_train_steps: 20000
```

### LoRA Fine-tuning

```yaml
_target_: TrainConfig
name: lora_finetune

model:
  _target_: Pi0Config
  paligemma_variant: gemma_2b_lora
  action_expert_variant: gemma_300m_lora

data:
  _target_: LeRobotLiberoDataConfig
  repo_id: my-org/my-dataset

weight_loader:
  _target_: CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi0_base/params

# Disable EMA for LoRA
ema_decay: null

num_train_steps: 10000
```


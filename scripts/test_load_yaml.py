# 
import tyro
import sys
import logging
from openpi.training.config import _CONFIGS_DICT

from openpi.training import config_loader

def main_custom():
    """Load config from YAML file and allow command-line overrides.
    
    Usage:
        python scripts/train_pytorch.py ./configs/train/TEST_SPLIT_MERGE.yaml --exp_name test --overwrite
    """
    # Find the YAML file path from command line arguments
    yaml_path = None
    remaining_args = []
    
    for arg in sys.argv[1:]:
        if yaml_path is None and arg.endswith('.yaml') and not arg.startswith('-'):
            yaml_path = arg
        else:
            remaining_args.append(arg)
    
    if yaml_path is None:
        raise ValueError(
            "Please provide a YAML config file path as the first argument.\n"
            "Usage: python scripts/train_pytorch.py ./configs/train/YOUR_CONFIG.yaml [--exp_name NAME] [--overwrite]"
        )
    
    # Load the config from YAML file
    base_config = config_loader.load_config(yaml_path)
    logging.info(f"Loaded config from {yaml_path}")
    
    # Apply remaining_args as overrides to base_config using tyro
    # Replace sys.argv to only include the remaining arguments
    sys.argv = [sys.argv[0], base_config.name] + remaining_args
    config = tyro.extras.overridable_config_cli({base_config.name: (base_config.name, base_config)})
    print(config)
    
def main():
    config2 = tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
    print(config2)
if __name__ == "__main__":
    main_custom()
    # main()

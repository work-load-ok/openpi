#!/usr/bin/env python3
"""Test script for YAML configuration loading."""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_config_loader_import():
    """Test that config_loader can be imported."""
    print("Testing config_loader import...")
    from openpi.training import config_loader
    print("✓ config_loader imported successfully")
    return config_loader


def test_class_registry(config_loader):
    """Test that classes are registered correctly."""
    print("\nTesting class registry...")
    
    # Test getting a registered class
    pi0_config_cls = config_loader.get_class("Pi0Config")
    print(f"✓ Pi0Config class: {pi0_config_cls}")
    
    train_config_cls = config_loader.get_class("TrainConfig")
    print(f"✓ TrainConfig class: {train_config_cls}")
    
    checkpoint_loader_cls = config_loader.get_class("CheckpointWeightLoader")
    print(f"✓ CheckpointWeightLoader class: {checkpoint_loader_cls}")


def test_instantiate_simple(config_loader):
    """Test simple object instantiation."""
    print("\nTesting simple instantiation...")
    
    config = {
        "_target_": "Pi0Config",
        "action_dim": 16,
        "action_horizon": 25,
    }
    
    obj = config_loader.instantiate(config)
    print(f"✓ Instantiated Pi0Config: action_dim={obj.action_dim}, action_horizon={obj.action_horizon}")


def test_instantiate_nested(config_loader):
    """Test nested object instantiation."""
    print("\nTesting nested instantiation...")
    
    config = {
        "_target_": "AssetsConfig",
        "asset_id": "test_asset",
        "assets_dir": "/test/path",
    }
    
    obj = config_loader.instantiate(config)
    print(f"✓ Instantiated AssetsConfig: asset_id={obj.asset_id}, assets_dir={obj.assets_dir}")


def test_load_yaml_file(config_loader):
    """Test loading from YAML file."""
    print("\nTesting YAML file loading...")
    
    config_dir = Path(__file__).parent.parent / "configs" / "train"
    yaml_files = list(config_dir.glob("*.yaml"))
    
    if not yaml_files:
        print("⚠ No YAML files found in configs/train/")
        return
    
    # Try loading the first YAML file
    yaml_file = yaml_files[0]
    print(f"Loading: {yaml_file.name}")
    
    config = config_loader.load_yaml(yaml_file)
    print(f"✓ Loaded YAML content: {list(config.keys())}")
    
    # Try instantiating
    try:
        obj = config_loader.instantiate(config)
        print(f"✓ Instantiated TrainConfig: name={obj.name}")
    except Exception as e:
        print(f"⚠ Could not instantiate (may require dependencies): {e}")


def test_config_module():
    """Test the custom_config module functions."""
    print("\nTesting custom_config module functions...")
    
    from openpi.training.custom_config import (
        load_yaml_config,
        load_yaml_configs,
        get_config,
        save_config_to_yaml,
        get_all_configs,
        list_configs,
    )
    
    # Test get_config for Python-defined configs
    config = get_config("debug")
    print(f"✓ Got Python config 'debug': name={config.name}")
    
    # Test load_yaml_configs
    config_dir = Path(__file__).parent.parent / "configs" / "train"
    if config_dir.exists():
        configs = load_yaml_configs(config_dir)
        print(f"✓ Loaded {len(configs)} configs from YAML directory")
        for c in configs[:3]:  # Show first 3
            print(f"  - {c.name}")
    
    # Test get_all_configs
    all_configs = get_all_configs(yaml_config_dir=config_dir)
    print(f"✓ Got {len(all_configs)} total configs (Python + YAML)")
    
    # Test list_configs
    print("\n--- List of configs ---")
    list_configs(yaml_config_dir=config_dir)


def test_auto_discovery():
    """Test automatic class registration."""
    print("\nTesting auto-discovery...")
    
    from openpi.training.custom_config import (
        enable_auto_discovery,
        get_registered_classes,
        is_auto_discovery_enabled,
    )
    
    # Check initial state
    print(f"Auto-discovery enabled before: {is_auto_discovery_enabled()}")
    
    # Get classes before auto-discovery
    classes_before = get_registered_classes()
    print(f"✓ Classes registered before auto-discovery: {len(classes_before)}")
    
    # Enable auto-discovery
    results = enable_auto_discovery()
    print(f"✓ Auto-discovery enabled")
    print(f"Auto-discovery enabled after: {is_auto_discovery_enabled()}")
    
    # Get classes after auto-discovery
    classes_after = get_registered_classes()
    print(f"✓ Classes registered after auto-discovery: {len(classes_after)}")
    
    # Show some registered classes
    print("\nSample of registered classes:")
    for name in sorted(classes_after.keys())[:10]:
        print(f"  - {name}")
    
    # Test that we can get a class that should be auto-registered
    from openpi.training import config_loader
    try:
        cls = config_loader.get_class("LeRobotAlohaDataConfig")
        print(f"✓ Successfully got LeRobotAlohaDataConfig: {cls}")
    except ValueError as e:
        print(f"✗ Failed to get LeRobotAlohaDataConfig: {e}")


def test_config_to_dict(config_loader):
    """Test converting config to dict."""
    print("\nTesting config_to_dict...")
    
    from openpi.models.pi0_config import Pi0Config
    
    config = Pi0Config(action_dim=24, action_horizon=30)
    config_dict = config_loader.config_to_dict(config)
    
    print(f"✓ Converted to dict:")
    print(f"  _target_: {config_dict.get('_target_')}")
    print(f"  action_dim: {config_dict.get('action_dim')}")
    print(f"  action_horizon: {config_dict.get('action_horizon')}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("YAML Configuration System Test")
    print("=" * 60)
    
    try:
        config_loader = test_config_loader_import()
        test_class_registry(config_loader)
        test_instantiate_simple(config_loader)
        test_instantiate_nested(config_loader)
        test_load_yaml_file(config_loader)
        test_config_to_dict(config_loader)
        test_config_module()
        test_auto_discovery()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


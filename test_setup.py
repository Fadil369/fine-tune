#!/usr/bin/env python3
"""
Test script to verify the fine-tuning framework setup
"""

import sys
import subprocess
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} missing: {filepath}")
        return False


def check_python_syntax(filepath):
    """Check Python file syntax"""
    try:
        subprocess.run(
            ["python", "-m", "py_compile", filepath],
            check=True,
            capture_output=True
        )
        print(f"✓ Valid Python syntax: {filepath}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Invalid Python syntax: {filepath}")
        return False


def check_yaml_syntax(filepath):
    """Check YAML file syntax"""
    try:
        import yaml
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        print(f"✓ Valid YAML syntax: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Invalid YAML syntax: {filepath}")
        print(f"  Error: {e}")
        return False


def main():
    print("="*60)
    print("Fine-Tune Framework Setup Verification")
    print("="*60)
    print()
    
    all_checks_passed = True
    
    # Check core Python files
    print("Checking Python Files...")
    python_files = [
        "finetune.py",
        "prepare_data.py",
        "inference.py",
        "evaluate.py",
    ]
    
    for pf in python_files:
        if not check_file_exists(pf, "Python file"):
            all_checks_passed = False
            continue
        if not check_python_syntax(pf):
            all_checks_passed = False
    
    print()
    
    # Check configuration files
    print("Checking Configuration Files...")
    yaml_files = [
        "config.yaml",
        "examples/config_instruction.yaml",
        "examples/config_classification.yaml",
    ]
    
    for yf in yaml_files:
        if not check_file_exists(yf, "YAML file"):
            all_checks_passed = False
            continue
        if not check_yaml_syntax(yf):
            all_checks_passed = False
    
    print()
    
    # Check other required files
    print("Checking Other Files...")
    other_files = [
        ("requirements.txt", "Requirements file"),
        ("README.md", "README"),
        ("TUTORIAL.md", "Tutorial"),
        ("CONTRIBUTING.md", "Contributing guide"),
        (".gitignore", "Gitignore"),
        ("Makefile", "Makefile"),
        ("quickstart.sh", "Quickstart script"),
    ]
    
    for filepath, description in other_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    print()
    
    # Check examples
    print("Checking Example Files...")
    example_files = [
        "examples/sample_data.json",
        "examples/instruction_data.json",
        "examples/README.md",
    ]
    
    for ef in example_files:
        if not check_file_exists(ef, "Example file"):
            all_checks_passed = False
    
    print()
    
    # Check Python imports (optional)
    print("Checking Python Dependencies (optional)...")
    optional_imports = [
        "yaml",
        "torch",
        "transformers",
    ]
    
    dependencies_available = True
    for module in optional_imports:
        try:
            __import__(module)
            print(f"✓ Python module available: {module}")
        except ImportError:
            print(f"! Python module not installed: {module}")
            dependencies_available = False
    
    if not dependencies_available:
        print("\nNote: Some dependencies are not installed.")
        print("Install with: pip install -r requirements.txt")
        print("(This is expected if you haven't run installation yet)")
    
    print()
    print("="*60)
    
    if all_checks_passed:
        print("✓ All checks passed! Setup is complete.")
        print()
        print("Next steps:")
        print("1. Prepare your data with prepare_data.py")
        print("2. Edit config.yaml for your use case")
        print("3. Run: python finetune.py --config config.yaml")
        print()
        print("See TUTORIAL.md for detailed instructions.")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

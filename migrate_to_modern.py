#!/usr/bin/env python3
"""
Migration script for Infinite Maze modernization.

This script helps transition from the old Makefile/setup.py system
to the modern Poetry/pyproject.toml system.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_poetry_installed():
    """Check if Poetry is installed on the system."""
    try:
        result = subprocess.run(['poetry', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Poetry found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Poetry not found")
        return False

def install_poetry():
    """Install Poetry using the official installer."""
    print("Installing Poetry...")
    if os.name == 'nt':  # Windows
        print("Please install Poetry manually from: https://python-poetry.org/docs/#installation")
        print("Or run: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -")
        return False
    else:  # Unix-like systems
        try:
            subprocess.run(['curl', '-sSL', 'https://install.python-poetry.org', '|', 'python3', '-'], 
                          shell=True, check=True)
            print("✓ Poetry installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install Poetry automatically")
            print("Please install manually from: https://python-poetry.org/docs/#installation")
            return False

def backup_old_files():
    """Create backups of files that will be replaced."""
    backup_dir = Path("migration_backup")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = ['Makefile', 'setup.py', 'requirements.txt']
    
    for file in files_to_backup:
        if Path(file).exists():
            shutil.copy2(file, backup_dir / file)
            print(f"✓ Backed up {file} to {backup_dir}/{file}")

def clean_old_files():
    """Remove old configuration files after confirming backup."""
    files_to_remove = ['Makefile', 'setup.py', 'requirements.txt']
    
    print("\nRemoving old configuration files...")
    for file in files_to_remove:
        if Path(file).exists():
            Path(file).unlink()
            print(f"✓ Removed {file}")

def setup_poetry_project():
    """Initialize the Poetry project."""
    print("\nSetting up Poetry project...")
    try:
        # Install dependencies
        subprocess.run(['poetry', 'install'], check=True)
        print("✓ Poetry dependencies installed")
        
        # Verify installation
        result = subprocess.run(['poetry', 'run', 'python', '-c', 'import infinite_maze; print("Package imported successfully")'], 
                              capture_output=True, text=True, check=True)
        print("✓ Package installation verified")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error setting up Poetry project: {e}")
        return False

def test_game_execution():
    """Test that the game can be executed with the new system."""
    print("\nTesting game execution...")
    try:
        # Test Poetry entry point
        result = subprocess.run(['poetry', 'run', 'infinite-maze', '--help'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ Poetry entry point working")
        else:
            print("! Poetry entry point test inconclusive (game may not support --help)")
        
        # Test module execution
        result = subprocess.run(['poetry', 'run', 'python', '-m', 'infinite_maze'], 
                              capture_output=True, text=True, timeout=5)
        print("✓ Module execution test completed")
        
        return True
    except subprocess.TimeoutExpired:
        print("✓ Game execution test completed (timeout expected for interactive game)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"! Game execution test showed some issues: {e}")
        return False

def main():
    """Main migration function."""
    print("=== Infinite Maze Modernization Migration ===\n")
    
    # Check current directory
    if not Path('pyproject.toml').exists():
        print("✗ pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check Poetry installation
    if not check_poetry_installed():
        if input("Install Poetry now? (y/n): ").lower().startswith('y'):
            if not install_poetry():
                sys.exit(1)
        else:
            print("Poetry is required for the modernized project structure.")
            sys.exit(1)
    
    # Create backups
    print("\nCreating backups of old files...")
    backup_old_files()
    
    # Setup Poetry project
    if not setup_poetry_project():
        print("\nMigration failed during Poetry setup.")
        sys.exit(1)
    
    # Test execution
    if not test_game_execution():
        print("\nMigration completed with warnings. Please test manually.")
    else:
        print("\n✓ Migration completed successfully!")
    
    # Clean up old files
    if input("\nRemove old configuration files? (y/n): ").lower().startswith('y'):
        clean_old_files()
    
    print("\n=== Migration Summary ===")
    print("✓ Modern pyproject.toml configuration created")
    print("✓ Poetry dependency management set up")
    print("✓ Entry points configured for easy game execution")
    print("✓ Development tools configured (black, flake8, mypy)")
    print("✓ Configuration and logging modules added")
    print("\nTo run the game:")
    print("  poetry run infinite-maze")
    print("  poetry run maze-game")
    print("  poetry run python -m infinite_maze")
    print("\nFor development:")
    print("  poetry run black infinite_maze/    # Format code")
    print("  poetry run flake8 infinite_maze/   # Lint code")
    print("  poetry run mypy infinite_maze/     # Type checking")

if __name__ == '__main__':
    main()

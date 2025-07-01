# Infinite Maze - Installation & Setup Guide

This comprehensive guide will help you install and set up Infinite Maze on any supported platform, whether you're a player wanting to enjoy the game or a developer looking to contribute.

## 🎯 Quick Start (For Players)

### One-Command Installation
The fastest way to get started:

**Windows:**
```cmd
setup.bat
```

**Linux/macOS:**
```bash
./setup.sh
```

These scripts will:
- Check if Poetry is installed
- Guide you through Poetry installation if needed
- Install all game dependencies
- Launch the game automatically

---

## 🖥️ System Requirements

### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 512 MB available memory
- **Storage**: 50 MB free disk space
- **Display**: 800x600 resolution minimum

### Recommended Requirements
- **Operating System**: Windows 11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or higher
- **RAM**: 1 GB available memory
- **Storage**: 100 MB free disk space
- **Display**: 1920x1080 resolution or higher

### Supported Python Versions
- Python 3.8 ✅
- Python 3.9 ✅
- Python 3.10 ✅ (Recommended)
- Python 3.11 ✅
- Python 3.12 ✅

---

## 📦 Installation Methods

### Method 1: Automated Setup (Recommended)

This is the easiest method for most users.

#### Windows
1. Download or clone the Infinite Maze repository
2. Open Command Prompt or PowerShell in the project directory
3. Run the setup script:
   ```cmd
   setup.bat
   ```
4. Follow the on-screen instructions
5. The game will launch automatically when setup is complete

#### Linux/macOS
1. Download or clone the Infinite Maze repository
2. Open Terminal in the project directory
3. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
4. Run the setup script:
   ```bash
   ./setup.sh
   ```
5. Follow the on-screen instructions
6. The game will launch automatically when setup is complete

### Method 2: Manual Installation with Poetry

If you prefer more control over the installation process:

#### Step 1: Install Poetry
Poetry is our recommended dependency manager for Python projects.

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

**Linux/macOS:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Alternative (using pip):**
```bash
pip install poetry
```

#### Step 2: Install Game Dependencies
```bash
# Navigate to the project directory
cd infinite-maze

# Install dependencies
poetry install

# Run the game
poetry run infinite-maze
```

### Method 3: Traditional pip Installation

For users who prefer not to use Poetry:

#### Step 1: Create Virtual Environment (Optional but Recommended)
```bash
# Create virtual environment
python -m venv infinite-maze-env

# Activate virtual environment
# Windows:
infinite-maze-env\Scripts\activate
# Linux/macOS:
source infinite-maze-env/bin/activate
```

#### Step 2: Install the Game
```bash
# Install in development mode
pip install -e .

# Run the game
infinite-maze
```

---

## 🚀 Running the Game

Once installed, you can run Infinite Maze in several ways:

### Primary Methods
```bash
# Modern entry point (recommended)
poetry run infinite-maze

# Alternative entry point
poetry run maze-game

# Python module execution
python -m infinite_maze
```

### Legacy Methods (Still Supported)
```bash
# Direct script execution
python run_game.py

# If installed with pip
infinite-maze
```

---

## 🛠️ Development Setup

For contributors and developers who want to modify the game:

### Step 1: Clone the Repository
```bash
git clone https://github.com/1024Adam/infinite-maze.git
cd infinite-maze
```

### Step 2: Install Development Dependencies
```bash
# Install with development tools
poetry install

# Verify installation
poetry run infinite-maze
```

### Step 3: Set Up Development Tools
```bash
# Code formatting
poetry run black infinite_maze/

# Code linting
poetry run flake8 infinite_maze/

# Type checking
poetry run mypy infinite_maze/
```

### Step 4: Optional Task Runner Setup

#### Using Just (Recommended)
Just is a modern task runner that simplifies development commands:

1. **Install Just:**
   - **Windows (Chocolatey):** `choco install just`
   - **macOS (Homebrew):** `brew install just`
   - **Linux (Cargo):** `cargo install just`

2. **Use Just commands:**
   ```bash
   just setup    # Setup project
   just run      # Run the game
   just format   # Format code
   just check    # Run all quality checks
   just test     # Run tests
   ```

---

## 🔧 Troubleshooting

### Common Installation Issues

#### Problem: "Python not found" or "python: command not found"
**Solution:**
1. Verify Python is installed: `python --version` or `python3 --version`
2. On some systems, use `python3` instead of `python`
3. Add Python to your system PATH if necessary

#### Problem: "Poetry not found" after installation
**Solution:**
1. Close and reopen your terminal
2. Add Poetry to your PATH manually:
   - **Windows:** Add `%APPDATA%\Python\Scripts` to PATH
   - **Linux/macOS:** Add `~/.local/bin` to PATH
3. Restart your terminal/command prompt

#### Problem: Permission denied errors
**Solution:**
1. **Windows:** Run Command Prompt as Administrator
2. **Linux/macOS:** Use `sudo` if necessary, or check file permissions
3. Ensure you have write access to the installation directory

#### Problem: "No module named 'pygame'" errors
**Solution:**
1. Ensure you're using the correct Python environment
2. Reinstall dependencies: `poetry install` or `pip install pygame`
3. Verify pygame installation: `python -c \"import pygame; print(pygame.version.ver)\"`

#### Problem: Game launches but displays incorrectly
**Solution:**
1. Update your graphics drivers
2. Try running in windowed mode
3. Check display resolution and scaling settings
4. Ensure your system meets minimum requirements

### Platform-Specific Issues

#### Windows
- **Issue:** PowerShell execution policy errors
- **Solution:** Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### macOS
- **Issue:** "Developer cannot be verified" warnings
- **Solution:** Right-click the app and select "Open" from the context menu

#### Linux
- **Issue:** Missing development headers for pygame compilation
- **Solution:** Install development packages:
  ```bash
  # Ubuntu/Debian
  sudo apt install python3-dev libsdl2-dev
  
  # Fedora/CentOS
  sudo yum install python3-devel SDL2-devel
  ```

---

## 🎮 First Launch

### What to Expect
1. **Game Window**: An 800x600 window will open with the maze game
2. **Initial Maze**: A randomly generated maze will appear
3. **Player Character**: A red dot representing your character
4. **UI Elements**: Score and time displays at the top

### First-Time Tips
1. **Test Controls**: Try all movement keys (WASD or arrows)
2. **Practice Pausing**: Press Space to pause/unpause
3. **Learn the Interface**: Familiarize yourself with the score and time displays
4. **Exit Safely**: Use Esc or Q to quit cleanly

### If Something Goes Wrong
1. **Game Won't Start:** Check the terminal/command prompt for error messages
2. **Controls Don't Work:** Ensure the game window has focus (click on it)
3. **Performance Issues:** Close other applications to free up resources
4. **Display Problems:** Try adjusting your display settings

---

## 🔄 Updating the Game

### For Poetry Users
```bash
# Pull latest changes
git pull origin main

# Update dependencies
poetry install

# Run the updated game
poetry run infinite-maze
```

### For pip Users
```bash
# Pull latest changes
git pull origin main

# Reinstall the game
pip install -e .

# Run the updated game
infinite-maze
```

---

## 🗑️ Uninstallation

### Complete Removal
1. **Delete the game directory:**
   ```bash
   rm -rf infinite-maze/  # Linux/macOS
   rmdir /s infinite-maze  # Windows
   ```

2. **Remove Poetry environment (if using Poetry):**
   ```bash
   poetry env remove python
   ```

3. **Remove pip installation (if used):**
   ```bash
   pip uninstall infinite-maze
   ```

### Keep Configuration
If you want to keep your configuration files for future reinstallation, only delete the main game files and leave any config directories intact.

---

## 📞 Getting Help

### Self-Help Resources
1. **Check this guide** for common solutions
2. **Review error messages** carefully for clues
3. **Verify system requirements** are met
4. **Try different installation methods** if one fails

### Community Support
1. **GitHub Issues:** Report bugs and technical problems
2. **Documentation:** Check other guides in the `docs/` directory
3. **Discord/Forums:** Join community discussions (if available)

### Reporting Issues
When reporting installation problems, please include:
- Operating system and version
- Python version (`python --version`)
- Complete error messages
- Steps you've already tried
- Any relevant system information

---

**Congratulations!** You should now have Infinite Maze successfully installed and running. If you encountered any issues not covered in this guide, please don't hesitate to reach out for support. Enjoy the game! 🎮

# Infinite Maze - Troubleshooting Guide

This guide helps resolve common issues that players and developers might encounter when installing, running, or developing Infinite Maze.

## 🎯 Quick Issue Resolution

### Game Won't Start
1. Check Python version: `python --version` (need 3.8+)
2. Verify installation: `poetry run infinite-maze` or `python -m infinite_maze`
3. Check dependencies: `poetry install`
4. Try alternative methods: `python run_game.py`

### Performance Issues
1. Close other applications to free memory
2. Update graphics drivers
3. Check system requirements
4. Try windowed mode if using fullscreen

### Control Problems
1. Ensure game window has focus (click on it)
2. Try alternative keys (WASD vs Arrow keys)
3. Check for stuck keys
4. Restart the game

---

## 🖥️ Installation Issues

### Python Installation Problems

#### Problem: "Python not found" or "python: command not found"
**Platforms**: Windows, macOS, Linux

**Symptoms:**
- Terminal/Command Prompt can't find Python
- Commands fail with "not recognized" or "command not found"

**Solutions:**
1. **Verify Python Installation:**
   ```bash
   python --version
   python3 --version  # Try this on macOS/Linux
   ```

2. **Install Python if Missing:**
   - **Windows**: Download from [python.org](https://python.org) and check "Add to PATH"
   - **macOS**: Use Homebrew: `brew install python`
   - **Linux**: Use package manager: `sudo apt install python3` (Ubuntu)

3. **Fix PATH Issues:**
   - **Windows**: Add Python installation directory to PATH environment variable
   - **macOS/Linux**: Add Python directory to your shell profile (.bashrc, .zshrc)

4. **Use Alternative Commands:**
   ```bash
   py --version        # Windows Python Launcher
   python3 --version   # Explicit Python 3
   ```

#### Problem: "Permission denied" during installation
**Platforms**: macOS, Linux

**Symptoms:**
- Installation fails with permission errors
- Cannot write to directories

**Solutions:**
1. **Use Virtual Environment:**
   ```bash
   python -m venv infinite-maze-env
   source infinite-maze-env/bin/activate  # Linux/macOS
   infinite-maze-env\Scripts\activate     # Windows
   ```

2. **User Installation:**
   ```bash
   pip install --user -e .
   ```

3. **Fix Permissions:**
   ```bash
   # Make sure you own the directory
   sudo chown -R $USER /path/to/infinite-maze
   ```

### Poetry Installation Problems

#### Problem: "Poetry not found" after installation
**Platforms**: All

**Symptoms:**
- Poetry commands fail after installation
- Terminal doesn't recognize poetry command

**Solutions:**
1. **Restart Terminal:**
   Close and reopen terminal/command prompt

2. **Check Installation Path:**
   ```bash
   # Check if Poetry is in PATH
   echo $PATH  # Linux/macOS
   echo %PATH% # Windows
   ```

3. **Manual PATH Addition:**
   - **Windows**: Add `%APPDATA%\Python\Scripts` to PATH
   - **macOS/Linux**: Add `~/.local/bin` to PATH

4. **Alternative Installation:**
   ```bash
   pip install poetry
   ```

5. **Use Full Path:**
   ```bash
   ~/.local/bin/poetry install  # Linux/macOS
   %APPDATA%\Python\Scripts\poetry install  # Windows
   ```

### Dependency Installation Issues

#### Problem: Pygame installation fails
**Platforms**: All

**Symptoms:**
- Error messages during pygame installation
- "Failed building wheel for pygame"
- Missing SDL libraries

**Solutions:**

**Windows:**
```bash
# Update pip first
python -m pip install --upgrade pip

# Install pygame
pip install pygame

# If that fails, try pre-compiled wheel
pip install pygame --only-binary=all
```

**macOS:**
```bash
# Install dependencies with Homebrew
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf

# Then install pygame
pip install pygame
```

**Linux (Ubuntu/Debian):**
```bash
# Install development headers
sudo apt update
sudo apt install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev

# Install pygame
pip install pygame
```

**Linux (Fedora/CentOS):**
```bash
# Install development packages
sudo yum install python3-devel SDL2-devel SDL2_image-devel SDL2_mixer-devel SDL2_ttf-devel

# Install pygame
pip install pygame
```

---

## 🎮 Runtime Issues

### Game Performance Problems

#### Problem: Game runs slowly or stutters
**Symptoms:**
- Low frame rate
- Jerky movement
- Input lag

**Solutions:**
1. **Close Background Applications:**
   - Close unnecessary programs
   - Check Task Manager/Activity Monitor for high CPU usage

2. **Update Graphics Drivers:**
   - **Windows**: Device Manager → Display adapters → Update driver
   - **macOS**: System updates usually include driver updates
   - **Linux**: Install proprietary drivers if using NVIDIA/AMD

3. **Adjust Game Settings:**
   ```python
   # In config.py, try reducing:
   FPS = 30  # Instead of 60
   SCREEN_WIDTH = 640  # Instead of 800
   SCREEN_HEIGHT = 480  # Instead of 600
   ```

4. **Check System Resources:**
   - Ensure at least 512MB RAM available
   - Check hard drive space (need 50MB+)
   - Monitor CPU usage

#### Problem: Game freezes or crashes
**Symptoms:**
- Game stops responding
- Window becomes unresponsive
- Python error messages

**Solutions:**
1. **Check Error Messages:**
   - Run from terminal to see error output
   - Look for specific error types

2. **Update Dependencies:**
   ```bash
   poetry update
   ```

3. **Clear Cache:**
   ```bash
   # Remove cached files
   find . -type d -name "__pycache__" -exec rm -rf {} +  # Linux/macOS
   ```

4. **Restart Python Environment:**
   ```bash
   poetry shell --rm
   poetry install
   ```

### Display Issues

#### Problem: Game window doesn't appear or appears incorrectly
**Symptoms:**
- Black screen
- Window appears off-screen
- Incorrect colors or graphics

**Solutions:**
1. **Check Display Configuration:**
   - Ensure monitor is set to native resolution
   - Try different display modes

2. **Graphics Driver Issues:**
   - Update graphics drivers
   - Try running in compatibility mode

3. **Multi-Monitor Setup:**
   - Try moving window to primary monitor
   - Check display scaling settings

4. **Color Depth Issues:**
   ```python
   # In pygame initialization, try:
   pygame.display.set_mode((640, 480), pygame.DOUBLEBUF)
   ```

### Audio Issues

#### Problem: No sound or audio errors
**Note**: Current version doesn't include audio, but for future reference:

**Solutions:**
1. **Check System Audio:**
   - Ensure system volume is up
   - Check if other applications have audio

2. **Audio Drivers:**
   - Update audio drivers
   - Check audio device settings

3. **Pygame Audio:**
   ```python
   # Initialize audio properly
   pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
   pygame.mixer.init()
   ```

---

## ⌨️ Control Issues

### Input Problems

#### Problem: Keyboard controls don't work
**Symptoms:**
- Player doesn't move
- Keys seem unresponsive
- Some keys work, others don't

**Solutions:**
1. **Check Window Focus:**
   - Click on the game window
   - Ensure game has input focus

2. **Try Alternative Keys:**
   ```
   Instead of WASD, try Arrow keys
   Instead of Arrow keys, try WASD
   ```

3. **Check Keyboard Layout:**
   - Ensure correct keyboard language/layout
   - Try with different keyboard if available

4. **Key Repeat Settings:**
   - **Windows**: Control Panel → Keyboard → adjust repeat rate
   - **macOS**: System Preferences → Keyboard → Key Repeat
   - **Linux**: Settings → Keyboard → repeat settings

#### Problem: Input lag or delayed response
**Symptoms:**
- Controls feel sluggish
- Delay between key press and action

**Solutions:**
1. **Reduce System Load:**
   - Close unnecessary applications
   - Check for background processes

2. **Game Performance:**
   - Lower frame rate if necessary
   - Reduce screen resolution

3. **Hardware Issues:**
   - Try different keyboard
   - Check USB ports/connections

---

## 🛠️ Development Issues

### Code Quality Tool Problems

#### Problem: Black formatting fails
**Symptoms:**
- Black command produces errors
- Can't format code

**Solutions:**
1. **Check Black Installation:**
   ```bash
   poetry run black --version
   ```

2. **Reinstall Black:**
   ```bash
   poetry remove black
   poetry add --group dev black
   ```

3. **Run with Verbose Output:**
   ```bash
   poetry run black --verbose infinite_maze/
   ```

#### Problem: Flake8 linting errors
**Symptoms:**
- Many linting errors appear
- Code doesn't meet style guidelines

**Solutions:**
1. **Understand Error Codes:**
   ```bash
   poetry run flake8 --help
   ```

2. **Fix Common Issues:**
   - E501: Line too long (use Black to fix)
   - W503: Line break before binary operator (ignore in config)
   - F401: Unused import

3. **Configuration:**
   ```ini
   # In setup.cfg or .flake8
   [flake8]
   max-line-length = 88
   extend-ignore = E203, W503
   ```

#### Problem: MyPy type checking errors
**Symptoms:**
- Type checking fails
- Many type-related errors

**Solutions:**
1. **Install Type Stubs:**
   ```bash
   poetry add --group dev types-pygame
   ```

2. **Add Type Annotations:**
   ```python
   # Before
   def move_player(x, y):
       pass
   
   # After
   def move_player(x: int, y: int) -> None:
       pass
   ```

3. **Ignore Specific Errors:**
   ```python
   import pygame  # type: ignore
   ```

### Git and Version Control Issues

#### Problem: Git conflicts during development
**Symptoms:**
- Merge conflicts when pulling updates
- Cannot commit or push changes

**Solutions:**
1. **Resolve Conflicts:**
   ```bash
   git status  # See conflicted files
   # Edit files to resolve conflicts
   git add .
   git commit
   ```

2. **Keep Fork Updated:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. **Use Merge Tools:**
   ```bash
   git mergetool
   ```

---

## 📱 Platform-Specific Issues

### Windows Issues

#### Problem: PowerShell execution policy errors
**Symptoms:**
- "Execution of scripts is disabled on this system"
- PowerShell scripts won't run

**Solutions:**
1. **Change Execution Policy:**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Use Command Prompt:**
   - Use cmd.exe instead of PowerShell
   - Run .bat files instead of .ps1 files

#### Problem: Path issues with Poetry
**Symptoms:**
- Poetry installed but not found
- Commands work in some terminals but not others

**Solutions:**
1. **Add to PATH Manually:**
   - Open Environment Variables
   - Add `%APPDATA%\Python\Scripts` to PATH

2. **Use Python Launcher:**
   ```cmd
   py -m poetry install
   ```

### macOS Issues

#### Problem: "Developer cannot be verified" warnings
**Symptoms:**
- Security warnings when running the game
- macOS blocks execution

**Solutions:**
1. **Allow in Security Preferences:**
   - System Preferences → Security & Privacy
   - Click "Allow" for the blocked application

2. **Right-click to Open:**
   - Right-click the application
   - Select "Open" from context menu

3. **Command Line Override:**
   ```bash
   xattr -dr com.apple.quarantine /path/to/infinite-maze
   ```

### Linux Issues

#### Problem: Missing development headers
**Symptoms:**
- Compilation fails during pip install
- "No such file or directory" for header files

**Solutions:**
1. **Install Development Packages:**
   ```bash
   # Ubuntu/Debian
   sudo apt install python3-dev build-essential
   
   # Fedora
   sudo dnf install python3-devel gcc
   
   # CentOS
   sudo yum install python3-devel gcc
   ```

2. **Install SDL Development Libraries:**
   ```bash
   # Ubuntu/Debian
   sudo apt install libsdl2-dev libsdl2-image-dev
   
   # Fedora
   sudo dnf install SDL2-devel SDL2_image-devel
   ```

---

## 🔧 Advanced Troubleshooting

### Environment Debugging

#### Problem: Unclear what's causing issues
**General debugging approach:**

1. **Check Environment:**
   ```bash
   python --version
   poetry --version
   pip list
   echo $PATH
   ```

2. **Run with Verbose Output:**
   ```bash
   poetry run python -v -m infinite_maze
   ```

3. **Check System Resources:**
   ```bash
   # Memory usage
   free -m  # Linux
   vm_stat  # macOS
   
   # Disk space
   df -h
   ```

4. **Test Minimal Setup:**
   ```python
   # Test basic pygame
   import pygame
   pygame.init()
   print("Pygame version:", pygame.version.ver)
   ```

### Clean Installation

#### When to do a clean installation:
- Multiple installation methods have been tried
- Dependencies are corrupted
- Persistent unexplained errors

#### Steps:
1. **Remove Everything:**
   ```bash
   # Remove virtual environment
   poetry env remove python
   
   # Remove Poetry cache
   poetry cache clear . --all
   
   # Remove pip cache
   pip cache purge
   ```

2. **Fresh Installation:**
   ```bash
   # Clone fresh copy
   git clone https://github.com/1024Adam/infinite-maze.git fresh-infinite-maze
   cd fresh-infinite-maze
   
   # Install fresh
   poetry install
   poetry run infinite-maze
   ```

---

## 📞 Getting Additional Help

### When to Seek Help
- Issues persist after trying solutions in this guide
- Encountering new or unusual errors
- System-specific problems not covered here

### Where to Get Help
1. **GitHub Issues**: Report bugs and technical problems
2. **Documentation**: Check other guides in docs/ directory
3. **Community Forums**: Join discussions with other users

### Information to Include When Asking for Help
- **Operating System**: Version and type
- **Python Version**: Output of `python --version`
- **Installation Method**: Poetry, pip, setup script
- **Error Messages**: Complete error text
- **Steps Taken**: What you've already tried
- **System Information**: Hardware, memory, graphics card

### Creating a Good Bug Report
```markdown
**Environment:**
- OS: Windows 10 / macOS 12.0 / Ubuntu 20.04
- Python: 3.10.4
- Installation: Poetry / pip / setup script

**Problem:**
Brief description of the issue

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Result:**
What should happen

**Actual Result:**
What actually happens

**Error Messages:**
```
Full error message here
```

**Attempted Solutions:**
- Tried solution A
- Tried solution B
```

---

## 🧪 Testing Issues

### Test Suite Not Running

**Problem**: `python run_tests.py` fails or tests don't execute

**Solutions:**
1. **Install Test Dependencies:**
   ```bash
   pip install -r requirements-test.txt
   ```

2. **Check Python Path:**
   ```bash
   # Add project to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
   set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
   ```

3. **Use Direct Pytest:**
   ```bash
   python -m pytest tests/
   ```

### Pygame Display Errors in Tests

**Problem**: "pygame.error: video system not initialized" during tests

**Solutions:**
- Tests should use `headless=True` mode automatically
- Check that test fixtures are properly configured
- Verify test imports use mocked pygame components

### Performance Tests Failing

**Problem**: Performance tests fail on slower systems

**Solutions:**
1. **Skip Performance Tests:**
   ```bash
   pytest --skip-slow
   python run_tests.py unit  # Runs only unit tests
   ```

2. **Run Performance Tests in Isolation:**
   ```bash
   pytest -m performance --run-performance
   ```

3. **Check System Load:**
   - Close other applications
   - Run tests when system is idle

### Import Errors in Tests

**Problem**: "ModuleNotFoundError" when running tests

**Solutions:**
1. **Install in Development Mode:**
   ```bash
   pip install -e .
   ```

2. **Check Package Structure:**
   ```bash
   python -c "import infinite_maze; print(infinite_maze.__file__)"
   ```

### Coverage Report Issues

**Problem**: Coverage reports show incorrect or missing coverage

**Solutions:**
1. **Install Coverage Tools:**
   ```bash
   pip install pytest-cov
   ```

2. **Run with Explicit Coverage:**
   ```bash
   pytest --cov=infinite_maze --cov-report=html
   ```

3. **Check Coverage Configuration:**
   - Verify `.coveragerc` file if present
   - Ensure source paths are correct

---

## ✅ Prevention Tips

### Avoiding Common Issues
1. **Keep Software Updated:**
   - Update Python regularly
   - Update Poetry and pip
   - Update system drivers

2. **Use Virtual Environments:**
   - Isolate project dependencies
   - Avoid system-wide package conflicts

3. **Follow Installation Guide:**
   - Use recommended installation methods
   - Don't skip steps
   - Read error messages carefully

4. **Regular Maintenance:**
   ```bash
   # Update dependencies occasionally
   poetry update
   
   # Clear caches if issues arise
   poetry cache clear . --all
   pip cache purge
   ```

---

*This troubleshooting guide covers the most common issues. If you encounter a problem not covered here, please consider contributing a solution to help other users!* 🛠️

# Contributing to Infinite Maze

Thank you for your interest in contributing to Infinite Maze! This guide will help you get started with contributing to the project, whether you're fixing bugs, adding features, improving documentation, or helping with testing.

## 🌟 Ways to Contribute

### Code Contributions
- **Bug Fixes**: Help resolve issues and improve stability
- **New Features**: Add gameplay mechanics, visual improvements, or quality-of-life features
- **Performance Optimizations**: Improve game performance and efficiency
- **Code Quality**: Refactoring, type annotations, and architectural improvements

### Non-Code Contributions
- **Documentation**: Improve guides, add examples, fix typos
- **Testing**: Manual testing, automated test creation, edge case discovery
- **Design**: UI/UX improvements, graphics, sprites, sound effects
- **Community**: Help other users, answer questions, provide feedback

### Feedback & Ideas
- **Bug Reports**: Report issues you encounter
- **Feature Requests**: Suggest new functionality
- **User Experience**: Share your gameplay experience and suggestions
- **Performance Reports**: Help identify performance bottlenecks

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** installed on your system
- **Git** for version control
- **Poetry** for dependency management (recommended)
- **Basic Python knowledge** for code contributions

### Setting Up Your Development Environment

#### 1. Fork and Clone the Repository
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/infinite-maze.git
cd infinite-maze

# Add the original repository as upstream
git remote add upstream https://github.com/1024Adam/infinite-maze.git
```

#### 2. Install Development Dependencies
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Verify installation
poetry run infinite-maze
```

#### 3. Set Up Development Tools
```bash
# Install pre-commit hooks (optional but recommended)
poetry run pre-commit install

# Run code quality checks
poetry run black infinite_maze/      # Format code
poetry run flake8 infinite_maze/     # Lint code
poetry run mypy infinite_maze/       # Type checking
```

#### 4. Create a Feature Branch
```bash
# Create and switch to a new branch for your changes
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b bugfix/issue-description
```

---

## 📝 Development Workflow

### 1. Code Style and Standards

#### Python Code Style
We use **Black** for automatic code formatting:
```bash
# Format all code
poetry run black infinite_maze/

# Check formatting without changes
poetry run black --check infinite_maze/
```

#### Code Quality
We use **Flake8** for linting:
```bash
# Run linting
poetry run flake8 infinite_maze/

# View specific error details
poetry run flake8 --show-source infinite_maze/
```

#### Type Annotations
We use **MyPy** for static type checking:
```bash
# Run type checking
poetry run mypy infinite_maze/
```

#### Code Standards
- **Function Names**: Use `snake_case`
- **Class Names**: Use `PascalCase`
- **Constants**: Use `UPPER_CASE`
- **Line Length**: Maximum 88 characters (Black default)
- **Docstrings**: Use triple quotes with clear descriptions
- **Type Hints**: Add type annotations for function parameters and return values

#### Example Code Style
```python
from typing import List, Tuple, Optional

class MazeGenerator:
    """Generates procedural maze layouts."""
    
    def __init__(self, width: int, height: int) -> None:
        """Initialize maze generator with dimensions."""
        self.width = width
        self.height = height
        
    def generate_maze(self, complexity: float = 0.5) -> List[Tuple[int, int]]:
        """Generate a maze with specified complexity.
        
        Args:
            complexity: Maze complexity factor (0.0 to 1.0)
            
        Returns:
            List of wall coordinates as (x, y) tuples
        """
        # Implementation here
        pass
```

### 2. Testing Your Changes

#### Manual Testing
```bash
# Run the game to test your changes
poetry run infinite-maze

# Test different scenarios
# - Normal gameplay
# - Pause/resume functionality
# - Game over and restart
# - Edge cases and error conditions
```

#### Code Quality Checks
```bash
# Run all quality checks
poetry run black infinite_maze/
poetry run flake8 infinite_maze/
poetry run mypy infinite_maze/
```

#### Testing Checklist
- [ ] Game starts without errors
- [ ] All controls work as expected
- [ ] No visual glitches or rendering issues
- [ ] Performance is acceptable
- [ ] Code follows style guidelines
- [ ] No new linting errors
- [ ] Type checking passes

### 3. Commit Guidelines

#### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Commit Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or modifying tests
- **chore**: Maintenance tasks

#### Examples
```bash
# Good commit messages
git commit -m "feat(player): add diagonal movement support"
git commit -m "fix(maze): resolve wall collision edge case"
git commit -m "docs: improve installation guide clarity"
git commit -m "refactor(engine): simplify game loop logic"

# Include more details for complex changes
git commit -m "feat(scoring): implement bonus point system

- Add combo multiplier for consecutive rightward moves
- Display bonus indicators in UI
- Update score calculation logic
- Add configuration options for bonus system

Closes #123"
```

---

## 🐛 Bug Reports

### Before Reporting a Bug
1. **Check existing issues** to see if the bug has already been reported
2. **Try the latest version** to see if the bug has been fixed
3. **Reproduce the bug** consistently
4. **Gather information** about your system and the bug

### Creating a Bug Report

Use this template for bug reports:

```markdown
**Bug Description**
A clear and concise description of what the bug is.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Actual Behavior**
A clear description of what actually happened.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment**
- OS: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g. 3.10.4]
- Game Version: [e.g. 0.2.0]
- Installation Method: [e.g. Poetry, pip, setup script]

**Additional Context**
Add any other context about the problem here.
```

---

## 💡 Feature Requests

### Before Requesting a Feature
1. **Check existing issues** for similar requests
2. **Consider the scope** - does it fit the game's vision?
3. **Think about implementation** - is it technically feasible?

### Creating a Feature Request

Use this template:

```markdown
**Feature Summary**
A brief description of the feature you'd like to see.

**Problem Statement**
Describe the problem this feature would solve or the improvement it would make.

**Proposed Solution**
Describe how you think this feature should work.

**Alternative Solutions**
Describe any alternative solutions you've considered.

**Use Cases**
Provide specific examples of when and how this feature would be used.

**Implementation Notes**
Any technical considerations or suggestions for implementation.
```

---

## 🔧 Development Guidelines

### Architecture Guidelines

#### 1. Modularity
- Keep components focused and single-purpose
- Use clear interfaces between modules
- Avoid tight coupling between classes

#### 2. Configuration
- Add new settings to `config.py`
- Use type hints for configuration values
- Provide sensible defaults

#### 3. Error Handling
- Handle expected errors gracefully
- Provide meaningful error messages
- Log errors appropriately

#### 4. Performance
- Avoid unnecessary computations in the main game loop
- Use efficient algorithms and data structures
- Profile performance-critical code

### Adding New Features

#### 1. Plan Your Feature
- Define the feature's purpose and scope
- Consider how it fits with existing architecture
- Plan the user interface and user experience
- Consider configuration options

#### 2. Implement Step by Step
- Start with core functionality
- Add tests as you go
- Update documentation
- Consider backwards compatibility

#### 3. Example: Adding a New Game Mode

```python
# 1. Add configuration
class GameConfig:
    GAME_MODES = {
        "classic": "Classic infinite maze",
        "timed": "Time-limited challenge",
        "speed": "High-speed variant"
    }

# 2. Extend the Game class
class Game:
    def __init__(self, mode: str = "classic"):
        self.mode = mode
        # Mode-specific initialization
        
    def update_mode_specific_logic(self):
        if self.mode == "timed":
            # Timed mode logic
            pass
        elif self.mode == "speed":
            # Speed mode logic
            pass

# 3. Update the engine
def maze(mode: str = "classic"):
    game = Game(mode=mode)
    # Rest of the game loop
```

### Code Review Process

#### What We Look For
- **Functionality**: Does the code work as intended?
- **Code Quality**: Is the code clean, readable, and well-structured?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security concerns?
- **Documentation**: Is the code properly documented?
- **Testing**: Are there adequate tests?

#### Review Checklist
- [ ] Code follows project style guidelines
- [ ] Changes are well-documented
- [ ] No breaking changes without justification
- [ ] Performance impact is acceptable
- [ ] Error handling is appropriate
- [ ] Tests cover new functionality

---

## 📋 Pull Request Process

### 1. Prepare Your Pull Request
```bash
# Ensure your branch is up to date
git fetch upstream
git rebase upstream/main

# Run all quality checks
poetry run black infinite_maze/
poetry run flake8 infinite_maze/
poetry run mypy infinite_maze/

# Test your changes thoroughly
poetry run infinite-maze
```

### 2. Create the Pull Request

#### Pull Request Template
```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Changes Made
- List of specific changes
- Include any architectural changes
- Mention any new dependencies

## Testing
- [ ] I have tested these changes locally
- [ ] I have run the code quality checks
- [ ] I have updated documentation as needed

## Screenshots
(If applicable, add screenshots to demonstrate the changes)

## Additional Notes
Any additional information or context about the changes.
```

### 3. After Submitting
- **Respond to feedback** promptly and constructively
- **Make requested changes** in additional commits
- **Keep the PR updated** with the latest main branch changes
- **Be patient** - reviews take time, especially for complex changes

---

## 🧪 Testing Guidelines

### Manual Testing Scenarios

#### Basic Functionality
- [ ] Game starts without errors
- [ ] Player movement works in all directions
- [ ] Collision detection works correctly
- [ ] Pause/resume functionality works
- [ ] Game over triggers correctly
- [ ] Restart functionality works
- [ ] Scoring system works accurately

#### Edge Cases
- [ ] Rapid key presses don't cause issues
- [ ] Pausing during movement transitions
- [ ] High pace values don't break collision detection
- [ ] Window resizing (if supported)
- [ ] Alt-tabbing and focus changes

#### Performance Testing
- [ ] Game maintains 60 FPS under normal conditions
- [ ] Memory usage remains stable over time
- [ ] No memory leaks during extended play
- [ ] CPU usage is reasonable

### Adding Automated Tests

```python
# Example test structure
import pytest
from infinite_maze.entities.player import Player
from infinite_maze.core.game import Game

class TestPlayer:
    def test_player_initialization(self):
        player = Player(100, 200, headless=True)
        assert player.getX() == 100
        assert player.getY() == 200
        
    def test_player_movement(self):
        player = Player(100, 200, headless=True)
        player.moveX(10)
        assert player.getX() == 110
        
    def test_player_boundaries(self):
        # Test boundary conditions
        pass

class TestGame:
    def test_game_initialization(self):
        game = Game(headless=True)
        assert game.getScore() == 0
        assert game.isActive() == True
        
    def test_scoring_system(self):
        game = Game(headless=True)
        game.incrementScore()
        assert game.getScore() == 1
```

---

## 📚 Documentation Guidelines

### Types of Documentation

#### 1. Code Documentation
- **Docstrings**: For all classes and functions
- **Type Hints**: For all function parameters and return values
- **Inline Comments**: For complex logic

#### 2. User Documentation
- **Installation guides**: Keep updated with new requirements
- **Player guides**: Update with new features
- **Troubleshooting**: Add common issues and solutions

#### 3. Developer Documentation
- **API references**: Update with new classes/methods
- **Architecture guides**: Explain design decisions
- **Contributing guides**: Keep this guide updated

### Documentation Style

#### Docstring Style
```python
def generate_maze(self, rows: int, cols: int, complexity: float = 0.5) -> List[Line]:
    """Generate a procedural maze layout.
    
    Creates a maze with the specified dimensions and complexity level.
    The maze is guaranteed to be solvable and have a reasonable difficulty.
    
    Args:
        rows: Number of maze rows (must be positive)
        cols: Number of maze columns (must be positive)
        complexity: Complexity factor from 0.0 (simple) to 1.0 (complex)
        
    Returns:
        List of Line objects representing maze walls
        
    Raises:
        ValueError: If rows or cols are not positive
        ValueError: If complexity is not between 0.0 and 1.0
        
    Example:
        >>> generator = MazeGenerator()
        >>> maze = generator.generate_maze(10, 15, 0.7)
        >>> len(maze) > 0
        True
    """
```

---

## 🎯 Project Vision & Priorities

### Core Principles
1. **Simplicity**: Keep the game accessible and easy to understand
2. **Performance**: Maintain smooth gameplay at 60 FPS
3. **Extensibility**: Design for easy modification and extension
4. **Quality**: Prioritize code quality and maintainability

### Current Priorities
1. **Stability**: Fix existing bugs and edge cases
2. **Performance**: Optimize game loop and rendering
3. **Documentation**: Improve user and developer documentation
4. **Testing**: Add automated tests for core functionality

### Future Vision
- **Enhanced Graphics**: Better sprites, animations, and visual effects
- **Game Modes**: Additional gameplay variants and challenges
- **Multiplayer**: Local or online multiplayer capabilities
- **Mobile Support**: Cross-platform deployment
- **Plugin System**: Extensible architecture for community mods

---

## 🏆 Recognition

### Contributors
All contributors are recognized in:
- **README.md**: Contributors section
- **Release Notes**: Acknowledgment of contributions
- **Git History**: Permanent record of contributions

### Types of Recognition
- **Code Contributors**: Major features, bug fixes, optimizations
- **Documentation Contributors**: Guides, API docs, examples
- **Community Contributors**: Testing, feedback, support
- **Design Contributors**: Graphics, UI/UX, game design

---

## 📞 Getting Help

### Development Questions
- **GitHub Discussions**: For general questions and brainstorming
- **GitHub Issues**: For specific bugs or feature requests
- **Code Review**: Ask for feedback on your approach before implementing

### Learning Resources
- **Python Documentation**: https://docs.python.org/
- **Pygame Documentation**: https://www.pygame.org/docs/
- **Poetry Documentation**: https://python-poetry.org/docs/
- **Git Documentation**: https://git-scm.com/doc

### Communication Guidelines
- **Be Respectful**: Treat all community members with respect
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Everyone is learning and contributing in their spare time
- **Be Clear**: Communicate clearly and provide context for your questions

---

Thank you for contributing to Infinite Maze! Your contributions help make the game better for everyone. Whether you're fixing a small bug or adding a major feature, every contribution is valuable and appreciated.

Happy coding! 🎮✨

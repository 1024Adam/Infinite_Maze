# Infinite Maze Game

## Overview
Infinite Maze is a dynamic maze-style game where players navigate through a randomly generated maze, attempting to progress as far as possible while maintaining pace. The maze is infinite - as long as you can keep up with the advancing pace, the game continues indefinitely.

What makes this truly challenging is the progressive difficulty: the game's pace increases every 30 seconds, forcing players to move faster while navigating increasingly complex maze layouts.

## Quick Start

### One-Command Setup (Easiest)
```bash
# Linux/macOS
./setup.sh

# Windows
setup.bat
```

### Manual Setup
```bash
# 1. Install Poetry (one-time setup per machine)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Install dependencies and run the game
poetry install
poetry run infinite-maze
```

### Alternative Installation Methods
```bash
# Using pip (if you prefer not to use Poetry)
pip install -e .
infinite-maze

# Direct execution
python -m infinite_maze

# Legacy method (still supported)
python run_game.py
```

### Development Setup
```bash
# Install with development dependencies
poetry install

# Code formatting
poetry run black infinite_maze/

# Code linting
poetry run flake8 infinite_maze/

# Type checking
poetry run mypy infinite_maze/
```

### First-Time Developer Setup
If you're new to the project:

1. **Clone the repository**
2. **Run the setup script**: `./setup.sh` (Linux/macOS) or `setup.bat` (Windows)
3. **Start developing!**

The setup script will:
- Check if Poetry is installed (and guide you to install it if needed)
- Install all project dependencies
- Verify everything works correctly

## Game Mechanics

|  Element   |  Description  |
| ---------- | ------------- |
| **Player** | Your character, represented by a dot that moves through the maze |
| **Pace**   | The advancing boundary that pursues you - stay ahead or the game ends |
| **Walls**  | Maze barriers that block your movement |
| **Points** | Scored based on rightward progress through the maze |

## Game Rules
- **Objective**: Navigate as far right through the infinite maze as possible
- **Scoring**: Gain points for rightward movement, lose points for moving left
- **Pace**: Starts after 30 seconds and accelerates every 30 seconds thereafter
- **Game Over**: When the pace catches up to your position

## Controls
|  Key(s)        |  Action                 |
| -------------- | ----------------------- |
| **W, A, S, D** | Move up, left, down, right |
| **Arrow Keys** | Alternative movement controls |
| **Space**      | Pause/unpause game |
| **Esc, Q**     | Quit game |
| **Y, N**       | Yes/No responses when prompted |

## Technical Requirements
- **Python**: 3.8+ (tested with Python 3.10.4)
- **Dependencies**: Pygame 2.5+ (automatically managed by Poetry)
- **Platforms**: Windows, macOS, Linux

## Architecture

### Modern Project Structure
```
infinite_maze/
├── infinite_maze/          # Main game package
│   ├── __init__.py         # Package initialization
│   ├── __main__.py         # Entry point for python -m infinite_maze
│   ├── infinite_maze.py    # Main game logic
│   ├── Game.py            # Core game state management
│   ├── Player.py          # Player character implementation
│   ├── Clock.py           # Game timing and frame rate control
│   ├── Line.py            # Maze generation and utilities
│   ├── config.py          # Configuration management
│   └── logger.py          # Centralized logging
├── assets/                # Game assets and resources
├── pyproject.toml         # Modern Python project configuration
├── DEVELOPMENT.md         # Development setup guide
└── migrate_to_modern.py   # Migration helper script
```

### Key Features
- **Modern Python packaging** with Poetry and pyproject.toml
- **Configuration management** - externalized settings for easy customization
- **Comprehensive logging** - structured logging for debugging and monitoring
- **Type annotations** - improved code documentation and IDE support
- **Modular architecture** - clean separation of concerns
- **Multiple entry points** - flexible ways to run the game

## Development

### Code Quality Tools
- **Black**: Code formatting
- **Flake8**: Code linting and style checking
- **MyPy**: Static type checking
- **Pre-commit**: Git hooks for code quality (optional)

### Task Runners (Optional)
For convenience, you can use modern task runners instead of remembering Poetry commands:

#### Just (Recommended)
```bash
# Install Just: cargo install just
just setup     # Setup project
just run       # Run the game
just format    # Format code
just check     # Run all quality checks
```

#### Make (Traditional)
If you prefer the traditional approach, basic commands:
```bash
# You can create your own Makefile with Poetry commands
poetry install  # instead of 'make install'
poetry run infinite-maze  # instead of 'make run'
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper type hints and documentation
4. Run the code quality tools
5. Submit a pull request

## Migration from Legacy Version
If you're upgrading from an older version using Makefile/setup.py:

```bash
# Run the migration script
python migrate_to_modern.py
```

This will:
- Back up your old configuration files
- Set up Poetry and modern dependencies
- Verify the new installation works correctly

## Dependencies
- **Pygame**: Game development library for graphics, sound, and input handling

For more information about Pygame, visit [pygame.org](https://www.pygame.org/docs/).

---

**Infinite Maze Game** - Evolution, not revolution. Modern Python practices with classic gameplay.

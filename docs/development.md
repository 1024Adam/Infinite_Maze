# Infinite Maze - Modern Development Guide

## Quick Start

### Installation with Poetry (Recommended)
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Run the game
poetry run infinite-maze
# or
poetry run maze-game
```

### Installation with pip (Legacy)
```bash
pip install -e .
infinite-maze
```

## Development Commands

### With Poetry
```bash
# Install development dependencies
poetry install

# Format code
poetry run black infinite_maze/

# Lint code
poetry run flake8 infinite_maze/

# Type checking
poetry run mypy infinite_maze/

# Clean cache files
poetry run python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
```

### Building and Distribution
```bash
# Build the package
poetry build

# Publish to PyPI (when ready)
poetry publish
```

## Project Structure
```
infinite_maze/
├── infinite_maze/          # Main game package
│   ├── __init__.py
│   ├── __main__.py         # Entry point for python -m infinite_maze
│   ├── infinite_maze.py    # Main game logic
│   ├── Game.py            # Game class
│   ├── Player.py          # Player class
│   ├── Clock.py           # Game clock
│   └── Line.py            # Line utilities
├── assets/                # Game assets
│   └── images/
├── pyproject.toml         # Modern Python project configuration
├── README.md             # Project documentation
└── run_game.py           # Legacy launcher (maintained for compatibility)
```

## Migration Notes
- `setup.py` → Replaced by `pyproject.toml`
- `requirements.txt` → Managed by Poetry in `pyproject.toml`
- `Makefile` → Replaced by Poetry scripts
- All functionality preserved with modern tooling

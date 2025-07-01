# Justfile - Modern task runner (alternative to Makefile)
# Install 'just' with: cargo install just
# Then run: just setup

# Default recipe (runs when you just type 'just')
default:
    @just --list

# Setup development environment
setup:
    @echo "🎮 Setting up Infinite Maze..."
    @if ! command -v poetry >/dev/null 2>&1; then \
        echo "❌ Poetry not found. Please install it first:"; \
        echo "   curl -sSL https://install.python-poetry.org | python3 -"; \
        exit 1; \
    fi
    poetry install
    @echo "✅ Setup complete!"

# Run the game
run:
    poetry run infinite-maze

# Alternative way to run the game
play:
    poetry run maze-game

# Development commands
format:
    poetry run black infinite_maze/

lint:
    poetry run flake8 infinite_maze/

typecheck:
    poetry run mypy infinite_maze/

# Run all quality checks
check: format lint typecheck
    @echo "✅ All checks passed!"

# Clean up build artifacts
clean:
    @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
    @echo "🧹 Cleaned up build artifacts"

# Build the package
build:
    poetry build

# Install in development mode
install:
    poetry install

# Show project info
info:
    @echo "📁 Project: Infinite Maze"
    @echo "🐍 Python: $(python --version)"
    @echo "📦 Poetry: $(poetry --version)"
    @echo "🎯 Entry points:"
    @echo "   poetry run infinite-maze"
    @echo "   poetry run maze-game"
    @echo "   python -m infinite_maze"

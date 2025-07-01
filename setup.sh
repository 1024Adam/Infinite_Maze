#!/usr/bin/env bash
# setup.sh - One-command project setup script

set -e

echo "🎮 Setting up Infinite Maze development environment..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows
        echo "Please install Poetry manually:"
        echo "Run: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"
        echo "Or visit: https://python-poetry.org/docs/#installation"
        exit 1
    else
        # Unix-like systems
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
else
    echo "✅ Poetry already installed: $(poetry --version)"
fi

# Install project dependencies
echo "📥 Installing project dependencies..."
poetry install

# Verify installation
echo "🔍 Verifying installation..."
poetry run python -c "import infinite_maze; print('✅ Package imported successfully')"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the game:"
echo "  poetry run infinite-maze"
echo ""
echo "For development:"
echo "  poetry run black infinite_maze/    # Format code"
echo "  poetry run flake8 infinite_maze/   # Lint code"
echo "  poetry run mypy infinite_maze/     # Type checking"

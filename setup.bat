@echo off
REM setup.bat - Windows setup script for Infinite Maze

echo 🎮 Setting up Infinite Maze development environment...

REM Check if Poetry is installed
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Poetry not found. Installing...
    echo Please run the following command in PowerShell as Administrator:
    echo (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content ^| python -
    echo.
    echo Or visit: https://python-poetry.org/docs/#installation
    echo.
    echo After installing Poetry, run this script again.
    pause
    exit /b 1
) else (
    echo ✅ Poetry found
)

REM Install project dependencies
echo 📥 Installing project dependencies...
poetry install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Verify installation
echo 🔍 Verifying installation...
poetry run python -c "import infinite_maze; print('✅ Package imported successfully')"
if %errorlevel% neq 0 (
    echo ❌ Package verification failed
    pause
    exit /b 1
)

echo.
echo 🎉 Setup complete!
echo.
echo To run the game:
echo   poetry run infinite-maze
echo.
echo For development:
echo   poetry run black infinite_maze/    # Format code
echo   poetry run flake8 infinite_maze/   # Lint code
echo   poetry run mypy infinite_maze/     # Type checking
echo.
pause

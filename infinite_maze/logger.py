"""
Logging configuration for Infinite Maze game.

Provides centralized logging setup with different levels for development and production.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class GameLogger:
    """Centralized logging configuration for the game."""

    _logger: Optional[logging.Logger] = None

    @classmethod
    def setup_logger(
        cls,
        name: str = "infinite_maze",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        console_output: bool = True,
    ) -> logging.Logger:
        """
        Set up and configure the game logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            console_output: Whether to output to console

        Returns:
            Configured logger instance
        """
        if cls._logger is not None:
            return cls._logger

        cls._logger = logging.getLogger(name)
        cls._logger.setLevel(level)

        # Clear any existing handlers
        cls._logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)

        return cls._logger

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the configured logger instance."""
        if cls._logger is None:
            return cls.setup_logger()
        return cls._logger


# Convenience functions for common logging operations
def debug(message: str) -> None:
    """Log a debug message."""
    GameLogger.get_logger().debug(message)


def info(message: str) -> None:
    """Log an info message."""
    GameLogger.get_logger().info(message)


def warning(message: str) -> None:
    """Log a warning message."""
    GameLogger.get_logger().warning(message)


def error(message: str) -> None:
    """Log an error message."""
    GameLogger.get_logger().error(message)


def critical(message: str) -> None:
    """Log a critical message."""
    GameLogger.get_logger().critical(message)


# Initialize default logger
logger = GameLogger.setup_logger()

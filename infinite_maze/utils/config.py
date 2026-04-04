"""
Configuration module for Infinite Maze game.

This module contains all game configuration settings, extracted from hardcoded
values to improve maintainability and allow for easy customization.
"""

from typing import Tuple, Dict, Any


class GameConfig:
    """Main game configuration class."""

    # Display settings
    SCREEN_WIDTH: int = 640
    SCREEN_HEIGHT: int = 480
    FPS: int = 60

    # Player settings
    PLAYER_START_X: int = 80
    PLAYER_START_Y: int = 223
    PLAYER_SPEED: int = 1
    PLAYER_WIDTH: int = 10
    PLAYER_HEIGHT: int = 10

    # Maze generation settings
    MAZE_ROWS: int = 15
    MAZE_COLS: int = 20
    MAZE_CELL_SIZE: int = 22

    # UI and display settings
    ICON_SIZE: int = 32
    TEXT_MARGIN: int = 10
    BORDER_OFFSET: int = 10
    FPS_DELAY_MS: int = 16
    PACE_UPDATE_INTERVAL: int = 30  # seconds

    # Colors (RGB tuples)
    COLORS: Dict[str, Tuple[int, int, int]] = {
        "BLACK": (0, 0, 0),
        "WHITE": (255, 255, 255),
        "RED": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "BLUE": (0, 0, 255),
        "YELLOW": (255, 255, 0),
        "PURPLE": (128, 0, 128),
        "ORANGE": (255, 165, 0),
        "GRAY": (128, 128, 128),
        "DARK_GRAY": (64, 64, 64),
    }

    # Game controls mapping
    CONTROLS: Dict[str, Any] = {
        "MOVE_RIGHT": ["RIGHT", "d"],
        "MOVE_LEFT": ["LEFT", "a"],
        "MOVE_UP": ["UP", "w"],
        "MOVE_DOWN": ["DOWN", "s"],
        "PAUSE": ["SPACE"],
        "QUIT": ["ESCAPE", "q"],
        "RESTART": ["r"],
    }

    # File paths
    ASSETS_DIR: str = "assets"
    IMAGES_DIR: str = "assets/images"

    # Image files
    IMAGES: Dict[str, str] = {
        "player": "assets/images/player.png",
        "player_paused": "assets/images/player_paused.png",
        "icon": "assets/images/icon.png",
    }

    # Game mechanics
    MOVEMENT_CONSTANTS: Dict[str, int] = {
        "DO_NOTHING": 0,
        "RIGHT": 1,
        "LEFT": 2,
        "UP": 3,
        "DOWN": 4,
    }

    # ML / RL training configuration
    ML_CONFIG: Dict[str, Any] = {
        # Observation normalisation
        "MAX_PACE": 10,
        "EPISODE_SCORE_CAP": 600,
        "MAX_WALL_SCAN_DIST": 200,
        "CONSECUTIVE_BLOCKED_CAP": 50,
        "GAP_SCAN_RADIUS": 220,
        "GRID_COLS": 4,
        "GRID_ROWS": 5,
        # Pace simulation (tick-based, not real-time)
        "TICKS_PER_PACE_UPDATE": 300,
        "PACE_SHIFT_INTERVAL": 10,
        # Reward weights
        "REWARD_MOVE_RIGHT": 1.0,
        "REWARD_MOVE_RIGHT_BLOCKED": -0.3,
        "REWARD_MOVE_LEFT": -1.5,
        "REWARD_DO_NOTHING": -0.1,
        "REWARD_MOVE_VERTICAL_UP": 0.0,
        "REWARD_MOVE_VERTICAL_DOWN": 0.0,
        "REWARD_VERTICAL_WHEN_BLOCKED_UP": 0.1,
        "REWARD_VERTICAL_WHEN_BLOCKED_DOWN": 0.1,
        "REWARD_TERMINAL": -10.0,
        "REWARD_BFS_MATCH": 0.05,
        # Training
        "DEFAULT_TIMESTEPS": 200_000,
        "CHECKPOINT_FREQ": 10_000,
        "EVAL_FREQ": 20_000,
        "EVAL_EPISODES": 10,
        "TENSORBOARD_LOG": "runs/",
    }

    @classmethod
    def get_color(cls, color_name: str) -> Tuple[int, int, int]:
        """Get a color by name, defaulting to white if not found."""
        return cls.COLORS.get(color_name.upper(), cls.COLORS["WHITE"])

    @classmethod
    def get_image_path(cls, image_name: str) -> str:
        """Get the full path for an image asset."""
        return cls.IMAGES.get(image_name, "")

    @classmethod
    def get_asset_path(cls, *path_parts: str) -> str:
        """Build a path relative to the assets directory."""
        import os

        return os.path.join(cls.ASSETS_DIR, *path_parts)

    @classmethod
    def get_movement_constant(cls, action: str) -> int:
        """Get movement constant by action name."""
        return cls.MOVEMENT_CONSTANTS.get(
            action.upper(), cls.MOVEMENT_CONSTANTS["DO_NOTHING"]
        )

    @classmethod
    def getPlayerImage(cls) -> str:
        """Get the player image path."""
        return cls.get_image_path("player")

    @classmethod
    def getPlayerPausedImage(cls) -> str:
        """Get the player paused image path."""
        return cls.get_image_path("player_paused")

    @classmethod
    def getIcon(cls) -> str:
        """Get the icon image path."""
        return cls.get_image_path("icon")


# Create a default config instance for easy importing
config = GameConfig()

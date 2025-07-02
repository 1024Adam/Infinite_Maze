"""
Pygame-specific mocking utilities for Infinite Maze testing.

This module provides comprehensive mocking capabilities for pygame
components to enable isolated testing without actual rendering or audio.
"""

import pygame
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Tuple, Optional
import contextlib


class MockPygameSurface:
    """Mock pygame Surface object for testing."""
    
    def __init__(self, size: Tuple[int, int] = (800, 600)):
        self.size = size
        self.fills = []
        self.blits = []
        self.draw_calls = []
        
    def get_size(self) -> Tuple[int, int]:
        return self.size
    
    def get_width(self) -> int:
        return self.size[0]
    
    def get_height(self) -> int:
        return self.size[1]
    
    def fill(self, color):
        self.fills.append(color)
    
    def blit(self, source, dest, area=None, special_flags=0):
        self.blits.append({
            'source': source,
            'dest': dest,
            'area': area,
            'special_flags': special_flags
        })
    
    def convert(self):
        return self
    
    def convert_alpha(self):
        return self


class MockPygameClock:
    """Mock pygame Clock object for controlled timing tests."""
    
    def __init__(self):
        self.fps = 60
        self.frame_time = 16  # milliseconds per frame at 60fps
        self.ticks = 0
        self.time_value = 0
        
    def tick(self, framerate: int = 0) -> int:
        self.ticks += 1
        if framerate > 0:
            self.frame_time = 1000 // framerate
        self.time_value += self.frame_time
        return self.frame_time
    
    def get_fps(self) -> float:
        return self.fps
    
    def get_time(self) -> int:
        return self.time_value
    
    def reset(self):
        self.ticks = 0
        self.time_value = 0


class MockPygameFont:
    """Mock pygame Font object for text rendering tests."""
    
    def __init__(self, name: str = "Arial", size: int = 12):
        self.name = name
        self.size = size
        self.rendered_texts = []
    
    def render(self, text: str, antialias: bool, color, background=None):
        mock_surface = MockPygameSurface((len(text) * 8, self.size))
        self.rendered_texts.append({
            'text': text,
            'antialias': antialias,
            'color': color,
            'background': background
        })
        return mock_surface
    
    def get_height(self) -> int:
        return self.size
    
    def size(self, text: str) -> Tuple[int, int]:
        return (len(text) * 8, self.size)


class MockPygameEvent:
    """Mock pygame Event for input testing."""
    
    def __init__(self, event_type: int, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class PygameMockContext:
    """Context manager for comprehensive pygame mocking."""
    
    def __init__(self, mock_display: bool = True, mock_time: bool = True, 
                 mock_font: bool = True, mock_image: bool = True,
                 mock_event: bool = True, mock_draw: bool = True):
        self.mock_display = mock_display
        self.mock_time = mock_time
        self.mock_font = mock_font
        self.mock_image = mock_image
        self.mock_event = mock_event
        self.mock_draw = mock_draw
        
        self.patches = []
        self.mocks = {}
        
    def __enter__(self):
        if self.mock_display:
            self._setup_display_mocks()
        if self.mock_time:
            self._setup_time_mocks()
        if self.mock_font:
            self._setup_font_mocks()
        if self.mock_image:
            self._setup_image_mocks()
        if self.mock_event:
            self._setup_event_mocks()
        if self.mock_draw:
            self._setup_draw_mocks()
            
        return self.mocks
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in reversed(self.patches):
            patch_obj.stop()
    
    def _setup_display_mocks(self):
        """Setup display-related mocks."""
        surface_mock = MockPygameSurface()
        
        set_mode_patch = patch('pygame.display.set_mode', return_value=surface_mock)
        set_caption_patch = patch('pygame.display.set_caption')
        set_icon_patch = patch('pygame.display.set_icon')
        flip_patch = patch('pygame.display.flip')
        
        self.patches.extend([set_mode_patch, set_caption_patch, set_icon_patch, flip_patch])
        
        self.mocks['display'] = {
            'set_mode': set_mode_patch.start(),
            'set_caption': set_caption_patch.start(),
            'set_icon': set_icon_patch.start(),
            'flip': flip_patch.start(),
            'surface': surface_mock
        }
    
    def _setup_time_mocks(self):
        """Setup time-related mocks."""
        clock_mock = MockPygameClock()
        
        clock_class_patch = patch('pygame.time.Clock', return_value=clock_mock)
        delay_patch = patch('pygame.time.delay')
        
        self.patches.extend([clock_class_patch, delay_patch])
        
        self.mocks['time'] = {
            'Clock': clock_class_patch.start(),
            'delay': delay_patch.start(),
            'clock_instance': clock_mock
        }
    
    def _setup_font_mocks(self):
        """Setup font-related mocks."""
        font_mock = MockPygameFont()
        
        sysfont_patch = patch('pygame.font.SysFont', return_value=font_mock)
        font_patch = patch('pygame.font.Font', return_value=font_mock)
        
        self.patches.extend([sysfont_patch, font_patch])
        
        self.mocks['font'] = {
            'SysFont': sysfont_patch.start(),
            'Font': font_patch.start(),
            'font_instance': font_mock
        }
    
    def _setup_image_mocks(self):
        """Setup image-related mocks."""
        image_surface = MockPygameSurface((20, 20))
        
        load_patch = patch('pygame.image.load', return_value=image_surface)
        
        self.patches.append(load_patch)
        
        self.mocks['image'] = {
            'load': load_patch.start(),
            'loaded_surface': image_surface
        }
    
    def _setup_event_mocks(self):
        """Setup event-related mocks."""
        event_get_patch = patch('pygame.event.get', return_value=[])
        key_get_pressed_patch = patch('pygame.key.get_pressed')
        
        # Default key state (no keys pressed)
        key_state = {key: False for key in range(512)}
        key_get_pressed_patch.return_value = key_state
        
        self.patches.extend([event_get_patch, key_get_pressed_patch])
        
        self.mocks['event'] = {
            'get': event_get_patch.start(),
            'key_get_pressed': key_get_pressed_patch.start(),
            'key_state': key_state
        }
    
    def _setup_draw_mocks(self):
        """Setup drawing-related mocks."""
        line_patch = patch('pygame.draw.line')
        rect_patch = patch('pygame.draw.rect')
        circle_patch = patch('pygame.draw.circle')
        
        self.patches.extend([line_patch, rect_patch, circle_patch])
        
        self.mocks['draw'] = {
            'line': line_patch.start(),
            'rect': rect_patch.start(),
            'circle': circle_patch.start()
        }


def create_mock_key_state(pressed_keys: List[int]) -> Dict[int, bool]:
    """Create a mock key state with specified keys pressed."""
    key_state = {key: False for key in range(512)}
    for key in pressed_keys:
        key_state[key] = True
    return key_state


def create_test_events(event_specs: List[Dict[str, Any]]) -> List[MockPygameEvent]:
    """Create a list of test events from specifications."""
    events = []
    for spec in event_specs:
        event_type = spec.pop('type')
        event = MockPygameEvent(event_type, **spec)
        events.append(event)
    return events


class InputSimulator:
    """Simulate input sequences for testing."""
    
    def __init__(self):
        self.key_sequence = []
        self.event_sequence = []
    
    def press_key(self, key: int, duration_frames: int = 1):
        """Simulate pressing a key for specified frames."""
        for frame in range(duration_frames):
            if frame == 0:
                # Key down event
                self.event_sequence.append(MockPygameEvent(pygame.KEYDOWN, key=key))
            # Key held state
            self.key_sequence.append({key: True})
        
        # Key up event
        self.event_sequence.append(MockPygameEvent(pygame.KEYUP, key=key))
        self.key_sequence.append({key: False})
    
    def press_keys_simultaneously(self, keys: List[int], duration_frames: int = 1):
        """Simulate pressing multiple keys simultaneously."""
        # All keys down
        for key in keys:
            self.event_sequence.append(MockPygameEvent(pygame.KEYDOWN, key=key))
        
        # Hold state
        for frame in range(duration_frames):
            key_state = {key: True for key in keys}
            self.key_sequence.append(key_state)
        
        # All keys up
        for key in keys:
            self.event_sequence.append(MockPygameEvent(pygame.KEYUP, key=key))
        self.key_sequence.append({key: False for key in keys})
    
    def add_pause(self, frames: int = 1):
        """Add frames with no input."""
        for _ in range(frames):
            self.key_sequence.append({})
    
    def get_events(self) -> List[MockPygameEvent]:
        """Get the sequence of events."""
        return self.event_sequence
    
    def get_key_states(self) -> List[Dict[int, bool]]:
        """Get the sequence of key states."""
        return self.key_sequence
    
    def clear(self):
        """Clear all sequences."""
        self.key_sequence.clear()
        self.event_sequence.clear()


# Commonly used mock configurations
@contextlib.contextmanager
def minimal_pygame_mocks():
    """Context manager for minimal pygame mocking (display and time only)."""
    with PygameMockContext(mock_font=False, mock_image=False, 
                          mock_event=False, mock_draw=False) as mocks:
        yield mocks


@contextlib.contextmanager
def full_pygame_mocks():
    """Context manager for complete pygame mocking."""
    with PygameMockContext() as mocks:
        yield mocks


@contextlib.contextmanager
def headless_pygame_mocks():
    """Context manager for headless testing (no display)."""
    with PygameMockContext(mock_display=False) as mocks:
        yield mocks

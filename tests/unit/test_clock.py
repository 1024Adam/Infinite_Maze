"""
Unit tests for the Clock class in Infinite Maze.

These tests verify the functionality of the Clock class including
timing, tick management, FPS tracking, and time formatting.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, MagicMock

from infinite_maze.core.clock import Clock


class TestClockInitialization:
    """Test Clock class initialization."""
    
    def test_clock_init(self):
        """Test clock initialization."""
        clock = Clock()
        
        # Check initial values
        assert clock.getMillis() == 0
        assert clock.getPrevMillis() == 0
        assert clock.getTicks() == 0
        assert clock.getSeconds() == 0
        assert clock.getPrevSeconds() == 0
    
    def test_clock_pygame_integration(self):
        """Test that clock initializes pygame Clock."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Should create pygame Clock instance
            mock_clock_class.assert_called_once()
            assert clock.time == mock_clock_instance


class TestClockTiming:
    """Test Clock timing functionality."""
    
    def test_update_advances_time(self):
        """Test that update advances time."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_instance.get_time.return_value = 16  # 60 FPS
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            initial_millis = clock.getMillis()
            
            clock.update()
            
            # Time should have advanced
            assert clock.getMillis() > initial_millis
            assert clock.getPrevMillis() == initial_millis
    
    def test_multiple_updates(self):
        """Test multiple consecutive updates."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_instance.get_time.return_value = 16
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Perform multiple updates
            times = []
            for i in range(5):
                times.append(clock.getMillis())
                clock.update()
            
            # Each update should advance time
            for i in range(1, len(times)):
                assert times[i] < clock.getMillis()
    
    def test_tick_counter(self):
        """Test tick counter functionality."""
        clock = Clock()
        
        initial_ticks = clock.getTicks()
        
        # Manually call tick
        clock.tick()
        
        assert clock.getTicks() == initial_ticks + 1
        
        # Multiple ticks
        for i in range(10):
            clock.tick()
        
        assert clock.getTicks() == initial_ticks + 11
    
    def test_update_calls_tick(self):
        """Test that update calls tick method."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_instance.get_time.return_value = 16
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            initial_ticks = clock.getTicks()
            
            clock.update()
            
            # Ticks should have incremented
            assert clock.getTicks() == initial_ticks + 1


class TestClockSeconds:
    """Test Clock seconds calculation."""
    
    def test_seconds_calculation(self):
        """Test seconds calculation from milliseconds."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Test various millisecond values
            test_cases = [
                (0, 0),      # 0ms = 0s
                (500, 0),    # 500ms = 0s
                (1000, 1),   # 1000ms = 1s
                (1500, 1),   # 1500ms = 1s
                (2000, 2),   # 2000ms = 2s
                (59000, 59), # 59s
                (60000, 0),  # 60s = 0s (modulo 60)
                (61000, 1),  # 61s = 1s (modulo 60)
            ]
            
            for millis, expected_seconds in test_cases:
                clock.millis = millis
                assert clock.getSeconds() == expected_seconds
    
    def test_prev_seconds_tracking(self):
        """Test previous seconds tracking."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_instance.get_time.return_value = 1000  # 1 second per update
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Initial state
            assert clock.getSeconds() == 0
            assert clock.getPrevSeconds() == 0
            
            # After first update
            clock.update()
            assert clock.getPrevSeconds() == 0  # Previous millis was 0
            
            # After second update
            clock.update()
            # Previous millis should be 1000, so prev seconds should be 1
            prev_seconds = int((clock.getPrevMillis() / 1000) % 60)
            assert clock.getPrevSeconds() == prev_seconds


class TestClockTimeString:
    """Test Clock time string formatting."""
    
    def test_time_string_format(self):
        """Test time string formatting."""
        clock = Clock()
        
        # Test various time values
        test_cases = [
            (0, "00:00"),        # 0 milliseconds
            (1000, "00:01"),     # 1 second
            (30000, "00:30"),    # 30 seconds
            (60000, "01:00"),    # 1 minute
            (90000, "01:30"),    # 1 minute 30 seconds
            (3661000, "61:01"),  # 61 minutes 1 second
        ]
        
        for millis, expected_string in test_cases:
            clock.millis = millis
            assert clock.getTimeString() == expected_string
    
    def test_time_string_zero_padding(self):
        """Test that time string properly zero-pads values."""
        clock = Clock()
        
        # Test single digit values are zero-padded
        clock.millis = 5000  # 5 seconds
        time_string = clock.getTimeString()
        assert time_string == "00:05"
        
        clock.millis = 65000  # 1 minute 5 seconds
        time_string = clock.getTimeString()
        assert time_string == "01:05"


class TestClockFPS:
    """Test Clock FPS functionality."""
    
    def test_get_fps(self):
        """Test getting FPS from pygame clock."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.get_fps.return_value = 60.0
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            fps = clock.getFps()
            
            assert fps == 60.0
            mock_clock_instance.get_fps.assert_called_once()
    
    def test_fps_different_values(self):
        """Test FPS with different return values."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Test different FPS values
            fps_values = [30.0, 45.5, 60.0, 120.0, 0.0]
            
            for expected_fps in fps_values:
                mock_clock_instance.get_fps.return_value = expected_fps
                assert clock.getFps() == expected_fps


class TestClockReset:
    """Test Clock reset functionality."""
    
    def test_reset_clears_values(self):
        """Test that reset clears all values."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.get_time.return_value = 16
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Advance clock state
            clock.millis = 5000
            clock.prevMillis = 4000
            clock.ticks = 100
            
            # Reset
            clock.reset()
            
            # Values should be reset
            assert clock.getMillis() == 0
            assert clock.getPrevMillis() == 0
            assert clock.getTicks() == 0
    
    def test_reset_creates_new_pygame_clock(self):
        """Test that reset creates new pygame Clock instance."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance1 = Mock()
            mock_clock_instance2 = Mock()
            mock_clock_class.side_effect = [mock_clock_instance1, mock_clock_instance2]
            
            clock = Clock()
            assert clock.time == mock_clock_instance1
            
            clock.reset()
            
            # Should create new pygame Clock
            assert mock_clock_class.call_count == 2
            assert clock.time == mock_clock_instance2


class TestClockMillisManagement:
    """Test Clock millisecond management."""
    
    def test_rollback_millis(self):
        """Test rollback milliseconds functionality."""
        clock = Clock()
        
        # Set initial time
        clock.millis = 5000
        
        # Rollback some time
        rollback_amount = 1000
        clock.rollbackMillis(rollback_amount)
        
        assert clock.getMillis() == 4000
    
    def test_rollback_millis_multiple(self):
        """Test multiple rollback operations."""
        clock = Clock()
        
        clock.millis = 10000
        
        # Multiple rollbacks
        clock.rollbackMillis(2000)
        assert clock.getMillis() == 8000
        
        clock.rollbackMillis(3000)
        assert clock.getMillis() == 5000
        
        clock.rollbackMillis(1000)
        assert clock.getMillis() == 4000
    
    def test_rollback_millis_to_negative(self):
        """Test rollback that would result in negative time."""
        clock = Clock()
        
        clock.millis = 1000
        
        # Rollback more than current time
        clock.rollbackMillis(2000)
        
        # Should result in negative time (or whatever the implementation allows)
        assert clock.getMillis() == -1000
    
    def test_get_prev_millis(self):
        """Test getting previous milliseconds."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_instance.get_time.return_value = 100
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Initial state
            assert clock.getPrevMillis() == 0
            
            # After update
            clock.update()
            assert clock.getPrevMillis() == 0  # Was 0 before update
            
            # After another update
            prev_millis = clock.getMillis()
            clock.update()
            assert clock.getPrevMillis() == prev_millis


class TestClockPausedTime:
    """Test Clock paused time functionality."""
    
    def test_millis_paused_tracking(self):
        """Test milliseconds paused tracking."""
        clock = Clock()
        
        # Initial state
        assert clock.getMillisPaused() == 0
        
        # Set paused time
        clock.setMillisPaused(5000)
        assert clock.getMillisPaused() == 5000
        
        # Update paused time
        clock.setMillisPaused(8000)
        assert clock.getMillisPaused() == 8000
    
    def test_millis_paused_reset(self):
        """Test that reset clears paused time."""
        clock = Clock()
        
        clock.setMillisPaused(3000)
        assert clock.getMillisPaused() == 3000
        
        clock.reset()
        assert clock.getMillisPaused() == 0


class TestClockEdgeCases:
    """Test Clock edge cases and error conditions."""
    
    def test_large_time_values(self):
        """Test clock with very large time values."""
        clock = Clock()
        
        # Set very large time value
        large_time = 999999999
        clock.millis = large_time
        
        # Should handle large values gracefully
        seconds = clock.getSeconds()
        time_string = clock.getTimeString()
        
        # Should not raise exceptions
        assert isinstance(seconds, int)
        assert isinstance(time_string, str)
    
    def test_negative_time_values(self):
        """Test clock with negative time values."""
        clock = Clock()
        
        # Set negative time
        clock.millis = -1000
        
        # Should handle negative values
        seconds = clock.getSeconds()
        # The behavior may vary based on implementation
        assert isinstance(seconds, int)
    
    def test_zero_time_handling(self):
        """Test clock behavior at exactly zero time."""
        clock = Clock()
        
        # Ensure time is zero
        clock.millis = 0
        
        assert clock.getMillis() == 0
        assert clock.getSeconds() == 0
        assert clock.getTimeString() == "00:00"
    
    def test_rapid_updates(self):
        """Test rapid consecutive updates."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_instance.get_time.return_value = 1  # Very small time increment
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Perform many rapid updates
            for _ in range(1000):
                clock.update()
            
            # Should handle rapid updates without issues
            assert clock.getTicks() == 1000
            assert clock.getMillis() >= 0


class TestClockIntegration:
    """Integration tests for Clock with game systems."""
    
    def test_clock_with_game_loop_simulation(self):
        """Test clock in simulated game loop."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_instance.get_time.return_value = 16  # ~60 FPS
            mock_clock_instance.get_fps.return_value = 60.0
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            # Simulate game loop
            for frame in range(60):  # 1 second at 60 FPS
                clock.update()
                
                # Check that time advances reasonably
                expected_min_time = frame * 16
                assert clock.getMillis() >= expected_min_time
            
            # After 60 frames, should be around 1 second
            assert clock.getSeconds() <= 1  # Might be 0 due to modulo
            assert clock.getTicks() == 60
    
    def test_clock_pause_simulation(self):
        """Test clock behavior during pause simulation."""
        clock = Clock()
        
        # Advance clock
        clock.millis = 5000
        
        # Simulate pause by rolling back time
        pause_duration = 2000
        clock.rollbackMillis(pause_duration)
        
        assert clock.getMillis() == 3000
        
        # Track paused time
        clock.setMillisPaused(pause_duration)
        assert clock.getMillisPaused() == pause_duration


@pytest.mark.performance
class TestClockPerformance:
    """Performance tests for Clock operations."""
    
    def test_update_performance(self):
        """Test performance of clock updates."""
        with patch('pygame.time.Clock') as mock_clock_class:
            mock_clock_instance = Mock()
            mock_clock_instance.tick.return_value = None
            mock_clock_instance.get_time.return_value = 1
            mock_clock_class.return_value = mock_clock_instance
            
            clock = Clock()
            
            import time
            start_time = time.time()
            
            # Perform many updates
            for _ in range(10000):
                clock.update()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete quickly
            assert duration < 1.0, f"Clock updates took too long: {duration}s"
    
    def test_time_calculation_performance(self):
        """Test performance of time calculations."""
        clock = Clock()
        
        import time
        start_time = time.time()
        
        # Perform many time calculations
        for i in range(10000):
            clock.millis = i * 1000
            _ = clock.getSeconds()
            _ = clock.getTimeString()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 1.0, f"Time calculations took too long: {duration}s"

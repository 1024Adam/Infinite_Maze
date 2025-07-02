# Infinite Maze Test Suite

This directory contains a comprehensive test suite for the Infinite Maze game, providing thorough testing coverage across all game components and functionality.

## Test Structure

```
tests/
├── conftest.py                              # Pytest configuration and shared fixtures
├── fixtures/                                # Test utilities and helpers
│   ├── __init__.py
│   ├── game_fixtures.py                     # Game-specific test fixtures
│   ├── pygame_mocks.py                      # Pygame mocking utilities  
│   └── test_helpers.py                      # Common test helper functions
├── unit/                                    # Unit tests (test individual components)
│   ├── test_player.py                       # Player entity tests
│   ├── test_game.py                         # Game class tests
│   ├── test_clock.py                        # Clock/timing tests
│   ├── test_maze.py                         # Maze generation tests
│   └── test_config.py                       # Configuration tests
├── integration/                             # Integration tests (test component interactions)
│   ├── test_player_maze_interaction.py      # Player-maze collision integration
│   ├── test_game_engine_integration.py      # Game engine integration
│   └── test_asset_loading_integration.py    # Asset loading integration
├── functional/                              # Functional tests (test complete workflows)
│   └── test_gameplay_scenarios.py           # End-to-end gameplay scenarios
└── performance/                             # Performance tests (test speed/efficiency)
    └── test_performance_benchmarks.py       # Performance benchmarks
```

## Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation:
- **Player Tests**: Movement, collision bounds, sprite management
- **Game Tests**: State management, scoring, timing, display coordination  
- **Clock Tests**: Time tracking, FPS monitoring, time formatting
- **Maze Tests**: Line generation, collision boundaries, maze algorithms
- **Config Tests**: Configuration management, asset paths, game settings

### Integration Tests (`tests/integration/`)  
Test interactions between components:
- **Player-Maze Integration**: Collision detection between player and maze walls
- **Game Engine Integration**: Coordination between game loop, player, clock, and maze
- **Asset Loading Integration**: Integration between configuration, asset loading, and game components

### Functional Tests (`tests/functional/`)
Test complete gameplay workflows:
- **Gameplay Scenarios**: End-to-end testing of complete game sessions from start to game over
- **User Interaction Simulation**: Testing complete user interaction patterns
- **Game State Transitions**: Testing pause/resume, game over, reset scenarios

### Performance Tests (`tests/performance/`)
Test performance characteristics:
- **Frame Rate Performance**: FPS stability and consistency testing
- **Memory Performance**: Memory usage and leak detection
- **Computational Performance**: Algorithm efficiency benchmarking  
- **Scalability Testing**: Performance under varying load conditions

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Quick Start

Run the test suite using the provided test runner:

```bash
# Run quick tests (unit + integration)
python run_tests.py

# Run all tests including performance tests
python run_tests.py all

# Run specific test categories
python run_tests.py unit
python run_tests.py integration  
python run_tests.py functional
python run_tests.py performance

# Run with coverage report
python run_tests.py unit --coverage

# Run specific test pattern
python run_tests.py run test_player_movement
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m functional  
pytest -m performance --run-performance

# Run with coverage
pytest --cov=infinite_maze --cov-report=html

# Run specific test file
pytest tests/unit/test_player.py

# Run specific test method
pytest tests/unit/test_player.py::TestPlayerMovement::test_move_right

# Skip slow tests
pytest --skip-slow

# Verbose output
pytest -v
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery settings
- Output formatting options
- Test markers definition
- Logging configuration
- Timeout settings

### Custom Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.functional`: Functional tests  
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.collision`: Collision detection tests

### Command Line Options
- `--skip-slow`: Skip slow-running tests
- `--run-performance`: Enable performance tests (disabled by default)

## Test Utilities

### Fixtures (`tests/fixtures/`)

**game_fixtures.py**: Game-specific test fixtures
- `MockGameEngine`: Mock game engine for testing
- `score_test_scenarios`: Predefined scoring test scenarios
- `collision_test_cases`: Collision detection test cases

**pygame_mocks.py**: Pygame mocking utilities  
- `PygameMockContext`: Comprehensive pygame mocking
- `InputSimulator`: Input event simulation
- `MockPygameSurface`: Display surface mocking

**test_helpers.py**: Common test utilities
- `PerformanceMonitor`: Performance measurement utilities
- `GameStateCapture`: Game state capture and restoration
- Collision detection helper functions

### Mock Framework

The test suite uses comprehensive pygame mocking to enable:
- **Headless Testing**: Tests run without actual pygame display/audio
- **Deterministic Testing**: Controlled timing and input simulation
- **CI/CD Compatibility**: Tests run in automated environments
- **Performance Testing**: Isolated performance measurement

## Writing Tests

### Test Structure Guidelines

```python
class TestComponentName:
    """Test class for ComponentName functionality."""
    
    def test_specific_behavior(self):
        """Test specific behavior with descriptive name."""
        # Arrange
        component = Component()
        
        # Act  
        result = component.method()
        
        # Assert
        assert result == expected_value
```

### Using Fixtures

```python
def test_with_game_fixture(headless_game, test_player):
    """Test using shared fixtures."""
    game = headless_game
    player = test_player
    
    # Test implementation
    assert game.isPlaying()
    assert player.getPosition() == (80, 223)
```

### Performance Testing

```python
@pytest.mark.performance
def test_performance_scenario(self):
    """Test performance with monitoring."""
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Performance test implementation
    
    monitor.stop()
    assert monitor.get_duration() < 1.0  # Should complete quickly
```

## Coverage Reporting

Generate coverage reports:

```bash
# HTML coverage report
pytest --cov=infinite_maze --cov-report=html

# Terminal coverage report  
pytest --cov=infinite_maze --cov-report=term-missing

# XML coverage report (for CI/CD)
pytest --cov=infinite_maze --cov-report=xml
```

Coverage reports are generated in:
- `htmlcov/index.html`: Interactive HTML report
- Terminal output: Summary with missing lines
- `coverage.xml`: Machine-readable XML format

## Continuous Integration

### Test Execution Strategy

**Fast Feedback Loop**:
```bash
# Quick tests for development
pytest -m "unit or integration" --skip-slow
```

**Full Test Suite**:
```bash  
# Complete testing for releases
pytest --run-performance --cov=infinite_maze
```

### Performance Monitoring

Performance tests establish benchmarks for:
- Frame rate stability (target: 60 FPS)
- Memory usage limits (target: <200MB)
- Collision detection speed (target: >1M checks/sec)
- Game loop efficiency (target: <16.67ms/frame)

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure infinite_maze package is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Pygame Display Errors**:
- Tests use mocking by default
- Ensure `headless=True` in test fixtures
- Check `mock_pygame_display` fixture usage

**Performance Test Failures**:
- Performance tests may be sensitive to system load
- Run performance tests in isolation: `pytest -m performance --run-performance`
- Use `--skip-slow` to skip performance tests during development

**Test Dependencies**:
```bash
# Install test requirements if missing
pip install -r requirements-test.txt
```

### Debug Mode

Enable debug logging:
```bash
pytest --log-cli-level=DEBUG
```

## Test Maintenance

### Adding New Tests

1. **Choose appropriate category**: unit, integration, functional, or performance
2. **Use existing fixtures**: Leverage shared fixtures from `conftest.py`
3. **Follow naming conventions**: `test_*` functions, `Test*` classes
4. **Add appropriate markers**: `@pytest.mark.unit`, etc.
5. **Update documentation**: Add test descriptions to this README

### Updating Test Data

Test data and fixtures are centralized in `tests/fixtures/` for easy maintenance.

### Performance Baselines

Performance test baselines should be updated when:
- Game optimization improvements are made
- System requirements change
- New features affect performance characteristics

## Test Metrics

The test suite provides comprehensive metrics:

### Coverage Metrics
- **Line Coverage**: >95% target
- **Branch Coverage**: >90% target  
- **Function Coverage**: 100% target

### Performance Metrics
- **Frame Rate**: 60 FPS target
- **Memory Usage**: <200MB peak
- **Test Execution Time**: <5 minutes full suite
- **Collision Detection**: >1M checks/second

### Test Count by Category
- Unit Tests: ~40 test methods across 5 components
- Integration Tests: ~15 integration scenarios  
- Functional Tests: ~20 end-to-end workflows
- Performance Tests: ~15 performance benchmarks

---

For questions about the test suite, see the main project documentation or contact the development team.

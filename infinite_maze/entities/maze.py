from random import randint
from ..utils.config import config


class Line:
    def __init__(self, start_pos=(0, 0), end_pos=(0, 0), side_a=0, side_b=0):
        self.start = start_pos
        self.end = end_pos
        self.side_a = side_a
        self.side_b = side_b
        self.is_horizontal = start_pos[1] == end_pos[1]

    def get_start(self):
        return self.start

    def set_start(self, new_start):
        self.start = new_start

    def get_end(self):
        return self.end

    def set_end(self, new_end):
        self.end = new_end

    def get_x_start(self):
        return self.start[0]

    def set_x_start(self, new_x):
        self.start = (new_x, self.start[1])

    def get_y_start(self):
        return self.start[1]

    def set_y_start(self, new_y):
        self.start = (self.start[0], new_y)

    def get_x_end(self):
        return self.end[0]

    def set_x_end(self, new_x):
        self.end = (new_x, self.end[1])

    def get_y_end(self):
        return self.end[1]

    def set_y_end(self, new_y):
        self.end = (self.end[0], new_y)

    def get_side_a(self):
        return self.side_a

    def set_side_a(self, side):
        self.side_a = side

    def get_side_b(self):
        return self.side_b

    def set_side_b(self, side):
        self.side_b = side

    def get_is_horizontal(self):
        return self.is_horizontal

    def reset_is_horizontal(self):
        self.is_horizontal = self.start[1] == self.end[1]

    @staticmethod
    def get_x_max(lines):
        if not lines:
            return 0
        x_max = lines[0].get_x_end()  # Initialize with first line's end
        for line in lines:
            line_end = line.get_x_end()
            if line_end > x_max:
                x_max = line_end
        return x_max

    @staticmethod
    def generate_maze(game, width, height, simplicity_factor=None):
        if simplicity_factor is None:
            simplicity_factor = config.MAZE_SIMPLICITY
            
        lines = []
        # Horizontal Line Gen
        for x in range(width * 2):
            side_a = (19 * x) + 1
            side_b = side_a + 1

            x_pos = (config.MAZE_CELL_SIZE * x) + config.MAZE_START_X
            for y in range(1, height - 1):
                y_pos = (config.MAZE_CELL_SIZE * y) + config.Y_MIN
                lines.append(Line((x_pos, y_pos), (x_pos + config.MAZE_CELL_SIZE, y_pos), side_a, side_b))
                side_a = side_b
                side_b += 1
        # Vertical Line Gen
        for y in range(height - 1):
            side_a = y + 1
            side_b = side_a + 19

            y_pos = (config.MAZE_CELL_SIZE * y) + config.Y_MIN
            for x in range(1, width * 2):
                x_pos = (config.MAZE_CELL_SIZE * x) + config.MAZE_START_X
                lines.append(Line((x_pos, y_pos), (x_pos, y_pos + config.MAZE_CELL_SIZE), side_a, side_b))
                side_a = side_b
                side_b += 19

        # Create 'maze' structure, Kruskal's algorithm
        # Randomly remove lines while ensuring all cells are connected
        # (will be complete when all 'cells' are connected to each other)
        sets = []
        while len(sets) != 1:
            length = len(lines)
            line_num = randint(0, length - 1)
            temp_side_a = lines[line_num].get_side_a()
            temp_side_b = lines[line_num].get_side_b()
            if temp_side_a != temp_side_b:
                del lines[line_num]
                for line in lines:
                    if line.get_side_a() == temp_side_b:
                        line.set_side_a(temp_side_a)
                    if line.get_side_b() == temp_side_b:
                        line.set_side_b(temp_side_a)
            sets = []
            for line in lines:
                temp_side_a = line.get_side_a()
                temp_side_b = line.get_side_b()
                if temp_side_a not in sets:
                    sets.append(temp_side_a)
                if temp_side_b not in sets:
                    sets.append(temp_side_b)
                    
        # At this point, we have a "perfect" maze with exactly one path between any two points
        # If simplicity_factor > 0, remove additional walls to create multiple paths
        if simplicity_factor > 0:
            # Calculate how many additional walls to remove
            remaining_walls = len(lines)
            walls_to_remove = int(remaining_walls * simplicity_factor)
            
            # Remove random walls (but not too many to keep the maze structure)
            for _ in range(walls_to_remove):
                if len(lines) > width + height:  # Keep a minimum number of walls
                    line_num = randint(0, len(lines) - 1)
                    del lines[line_num]

        return lines


class Maze:
    """
    Maze class that wraps the line-based maze representation for AI training.
    
    This class provides an interface for the AI environment to interact with the maze
    while using the existing Line-based implementation underneath.
    """

    def __init__(self, game=None, lines=None, simplicity_factor=None):
        """
        Initialize the Maze with either a game instance or a list of lines.
        
        Args:
            game: The Game instance to generate a maze for
            lines: Existing line objects that make up the maze
            simplicity_factor: Override for the maze simplicity factor (0.0 = perfect maze, higher values create easier mazes)
        """
        self.game = game
        self.lines = lines if lines is not None else []
        self.visited = set()  # Track visited cells for the AI
        self.simplicity_factor = simplicity_factor if simplicity_factor is not None else config.MAZE_SIMPLICITY
        
        if game is not None and not lines:
            # Generate a new maze if a game is provided
            self.lines = Line.generate_maze(game, config.MAZE_ROWS, config.MAZE_COLS, self.simplicity_factor)
    
    def regenerate(self, game, simplicity_factor=None):
        """
        Regenerate the maze for a new game
        
        Args:
            game: The Game instance to generate a maze for
            simplicity_factor: Optional override for the maze simplicity factor
        """
        self.game = game
        if simplicity_factor is not None:
            self.simplicity_factor = simplicity_factor
        self.lines = Line.generate_maze(game, config.MAZE_ROWS, config.MAZE_COLS, self.simplicity_factor)
        self.visited = set()
    
    def get_lines(self):
        """Get all lines in the maze"""
        return self.lines
    
    def update_lines(self, lines):
        """Update the maze with new lines"""
        self.lines = lines
    
    def is_wall(self, x, y):
        """
        Check if there's a wall at the given coordinates.
        This is used by the AI to detect collision points.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if a wall is present at (x,y), False otherwise
        """
        # Wall detection logic checks against maze lines
        # This is an approximate check to see if the point is very close to any wall line
        buffer = 5  # Increased buffer for more strict collision detection
        player_size = 10  # Consider player size for collision detection
        
        # Check for collisions with wall lines
        for line in self.lines:
            if line.get_is_horizontal():
                # Horizontal line check - expand collision area to account for player size
                if (abs(y - line.get_y_start()) < buffer + (player_size/2) and
                    x >= line.get_x_start() - buffer - (player_size/2) and 
                    x <= line.get_x_end() + buffer + (player_size/2)):
                    return True
            else:
                # Vertical line check - expand collision area to account for player size
                if (abs(x - line.get_x_start()) < buffer + (player_size/2) and
                    y >= line.get_y_start() - buffer - (player_size/2) and 
                    y <= line.get_y_end() + buffer + (player_size/2)):
                    return True
        
        # Also check game boundary walls
        if self.game:
            if (x < config.X_MIN + buffer or
                y < config.Y_MIN + buffer or
                y > config.Y_MAX - buffer):
                return True
            
        return False
        
    def check_collision(self, x, y, dx, dy):
        """
        Check if moving from (x,y) by (dx,dy) would result in a collision.
        Performs multiple checks along the path to prevent tunneling through walls.
        
        Args:
            x: Starting X coordinate
            y: Starting Y coordinate
            dx: X movement amount
            dy: Y movement amount
            
        Returns:
            True if collision would occur, False otherwise
        """
        # Check multiple points along the path to prevent tunneling
        steps = max(1, int(max(abs(dx), abs(dy)) / 2))
        
        for i in range(steps + 1):
            # Calculate intermediate position
            t = i / steps if steps > 0 else 0
            check_x = x + dx * t
            check_y = y + dy * t
            
            # Check for wall at this position
            if self.is_wall(check_x, check_y):
                return True
                
        return False
    
    def mark_visited(self, x, y):
        """Mark a cell as visited (for exploration tracking)"""
        # Convert to grid coordinates for tracking
        grid_x = int(x // config.MAZE_CELL_SIZE)
        grid_y = int(y // config.MAZE_CELL_SIZE)
        self.visited.add((grid_x, grid_y))
    
    def is_visited(self, x, y):
        """Check if a cell has been visited"""
        grid_x = int(x // config.MAZE_CELL_SIZE)
        grid_y = int(y // config.MAZE_CELL_SIZE)
        return (grid_x, grid_y) in self.visited

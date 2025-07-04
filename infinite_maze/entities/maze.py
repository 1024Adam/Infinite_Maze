from random import randint
from ..utils.config import config


class Line:
    def __init__(self, startPos=(0, 0), endPos=(0, 0), sideA=0, sideB=0):
        self.start = startPos
        self.end = endPos
        self.sideA = sideA
        self.sideB = sideB
        self.isHorizontal = startPos[1] == endPos[1]

    def getStart(self):
        return self.start

    def setStart(self, newStart):
        self.start = newStart

    def getEnd(self):
        return self.end

    def setEnd(self, newEnd):
        self.end = newEnd

    def getXStart(self):
        return self.start[0]

    def setXStart(self, newX):
        self.start = (newX, self.start[1])

    def getYStart(self):
        return self.start[1]

    def setYStart(self, newY):
        self.start = (self.start[0], newY)

    def getXEnd(self):
        return self.end[0]

    def setXEnd(self, newX):
        self.end = (newX, self.end[1])

    def getYEnd(self):
        return self.end[1]

    def setYEnd(self, newY):
        self.end = (self.end[0], newY)

    def getSideA(self):
        return self.sideA

    def setSideA(self, side):
        self.sideA = side

    def getSideB(self):
        return self.sideB

    def setSideB(self, side):
        self.sideB = side

    def getIsHorizontal(self):
        return self.isHorizontal

    def resetIsHorizontal(self):
        self.isHorizontal = self.start[1] == self.end[1]

    @staticmethod
    def getXMax(lines):
        if not lines:
            return 0
        xMax = lines[0].getXEnd()  # Initialize with first line's end
        for line in lines:
            lineEnd = line.getXEnd()
            if lineEnd > xMax:
                xMax = lineEnd
        return xMax

    @staticmethod
    def generateMaze(game, width, height):
        lines = []
        # Horizontal Line Gen
        for x in range(width * 2):
            sideA = (19 * x) + 1
            sideB = sideA + 1

            xPos = (config.MAZE_CELL_SIZE * x) + game.X_MAX
            for y in range(1, height - 1):
                yPos = (config.MAZE_CELL_SIZE * y) + game.Y_MIN
                lines.append(Line((xPos, yPos), (xPos + config.MAZE_CELL_SIZE, yPos), sideA, sideB))
                sideA = sideB
                sideB += 1
        # Vertical Line Gen
        for y in range(height - 1):
            sideA = y + 1
            sideB = sideA + 19

            yPos = (config.MAZE_CELL_SIZE * y) + game.Y_MIN
            for x in range(1, width * 2):
                xPos = (config.MAZE_CELL_SIZE * x) + game.X_MAX
                lines.append(Line((xPos, yPos), (xPos, yPos + config.MAZE_CELL_SIZE), sideA, sideB))
                sideA = sideB
                sideB += 19

        # Create 'maze' structure
        # (will be complete when all 'cells' are connected to each other)
        sets = []
        while len(sets) != 1:
            length = len(lines)
            lineNum = randint(0, length - 1)
            tempSideA = lines[lineNum].getSideA()
            tempSideB = lines[lineNum].getSideB()
            if tempSideA != tempSideB:
                del lines[lineNum]
                for line in lines:
                    if line.getSideA() == tempSideB:
                        line.setSideA(tempSideA)
                    if line.getSideB() == tempSideB:
                        line.setSideB(tempSideA)
            sets = []
            for line in lines:
                tempSideA = line.getSideA()
                tempSideB = line.getSideB()
                if tempSideA not in sets:
                    sets.append(tempSideA)
                if tempSideB not in sets:
                    sets.append(tempSideB)

        return lines


class Maze:
    """
    Maze class that wraps the line-based maze representation for AI training.
    
    This class provides an interface for the AI environment to interact with the maze
    while using the existing Line-based implementation underneath.
    """
    
    def __init__(self, game=None, lines=None):
        """
        Initialize the Maze with either a game instance or a list of lines.
        
        Args:
            game: The Game instance to generate a maze for
            lines: Existing line objects that make up the maze
        """
        self.game = game
        self.lines = lines if lines is not None else []
        self.visited = set()  # Track visited cells for the AI
        
        if game is not None and not lines:
            # Generate a new maze if a game is provided
            self.lines = Line.generateMaze(game, config.MAZE_ROWS, config.MAZE_COLS)
    
    def regenerate(self, game):
        """Regenerate the maze for a new game"""
        self.game = game
        self.lines = Line.generateMaze(game, config.MAZE_ROWS, config.MAZE_COLS)
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
            if line.getIsHorizontal():
                # Horizontal line check - expand collision area to account for player size
                if (abs(y - line.getYStart()) < buffer + (player_size/2) and
                    x >= line.getXStart() - buffer - (player_size/2) and 
                    x <= line.getXEnd() + buffer + (player_size/2)):
                    return True
            else:
                # Vertical line check - expand collision area to account for player size
                if (abs(x - line.getXStart()) < buffer + (player_size/2) and
                    y >= line.getYStart() - buffer - (player_size/2) and 
                    y <= line.getYEnd() + buffer + (player_size/2)):
                    return True
        
        # Also check game boundary walls
        if self.game:
            if (x < self.game.X_MIN + buffer or
                y < self.game.Y_MIN + buffer or
                y > self.game.Y_MAX - buffer):
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

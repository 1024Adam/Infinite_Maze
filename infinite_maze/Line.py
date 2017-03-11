class Line:
    def __init__(self):
        self.start = (0, 0)
        self.end = (0, 0)
        self.sideA = 0
        self.sideB = 0

    def __init__(self, startPos, endPos, sideA, sideB):
        self.start = startPos
        self.end = endPos
        self.sideA = sideA
        self.sideB = sideB

    def getStart(self):
        return (self.start)

    def setStart(self, newStart):
        self.start = newStart

    def getEnd(self):
        return (self.end)

    def setEnd(self, newEnd):
        self.end = newEnd

    def getXStart(self):
        return (self.start[0])

    def setXStart(self, newX):
        self.start = (newX, self.start[1])

    def getYStart(self):
        return (self.start[1])

    def setYStart(self, newY):
        self.start = (self.start[0], newY)

    def getXEnd(self):
        return (self.end[0])

    def setXEnd(self, newX):
        self.end = (newX, self.end[1])

    def getYEnd(self):
        return (self.end[1])

    def setYEnd(self, newY):
        self.end = (self.end[0], newY)

    @staticmethod
    def generateMaze(game, width, height):
        lines = []
        #side = 0
        # Horizontal Line Gen
        for x in range(width):
            xPos = (22 * x) + game.getXMax()
            for y in range(1, height - 1):
                yPos = (22 * y) + game.getYMin()
                lines.append(Line((xPos, yPos), (xPos + 22, yPos), 0, 0))
        # Vertical Line Gen
        for y in range(height - 1):
            yPos = (22 * y) + game.getYMin()
            for x in range(1, width):
                xPos = (22 * x) + game.getXMax()
                lines.append(Line((xPos, yPos), (xPos, yPos + 22), 0, 0))

        return (lines)

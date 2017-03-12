from random import randint

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

    def getSideA(self):
        return (self.sideA)

    def setSideA(self, side):
        self.sideA = side

    def getSideB(self):
        return (self.sideB)

    def setSideB(self, side):
        self.sideB = side

    @staticmethod
    def generateMaze(game, width, height):
        lines = []
        # Horizontal Line Gen
        for x in range(width):
            sideA = (19 * x) + 1
            sideB = sideA + 1

            xPos = (22 * x) + game.getXMax()
            for y in range(1, height - 1):
                yPos = (22 * y) + game.getYMin()
                lines.append(Line((xPos, yPos), (xPos + 22, yPos), sideA, sideB))
                sideA = sideB
                sideB += 1
        # Vertical Line Gen
        for y in range(height - 1):
            sideA = y + 1
            sideB = sideA + 19

            yPos = (22 * y) + game.getYMin()
            for x in range(1, width):
                xPos = (22 * x) + game.getXMax()
                lines.append(Line((xPos, yPos), (xPos, yPos + 22), sideA, sideB))
                sideA = sideB
                sideB += 19

        sets = []
        length = len(lines)
        while (len(sets) != 1):
        #count = 0
        #while (count < 400):
            #count += 1
            length = len(lines)
            lineNum = randint(0, length - 1)
            tempSideA = lines[lineNum].getSideA()
            tempSideB = lines[lineNum].getSideB()
            if(tempSideA != tempSideB):
                del lines[lineNum]
                for line in lines:
                    if (line.getSideA() == tempSideB):
                        line.setSideA(tempSideA)
                    elif (line.getSideB() == tempSideB):
                        line.setSideB(tempSideA)
            sets = []
            for line in lines:
                tempSideA = line.getSideA()
                tempSideB = line.getSideB()
                if (tempSideA not in sets):
                    sets.append(tempSideA)
                if (tempSideB not in sets):
                    sets.append(tempSideB)

        return (lines)

class Line:
    def __init__(self):
        self.start = (0, 0)
        self.end = (0, 0)

    def __init__(self, startPos, endPos):
        self.start = startPos
        self.end = endPos

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

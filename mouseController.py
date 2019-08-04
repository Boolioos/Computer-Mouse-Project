import pyautogui

############### GESTURE REFERENCE ###############
# point = click
# fist = click and hold
# swing = right click
# palm + up/down movement = scroll
#################################################

class mouseMover():
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.deskSizeX, self.deskSizeY = pyautogui.size()[0], pyautogui.size()[1]
        self.ratioX, self.ratioY = self.deskSizeX/self.windowSize[0], self.deskSizeY/self.windowSize[1]
        self.currentPos = pyautogui.position()
        self.rightClickFlag = False
        self.holdFlag = False
        
    # have to remember the X and Y coordinates are messed up
    def moveTo(self, xDelta = None, yDelta = None):
        newPosX = xDelta * self.ratioX
        newPosY = yDelta * self.ratioY
        self.currentPos = (newPosX, newPosY)
        pyautogui.moveTo(newPosX, newPosY)
            
    def click(self):
        self.rightClickFlag = False
        pyautogui.click()
        
    def hold(self):
        self.holdFlag = True
        pyautogui.mouseDown()
        
    def rightClick(self):
        # no need to right click numerous times
        if not self.rightClickFlag:
            self.rightClickFlag = True
            pyautogui.rightClick()
            
    def halt(self):
        self.holdFlag = False
        self.rightClickFlag = False
        pyautogui.mouseUp()
            
    def executeGesture(self, gesture, move = None):
        if move:
            self.moveTo(move[0], move[1])
            
        if gesture == "swing": 
            self.rightClick()
    
        elif gesture == "ok":
            self.hold()
            
        elif gesture == "palm":
            self.halt()
        
        elif gesture == "point":
            self.click()
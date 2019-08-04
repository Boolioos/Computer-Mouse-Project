import numpy as np
import cv2
from backgroundSubController import showVideo
from backgroundSubController import WINDOW_NAME

def captureAndSave(gray=False, device=0, train = "aquaman", extension = "./training/aquamanBS/"):
    
    capture = cv2.VideoCapture(device)
    capWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow(WINDOW_NAME)
    
    counter = 0
    currentBGS = 0 # stores the amount of times one BGS cycle is used
    while(capture.isOpened()):
        
        ret, frame = capture.read()                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ############### BACKGROUND SUBTRACTION ###############
        
        if currentBGS == 0:
            print ("restarting calibration...")
            bgMu = frame # initialized with unchanged intensity
            d = abs(np.subtract(bgMu, frame))
            sigmaSq = np.ones(bgMu.shape) # starting with sigma = 1
            
        if currentBGS < 100:
            rho = 0.01
            bgMu = rho*frame + (1 - rho)*bgMu
        
        elif currentBGS == 100:
            print ("calibration finished!")
            
        currentBGS = currentBGS + 1
        
        d = abs(np.subtract(bgMu, frame))
        sigmaSq = (d**2)*rho + (1 - rho)*sigmaSq        
            
        foreground = d/np.sqrt(sigmaSq)
        foreground[foreground > 1.5] = 255
        foreground[foreground <= 1.5] = 0
        showVideo(foreground)  
        
        press = cv2.waitKey(1) & 0xFF  
        
        if press == ord('b'):
            currentBGS = 0 # recalibrate
            
        if press == ord('q'):
            break
        
        if press == ord('s'):
            cv2.imwrite(extension + train + str(counter) + ".jpg", foreground) # fgmask for background subtracted
            counter = counter + 1
        
    capture.release()
    cv2.destroyAllWindows()

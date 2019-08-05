################## IMPORTS #####################
import numpy as np
import cv2
from keras.models import load_model
from mouseController import mouseMover

################# CONSTANTS #####################
#Wren's constant
RHO = 0.01

#The gestures avialable currently
CLASSES = ["swing", "ok", "palm", "point"]

#Calibration constants
CALIB_WIDTH = 200
CALIB_HEIGHT = 250

#bounding box dimension for the tracker
BOX_DIM = 150

#Sample constant
SAMPLE_FRAMES = 40

WINDOW_NAME = "Real time video"
############### UTILITY FUNCTIONS ###############
def showVideo(frame, gray=False):
    ''' Displays each from of the video'''
    #For grayscale, set gray to True
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    cv2.imshow(WINDOW_NAME,frame)

def separateBox(frame, topLeft, bottomRight):
    ''' Only take image within the box to classify (Bounding box contains the hand)'''
    #Take the image within the bounding box and padd the rest with 0s
    boxImage = cv2.erode(frame, np.ones((3, 3)), 1)#np.zeros(frame.shape)
    
    #Set to black except for cells in given box specified by topLeft, bottomRight
    boxImage[topLeft[0]:bottomRight[0]+1, 
             topLeft[1]:bottomRight[1]+1] = frame[topLeft[0]:bottomRight[0]+1, 
                    topLeft[1]:bottomRight[1]+1]
    
    return boxImage

def checkOutOfBounds(x,y, frame):
    ''' Make sure the mouse doesnt go out of bounds'''
    #Check if it is out of bounds (being too low)
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    
    #Check if it is out of bounds (being too high)
    x = frame.shape[1] if x > frame.shape[1] else x
    y = frame.shape[0] if y > frame.shape[0] else y
    
    return x,y
    
############### MODEL LOADING ###############    
#This model is trained to classify the gesture
hand_model = load_model("model_keras_final.h5")

############### BACKGROUND SUBTRACTION (WREN) ###############
def computeBGS(reset, frame, bgMu, dist, sigmaSq):
    ''' Compute the background subtraction based on Wren's algorithm'''
    #Display purposes
    if reset:
        print("Starting calibration...")
    
    #Wren's running Gaussian average
    else:       
        #bgMu is the mean of the background
        bgMu = RHO*frame + (1 - RHO)*bgMu
        dist = abs(np.subtract(bgMu, frame))
        sigmaSq = (dist**2)*RHO + (1 - RHO)*sigmaSq 
        
    return bgMu, dist, sigmaSq
    
############### TRACKER CALIBRATION ###############     
def calibration(capture, timer=100):
    ''' Opens another window where the user places their hand in the calibration
        box in order to begin tracking their hand. The hand is tracked after a
        timer runs out'''
    #Intilizing variables
    text = "Place hand within the box"
    calibWindow = "Calibration Screen"
    colorGREEN = (0,255,0)
    cv2.namedWindow(calibWindow)
    
    #Create a new window only for calibration
    while True:
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Top left and bottom right of the bounding box
        tl, br = calculateBox(frame)
        
        if not ret:
            break;
        
        #Wait until the timer runs out
        if timer > 0:
            #Display purposes
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) #Green box over the image :)
            cv2.rectangle(frame, tl, br, colorGREEN, 2)
            cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_SIMPLEX, 1, colorGREEN, 2)
            cv2.imshow(calibWindow, frame)
            
            timer-=1
        
        #If timer is 0 return the calibration
        else:
            cv2.destroyWindow(calibWindow)
            return tl, br
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    #if Q is pressed or ret is false before timer counts down
    cv2.destroyWindow(calibWindow)
    
def calculateBox(frame, width=CALIB_WIDTH, height=CALIB_HEIGHT):
    ''' Returns the calibration bounding box given a desired height/width '''
    x,y = (frame.shape[1] - width)//2, (frame.shape[0] - height)//2
    x2,y2 = x + width, y + height
    return (x,y), (x2,y2)
    
def createTracker(frame, positional):
    ''' Creates a new MOSSE tracker instance. The MOSSE tracker works with a 
        mini "training" phase, in which it averages out a bunch of frames and 
        convolves it with a filter premised on the frame average. The area with 
        the lowest sum squared error is chosen as the new place for the object.'''
    tracker = cv2.TrackerMOSSE_create()  
    trackSuccess = tracker.init(frame, positional)
    return tracker, trackSuccess    

############### CONTROLLER HELPER FUNCTIONS ###############     
def checkAndCreateTracker(capture, frame):
    ''' Checks and creates a new tracker if the calibration is successful'''
    boundingBox = None
    
    #Prints are used For display purposes
    print("Hand calibration starting...")
    while(not boundingBox):
        boundingBox = calibration(capture)
    cv2.rectangle(frame, boundingBox[0], boundingBox[1], (0, 255, 0), 2)
    print("Hand calibration done!")  
    
    #Create a  new tracker
    trackerParms = (boundingBox[0][0],boundingBox[0][1], CALIB_WIDTH, CALIB_HEIGHT)
    tracker, trackSuccess = createTracker(frame, trackerParms)
    return tracker, trackSuccess, boundingBox

############### MAIN CONTROLLER ###############
def main(gray=False, device=1, model=hand_model):
    ''' Starts the webcame, and performs all the mentioned operations on each
        frame of the video. After getting the frame, classify the gesture 
        present in the frame and perform the mouse operation.'''
    #Video Capture and properties
    capture = cv2.VideoCapture(device)  
    capWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cameraDim = (capHeight, capWidth, 1)
    cv2.namedWindow(WINDOW_NAME)

    #Keeps track of the previous and current gestures
    #gesture, prevGesture = None, None
    gesture = None
    
    #The timers recalibrate background subtraction and Tracker after certain time
    calibrateBGS, calibrateHand = True, True
    
    #Tracker related variables
    boundingBox, trackSuccess = None, False

    #Intialize the mouse object to use as a controller
    print((capWidth, capHeight))
    mouseController = mouseMover((capWidth, capHeight))
    
    #Runtime, used to sameple gestures.
    cycleTimer = 1
    
    #For demo purposes 
    demoMouse = False
    text = 'Gesture Detected: NONE'
    
    while(capture.isOpened()):
        #Camera poperties
        ret, frame = capture.read()            
        colorframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #check whether to calibrate the Background subtraction, sigma = 1 to start off
        if calibrateBGS:
            bgMu, dist, sigmaSq = computeBGS(True, 
                                          frame, 
                                          frame, 
                                          np.zeros(frame.shape), 
                                          np.ones(frame.shape))
            calibrateBGS = False
            
        #Compute the background subtraction every frame to normalize the frame
        bgMu, dist, sigmaSq = computeBGS(False, frame, bgMu, dist, sigmaSq)
        foreground = dist/np.sqrt(sigmaSq)
        foreground[foreground > 1.5] = 255
        foreground[foreground <= 1.5] = 0 
        
        #print(calibrateHand)
        #Check whether to recalbirate the bounding box of the tracker
        if calibrateHand:
           tracker, trackSuccess, boundingBox = checkAndCreateTracker(capture, colorframe)
           calibrateHand = False
           continue
        
        # if object was successfully tracked, update bounding box
        trackSuccess, boxDim = tracker.update(colorframe)
        
        if trackSuccess: 
            x,y = int(boxDim[0]), int(boxDim[1])
            boundingBox = ((x, y), (x+int(boxDim[2]), y+int(boxDim[3])))
            cv2.rectangle(frame, boundingBox[0], boundingBox[1], (0, 255, 0), 2)
        
            #Every SAMPLE_FRAMES, sample for gesture
            #if ((cycleTimer%SAMPLE_FRAMES) == 0): <--------------------ORIGINAL
            if (demoMouse): #<-----------------------------------------FOR DEMO
                # perform super background subtraction
                foreground = separateBox(foreground,boundingBox[0],boundingBox[1])
                
                #Foreground is reshaped to match the input for the model
                modelInput = np.array([np.reshape(foreground, cameraDim)])
                #Predict the hand gesture
                prediction = hand_model.predict(modelInput)
                predict = prediction[0].argmax()
                
                #Store the previous gesture to compare whether a new gesture is made
                #prevGesture = gesture
                gesture = CLASSES[predict]
                text = 'Gesture Detected: ' + gesture
                
                '''
                #readjust the tracker for the new gesture
                if gesture != prevGesture:
                    trackerParms = (boundingBox[0][0],boundingBox[0][1], CALIB_WIDTH, CALIB_HEIGHT)
                    tracker, trackSuccess = createTracker(colorframe, trackerParms)
                '''
                
                #For countour finding
                foreground = foreground.astype(np.uint8)
                
                #Finally execture the command
                mouseController.executeGesture(gesture)#, (boundingBox[0][0] + boundingBox[1][0]/2, boundingBox[1][1]))  
                
                #For demo purposes
                demoMouse = False
        
        #If the tracker couldn't detect anything, restart
        else:
            trackerParms = (boundingBox[0][0],boundingBox[0][1], CALIB_WIDTH, CALIB_HEIGHT)
            tracker, trackSuccess = createTracker(colorframe, trackerParms)
                    
        #How often to check and execute a command and background subtraction
        cycleTimer += 1
        
        #########################
        ##Display For the Video##
        #########################
        #FOR DEMO PURPOSES
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        x,y = checkOutOfBounds(((boundingBox[0][0]+boundingBox[1][0])//2),
                               ((boundingBox[0][1]+boundingBox[1][1])//2), frame)
        
        cv2.circle(frame, (x,y) , 10, (0,0,255), -1)
        cv2.putText(frame, text, 
                            (200,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if (cycleTimer%2 == 0):
            mouseController.moveTo(x,y)
        showVideo(frame)  
        
        #########################
        ##VIDEO CAPTURE OPTIONS##
        #########################
        #For Demo purposes 
        if cv2.waitKey(1) & 0xFF == ord('a'):
            demoMouse = True
        
        #Reset the command        
        if cv2.waitKey(1) & 0xFF == ord('r'):
            mouseController.executeGesture('palm')
            text = 'Gesture Detected: NONE'
            
        #Recalibrate Background subtraction
        if cv2.waitKey(1) & 0xFF == ord('b'):
            calibrateBGS = True
            
        #Recalibrate hand 
        if cv2.waitKey(1) & 0xFF == ord('c'):
            calibrateHand = True 
        
        #Quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()

############### RUNNING ###############
        
if __name__ == "__main__":
    main()
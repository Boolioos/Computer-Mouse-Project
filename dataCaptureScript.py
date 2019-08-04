####NOTE: This is ascript for generating the pictures quickly and placing them in the correct folder####
from trainingDataGenerator import captureAndSave

#Capture and store pictures for the training data 
captureAndSave(False, 0, "swing", "./training/swingBS/")
captureAndSave(False, 0, "ok", "./training/okBS/")
captureAndSave(False, 0, "palm", "./training/palmBS/")
captureAndSave(False, 0, "point", "./training/pointBS/")

#Capture and store pictures for the validation data 
captureAndSave(False, 0, "swing", "./validation/swingBS/")
captureAndSave(False, 0, "ok", "./validation/okBS/")
captureAndSave(False, 0, "palmn", "./validation/palmBS/")
captureAndSave(False, 0, "point", "./validation/pointBS/")
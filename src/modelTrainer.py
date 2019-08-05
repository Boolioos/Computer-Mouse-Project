import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

### REQUIRES pip install argparse and pip install keras
#import tensorflow
import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.utils import to_categorical

############### SETTING UP ENVIRONMENT ###############

trainDir = "./training/" # these directories must exist in the same src folder 
valDir = "./validation/"

def classifyData(direct):
    
    images = []
    label = []
    dirent = os.listdir(direct)
    for classes in range(len(dirent)):
        #print(dirent[classes])
        files = os.listdir(direct + dirent[classes])
        for data in files:
            gray = cv2.cvtColor(cv2.imread(direct + dirent[classes] + "/" + data), cv2.COLOR_BGR2GRAY)
            #print(data)
            stack = np.reshape(gray, (480, 640, 1))
            images.append(stack)
            label.append(classes)
    label = to_categorical(label)
    return images, label

def collateClasses(trainDir = trainDir, valDir = valDir):
    tImages, tLabels = classifyData(trainDir)
    tImages = np.array(tImages)
    tLabels = np.array(tLabels)
    
    vImages, vLabels = classifyData(valDir)
    vImages = np.array(vImages)
    vLabels = np.array(vLabels)
    
    return tImages, tLabels, vImages, vLabels

############### CNN MODEL (BASED ON ONLINE MATERIAL- JULIUS) ###############

def CNN():
    
    model = keras.models.Sequential()
    model.add(Conv2D(32, 3, activation = "relu", input_shape=(480, 640, 1)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64, 3, activation = "relu"))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, 3, activation = "relu"))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, 3, activation = "relu"))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation ='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) # automatically uses categorical accuracy
    
    return model

tImages, tLabels, vImages, vLabels = collateClasses()

model = CNN()

############### RUNNING ###############

from keras.utils import np_utils
batchSize = 16

#tLabels=np_utils.to_categorical(tLabels)
#vLabels=np_utils.to_categorical(vLabels)

trainGen = ImageDataGenerator(rescale = 1./255,
                              rotation_range = 40,
                                   #width_shift_range = 0.2,
                                   #height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
valGen = ImageDataGenerator(rescale = 1./255)

trainGeneration = trainGen.flow(tImages, tLabels, batch_size = batchSize)
valGeneration = valGen.flow(vImages, vLabels, batch_size = batchSize)

# optional load function for retraining
model = load_model('model_keras_final.h5') 

train = model.fit_generator(trainGeneration, 
                            steps_per_epoch = len(tLabels)//batchSize,
                            epochs = 35,
                            verbose = 1,
                            validation_data = valGeneration,
                            validation_steps = len(vLabels)//batchSize)

model.save_weights('model_weights_final.h5')
model.save('model_keras_final.h5')

###########################################################################

# heavily based on:
# https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9
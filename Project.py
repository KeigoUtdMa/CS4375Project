from github import Github
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import tensorflow_hub as hub
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#connect to github
g = Github("75229308aac2dcb2943b461cb13665ee6b86e186")
repo = g.get_user().get_repo("CS4375Project")

#set the basic parameter for the learning
epochs = 10
train_image = 1000
test_Image = 400
trainimagedata = np.zeros((train_image, 256, 256, 3))
testimagedata = np.zeros((test_Image, 256, 256, 3))
shape = (256, 256)

trains = repo.get_contents("training_set")
train_file = []
traincategories = []

#read files from training data set
for file in trains:
  trainfilename = file.name
  traincategory = trainfilename.split('.')[0]
  train_file.append(trainfilename)
  if (traincategory == "cat"):
    traincategories.append(0)
  else:
    traincategories.append(1)

traincategories = to_categorical(traincategories)
traincount = 0

for name in train_file:
  trainresponse = requests.get("https://github.com/KeigoUtdMa/CS4375Project/blob/main/training_set/" + name + "?raw=true")
  trainimage = np.array(Image.open(BytesIO(trainresponse.content)).resize(shape))
  trainimagedata[traincount] = trainimage
  traincount += 1

model = Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/4", input_shape = shape + (3,), trainable = False)])
model.add(Dense(2, activation = "softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#image Data Generator
datagen = ImageDataGenerator(
    rotation_range = 15,
    rescale = 1 / 255,
    shear_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1)

tests = repo.get_contents("test_set")
testfilenames = []
testcategories = []
#go thought the 
for file in tests:
  testfilename = file.name
  testcategory = testfilename.split('.')[0]
  testfilenames.append(testfilename)
  if (testcategory == "cat"):
    testcategories.append(0)
  else:
    testcategories.append(1)
testcategories = to_categorical(testcategories)
testcount = 0

for name in testfilenames:
  testresponse = requests.get("https://github.com/KeigoUtdMa/CS4375Project/blob/main/test_set/" + name + "?raw=true")
  testimage = np.array(Image.open(BytesIO(testresponse.content)).resize(shape))
  testimagedata[testcount] = testimage
  testcount += 1


earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

hist = model.fit(datagen.flow(trainimagedata, traincategories), epochs = epochs, validation_data = (testimagedata, testcategories), callbacks = callbacks)
"""
plot for the graph
"""
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["training", "testing"], loc = "upper right")
plt.show()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["training", "testing"], loc = "upper right")
plt.show()
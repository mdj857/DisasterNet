# This file uses transfer learning via the VGG16 convnet to build a model to identify 
# the urgency of each images as coded by Brett

# from keras.models import Sequential
# from scipy.misc import imread
# import matplotlib.pyplot as plt
# import numpy as np
# import keras
# from keras.layers import Dense
 

# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# import numpy as np
# from keras.applications.vgg16 import decode_predictions
# from scipy.misc import imresize

import pandas as pd
from datetime import datetime  
from datetime import timedelta 

images_path = 'coded_images/'
# combine the .csvs into a single dataframe
coder1 = pd.read_csv('Coder 1 CSV File.csv')
coder2 = pd.read_csv('Coder 2 CSV File.csv')
combined = coder1.append(coder2) 

# function that gets the date given an offset that's been coded
def get_date(coded_date):
	date = datetime(2017,8,16) + timedelta(days=coded_date)
	date = date.strftime('X%m-X%d-%Y').replace('X0','X').replace('X','')
	return date

def get_image_name(image_num):
	return 'Image' + str(image_num) + '.jpg'

def get_relative_file_path(date_num, im_num):
	return get_date(date_num)


#preparing the dataset -- assembling the images and their relevant labels into an x and y
df.apply (lambda row: label_race (row),axis=1)


x_img=[]
y_urgency =[]
for index, row in combined.iterrows():
	
	image_number = row['Q3ImageNumber']
    temp_img=image.load_img(images_path+,target_size=(224,224))

    temp_img=image.img_to_array(temp_img)

    train_img.append(temp_img)

#converting train images to array and applying mean subtraction processing

x_img=np.array(x_img) 
train_img=preprocess_input()
# applying the same procedure with the test dataset

# test_img=[]
# for i in range(len(test)):

#     temp_img=image.load_img(test_path+test['filename'][i],target_size=(224,224))

#     temp_img=image.img_to_array(temp_img)

#     test_img.append(temp_img)

# test_img=np.array(test_img) 
# test_img=preprocess_input(test_img)

# loading VGG16 model weights
model = VGG16(weights='imagenet', include_top=False)
# Extracting features from the train dataset using the VGG16 pre-trained model

features_train=model.predict(train_img)
# Extracting features from the train dataset using the VGG16 pre-trained model

features_test=model.predict(test_img)

# flattening the layers to conform to MLP input

train_x=features_train.reshape(49000,25088)
# converting target variable to array

train_y=np.asarray(train['label'])
# performing one-hot encoding for the target variable

train_y=pd.get_dummies(train_y)
train_y=np.array(train_y)
# creating training and validation set

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(train_x,train_y,test_size=0.3, random_state=42)

 

# creating a mlp model
from keras.layers import Dense, Activation
model=Sequential()

model.add(Dense(1000, input_dim=25088, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(500,input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(150,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

# number of output classification categori
model.add(Dense(units=5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# fitting the model 

model.fit(X_train, Y_train, epochs=20, batch_size=128,validation_data=(X_valid,Y_valid))
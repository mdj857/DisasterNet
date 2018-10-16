
# This file uses transfer learning via the VGG16 convnet to build a model to identify 
# the urgency of each images as coded by Brett

import pickle
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Activation

with open('../feat_vecs.pkl', 'rb') as f:
	feature_vecs_old= pickle.load(f)
with open('../y_urgency.pkl', 'rb') as f:
	Y_urgency_old = pickle.load(f)


# In[9]:

print len(feature_vecs_old)
print len(Y_urgency_old)

feature_vecs = []
Y_urgency = []
for a, b in zip(feature_vecs_old, Y_urgency_old):
    if b != u' ':
        feature_vecs.append(a)
        Y_urgency.append(b)
        
print len(feature_vecs)
print len(Y_urgency)

print set(Y_urgency)


# In[10]:

# loading VGG16 model weights
vgg_model = VGG16(weights='imagenet', include_top=False)


# In[14]:

feature_vecs = np.asarray(feature_vecs)
# flattening the layers to conform to MLP input
X=feature_vecs.reshape(2242, 25088)
# converting target variable to array
Y=np.asarray(Y_urgency)
#performing one-hot encoding for the target variable
Y=pd.get_dummies(Y)


# In[26]:

#creating training and validation set

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(X,Y,test_size=0.2, random_state=42)



X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)

model=Sequential()

model.add(Dense(1000, input_dim=25088, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(500,input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(150,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

model.add(Dense(units=5))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=20, batch_size=128,validation_data=(X_valid,Y_valid))

# serialize model to JSON
model_json = model.to_json()
with open("urgency_predictor_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("urgency_predictor_weights.h5")



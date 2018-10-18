# This file uses a MLP with three hidden layers to predict the "relvancy" of an image from a feature vector
# created by passing the corresponding image through the VGG16 Convnet. 

# Created by Matt Johnson 10/18/18

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
from keras.callbacks import EarlyStopping, ModelCheckpoint



with open('../feat_vecs.pkl', 'rb') as f:
	feature_vecs_old= pickle.load(f)


# combine the .csvs into a single dataframe
coder1 = pd.read_csv('../Coder 1 CSV File.csv')
coder2 = pd.read_csv('../Coder 2 CSV File.csv')
combined = coder1.append(coder2) 

Y_time_period_old = combined['Q2TimePeriod'].tolist()
print set(Y_time_period_old)

# ax = combined['Q4Relevancy'].value_counts().plot(kind='bar',
#                                     figsize=(14,8),
#                                     title="Frequency of Relevancy Labels")
# ax.set_xlabel("Labels")
# ax.set_ylabel("Frequency")
# plt.show()

# remove the feature vecs and urgency levels for images with an empty label
feature_vecs = []
Y_time_period = []
for a, b in zip(feature_vecs_old, Y_time_period_old):
    if b != u' ':
        feature_vecs.append(a)
        Y_time_period.append(str(b))
        
print len(Y_time_period)
print set(Y_time_period)

# loading VGG16 model weights
vgg_model = VGG16(weights='imagenet', include_top=False)



feature_vecs = np.asarray(feature_vecs)
# flattening the layers to conform to MLP input
X=feature_vecs.reshape(2246, 25088)
# converting target variable to array
Y=np.asarray(Y_time_period)
#performing one-hot encoding for the target variable
Y=pd.get_dummies(Y)
print Y.shape


#creating training and validation set
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(X,Y,test_size=0.2, random_state=42)



X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)


'''
# model creation and fit 
model=Sequential()
model.add(Dense(1000, input_dim=25088, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)
model.add(Dense(500,input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)
model.add(Dense(150,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)
model.add(Dense(units=3))
model.add(Activation('softmax'))

callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model_timeperiod.h5', monitor='val_loss', save_best_only=True)]

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=20, batch_size=32, callbacks = callbacks, validation_data=(X_valid,Y_valid))

import matplotlib.pyplot as plt

#Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Time Period Classifier Accuracy per Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Time Period Classifier Categorical Cross-entropy Loss per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

from keras.utils import plot_model
plot_model(model, to_file='timeperiod_model.png', show_shapes=True)
'''
from keras.models import load_model
model = load_model('time_period_model.h5')

# get confusion matrix 
import sklearn.metrics
y_pred = model.predict(X_valid)
matrix = sklearn.metrics.confusion_matrix(Y_valid.argmax(axis=1), y_pred.argmax(axis=1))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
       
df_cm = pd.DataFrame(matrix, range(3),
                  range(3))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g')# font size
ax.set_ylabel("Predicted")
ax.set_xlabel("Actual")
plt.show()


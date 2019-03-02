
# This file uses a MLP with three hidden layers to predict the 'relevancy' of an image from a feature vector
# created by passing the corresponding image through the VGG16 Convnet. 

# Created by Matt Johnson 1/27/18

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
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.utils import class_weight



with open('feat_vecs.pkl', 'rb') as f:
	feature_vecs= pickle.load(f)

df = pd.read_csv('Coder1.csv')
Y_relief = df['ContainsMotif10']

feature_vecs = np.asarray(feature_vecs)

print feature_vecs.shape

# flattening the layers to conform to MLP input
X=feature_vecs.reshape(1128, 25088)
# converting target variable to array
Y=np.asarray(Y_relief)
print Y.shape
#performing one-hot encoding for the target variable
Y=pd.get_dummies(Y)


#creating training and validation set
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=42)


X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)

from keras.models import load_model
model = load_model('best_model_relief.h5')


# confusion matrix
import sklearn.metrics
y_pred = model.predict(X_valid).argmax(axis=1)
print y_pred
print Y_valid

from sklearn.metrics import f1_score
print f1_score(np.asarray(Y_valid).argmax(axis=1), y_pred, average='macro')
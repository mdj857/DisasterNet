
# This file uses a MLP with three hidden layers to predict the time period of an image from a feature vector
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

df = pd.read_csv('C1.csv')
Y_tp = df['Q2TimePeriod']

feature_vecs = np.asarray(feature_vecs)

print feature_vecs.shape

# flattening the layers to conform to MLP input
X=feature_vecs.reshape(1128, 25088)
# converting target variable to array
Y=np.asarray(Y_tp)
print Y_tp.shape
#performing one-hot encoding for the target variable
Y=pd.get_dummies(Y)


#creating training and validation set
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=42)


X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)

from sklearn.utils.class_weight import compute_class_weight
# create class weights to combat class imbalance 
y_integers = np.asarray(Y_train).argmax(axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))
print d_class_weights


from sklearn.utils.class_weight import compute_class_weight
# create class weights to combat class imbalance 
y_integers = np.asarray(Y_train).argmax(axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))


# model creation and fit 
model=Sequential()
model.add(Dense(1000, input_dim=25088, activation='sigmoid'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)
model.add(Dense(20,input_dim=1000,activation='relu'))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(20,input_dim=20,activation='relu'))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(20,input_dim=20,activation='relu'))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(units=3))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model_timeperiod.h5', monitor='val_loss', save_best_only=True)]

history = model.fit(X_train, Y_train, epochs=100, batch_size=32, callbacks=callbacks, validation_data=(X_valid,Y_valid), class_weight=d_class_weights)



#Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Time Period Classifier Accuracy per Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('time_period_accuracy.png')
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Time Period Classifier Categorical Cross-entropy Loss per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('time_period_loss.png')
plt.clf()

model.save('time_period_model.h5')

# confusion matrix
import sklearn.metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
y_pred = model.predict(X_valid)
matrix = sklearn.metrics.confusion_matrix(np.asarray(Y_valid).argmax(axis=1), y_pred.argmax(axis=1))       
df_cm = pd.DataFrame(matrix, range(3),
                  range(3))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g')# font size
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.savefig("cm_time_period.png")
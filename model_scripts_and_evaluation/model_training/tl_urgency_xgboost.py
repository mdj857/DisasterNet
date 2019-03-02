
# This file uses gradient-boosted random forest, "XGBOOST" to predict the "urgency" of an image from a feature vector
# created by passing the corresponding image through the VGG16 Convnet. 

# Created by Matt Johnson 01/16/19

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



with open('feat_vecs.pkl', 'rb') as f:
	feature_vecs = pickle.load(f)

df = pd.read_csv('C1.csv')


Y_urgency = df['Q5Urgency']
feature_vecs = np.asarray(feature_vecs)

# flattening the layers to conform to MLP input
X=feature_vecs.reshape(1128, 25088)
# converting target variable to array
Y=np.asarray(Y_urgency)


#creating training and validation set
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(X,Y,test_size=0.1, stratify=Y, random_state=42)


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

print Y_valid.shape

model = XGBClassifier(objective='multi:softmax', gamma=.1, n_estimators=100)
eval_set = [(X_train, Y_train), (X_valid, Y_valid)]
eval_metric = 'merror'
model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
print model.eval_results

#strat k-fold
# skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

# params = {
#         'learning_rate': [.01, .05, .025],
#         'max_depth': [3, 4, 5, 10]
#         }

# grid = GridSearchCV(estimator=model, param_grid=params, scoring='f1_weighted', n_jobs=4, cv=skf.split(X_train,Y_train), verbose=20)
# grid.fit(X_train, Y_train)

# print('\n All results:')
# print(grid.cv_results_)
# print('\n Best estimator:')
# print(grid.best_estimator_)
# print('\n Best score:')
# print(grid.best_score_)
# print('\n Best parameters:')
# print(grid.best_params_)
# results = pd.DataFrame(grid.cv_results_)
# print(results)
# results.to_csv('xgb-grid-search-results-01.csv', index=False)


# make predictions for test data
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_valid)

# evaluate predictions
print "Train acc: " + str(accuracy_score(Y_train, y_pred_train))
print "Test acc: " + str(accuracy_score(Y_valid, y_pred_test))








# This file uses logistic regression to predict the "urgency" of an image from a feature vector
# created by passing the corresponding image through the VGG16 Convnet. 

# Created by Matt Johnson 01/09/19

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np



with open('feat_vecs.pkl', 'rb') as f:
	feature_vecs_old = pickle.load(f)

df = pd.read_csv('C1.csv')
Y_urgency_old = df['Q5Urgency']





# remove the feature vecs and urgency levels for images with an empty label
feature_vecs = []
Y_urgency = []
for a, b in zip(feature_vecs_old, Y_urgency_old):
    if b != u' ':
        feature_vecs.append(a)
        Y_urgency.append(b)



feature_vecs = np.asarray(feature_vecs)

print feature_vecs.shape
# flattening the layers to conform to MLP input
X=feature_vecs.reshape(1127, 25088)
# converting target variable to array
Y=np.asarray(Y_urgency)


#creating training and validation set
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(X,Y,test_size=0.1, stratify=Y, random_state=42)

X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)

# Create Logistic Regression Classifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


solver_options = ['newton-cg', 'lbfgs', 'sag']
class_weight_options = [None, 'balanced']
C = np.logspace(-5, 0, 5, 10, 100)
print C

param_grid = dict(solver = solver_options, class_weight = class_weight_options, C = C)
logit = LogisticRegression(multi_class='multinomial', C = 0.00000003, solver='lbfgs', class_weight=None)
logit_model = logit.fit(X_train, Y_train)

# grid = GridSearchCV(logit, param_grid, cv=5, scoring = 'f1_micro', verbose=20)
# best_model = grid.fit(X_train, Y_train)


# print('Best Class Weight:', best_model.best_estimator_.get_params()['class_weight'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])
# print('Best Optimizer:', best_model.best_estimator_.get_params()['solver'])
import sklearn.metrics


# get confusion matrix 
y_pred_test = logit_model.predict(X_valid)
y_pred_train = logit_model.predict(X_train)

print sklearn.metrics.f1_score(y_true=Y_train, y_pred=y_pred_train, average='weighted')
print sklearn.metrics.f1_score(y_true=Y_valid, y_pred=y_pred_test, average='weighted')
matrix = sklearn.metrics.confusion_matrix(y_true=Y_valid, y_pred=y_pred_test)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
       
df_cm = pd.DataFrame(matrix, range(5),
                  range(5))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g')# font size
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
ax.set_title('Logistic Regression Confusion Matrix')
plt.savefig("cm_urgency_logit.png")






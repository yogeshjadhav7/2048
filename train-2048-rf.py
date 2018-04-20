
# coding: utf-8

# In[1]:


import os
import cv2
import pandas as pd
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

GAME_STATE_FILE_NAME = "game"
GAME_STATE_FILE_EXT = ".csv"
GAMES_DIR = "games/"
PROCESSED_GAMES_DIR = "processed_games/"
MODEL_NAME = "2048_model.h5"
MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVE_COL_NAME = "MOVE"
N_SIZE = 4
N_FILES = len(os.listdir(PROCESSED_GAMES_DIR))
TRAIN_MODEL = True
MODEL_NAME = "model.pkl"

def load_data(file, direc=GAMES_DIR, header=True):
    csv_path = os.path.join(direc, file)
    if header:
        return pd.read_csv(csv_path)
    else:
        return pd.read_csv(csv_path, header=None)


# In[2]:


def get_features_labels(n_file, direc, group_n_games=N_FILES, validation=False):
    x = []
    y = []
    
    if not validation:
        group_n_games = 1
        
    for indx in range(group_n_games):
        
        filename = GAME_STATE_FILE_NAME + str(n_file % N_FILES) + GAME_STATE_FILE_EXT
        n_file = n_file - 1
        
        data = load_data(file=filename, direc=direc)
    
        labels = data[MOVE_COL_NAME].values
        data.drop(MOVE_COL_NAME, axis=1, inplace=True)

        features = data.values
        
        if len(x) == 0:
            x = features
            y = labels
        else:
            x = np.concatenate((x, features), axis=0)
            y = np.concatenate((y, labels), axis=0)
                                              
    return x, y


# In[3]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import log_loss

def custom_score_calculator(ground_truth, predictions):
    error = np.abs(ground_truth - predictions).sum()
    return 10.0 / error

custom_scorer = make_scorer(custom_score_calculator, greater_is_better=True)

features, labels = get_features_labels(0, direc=PROCESSED_GAMES_DIR, validation=True)

parameters = {
    'max_depth' : [(x + 2) for x in range(50)],
    'min_samples_split' : [(x + 2) for x in range(30)],
    'n_estimators' : [(x + 1) for x in range(200)]
}

print("\n\n\nTuned params: ", parameters)
print("Training started ...")
clf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                   param_grid=parameters, 
                   cv=N_FILES,
                   scoring=custom_scorer,
                   verbose=1)
clf.fit(features, labels)
print("Training ended ...")
print("Best params: ", clf.best_params_)

rf_clf = clf.best_estimator_
print("\n\nAccuracy on all training data set\n", sum(labels == rf_clf.predict(features)) / len(labels))


# In[ ]:


from sklearn.externals import joblib
joblib.dump(rf_clf, MODEL_NAME, compress = 1)


# In[ ]:


clf = joblib.load(MODEL_NAME)
for n_file in range(N_FILES):
    features, labels = get_features_labels(n_file, direc=PROCESSED_GAMES_DIR)
    print("Accuracy on " + str(n_file) + " training data set\n", sum(labels == clf.predict(features)) / len(labels))


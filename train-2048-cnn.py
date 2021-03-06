
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
N_VALIDATION_FILES = N_FILES % 50
TRAIN_MODEL = True

def load_data(file, direc=GAMES_DIR, header=True):
    csv_path = os.path.join(direc, file)
    if header:
        return pd.read_csv(csv_path)
    else:
        return pd.read_csv(csv_path, header=None)


# In[2]:


# CNN Classifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

batch_size = 256
epochs = 20

size = N_SIZE
num_classes = len(MOVES)
droprate = 0.7
N_MODELS = N_FILES

def create_model(index, show_summary=False):
    model_name = MODEL_NAME#str(index) + "_" + MODEL_NAME
    
    try:
        model = load_model(model_name)
    except:
        model = None

    if model is None:
        activation_fn = 'elu'
        n_feature_maps = 128

        model = Sequential()
        model.add(Conv2D(8 * n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, input_shape=(N_SIZE, N_SIZE, 1)))
        model.add(BatchNormalization())

        model.add(Conv2D(8 * n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(8 * n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(4 * n_feature_maps, kernel_size=(2, 2), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(4 * n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(4* n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(4 * n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(2 * n_feature_maps, kernel_size=(2, 2), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(2 * n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(2 * n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(2 * n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(n_feature_maps, kernel_size=(2, 2), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(n_feature_maps, kernel_size=(1, 1), strides=(1, 1), activation=activation_fn, padding='valid'))
        model.add(BatchNormalization())

        model.add(Dropout(droprate))
        model.add(Flatten())

        model.add(Dense(512, activation=activation_fn))
        model.add(BatchNormalization())
        model.add(Dropout(droprate))

        model.add(Dense(256, activation=activation_fn))
        model.add(BatchNormalization())
        model.add(Dropout(droprate))

        model.add(Dense(128, activation=activation_fn))
        model.add(BatchNormalization())
        model.add(Dropout(droprate))

        model.add(Dense(64, activation=activation_fn))
        model.add(BatchNormalization())
        model.add(Dropout(droprate))

        model.add(Dense(num_classes, activation='softmax'))

    else:
        print(model_name, " is restored.")

    if show_summary:
        model.summary()
        
    adam = Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    
    return model_name, model


# In[3]:


def get_features_labels(n_file, direc, group_n_games = N_FILES, validation=False):
    x = []
    y = []
    
    #if not validation:
        #group_n_games = 1
    
    for indx in range(group_n_games):
        
        filename = GAME_STATE_FILE_NAME + str(n_file % N_FILES) + GAME_STATE_FILE_EXT
        n_file = n_file - 1
        
        if not validation:
            print("Training on " + filename)
        else:
            print("Validating on " + filename)
            
        data = load_data(file=filename, direc=direc)
    
        labels = data[MOVE_COL_NAME].values
        data.drop(MOVE_COL_NAME, axis=1, inplace=True)
        binarizer = LabelBinarizer()
        binarizer.fit([0, 1, 2, 3])
        labels = binarizer.transform(labels)

        features = data.values
        features = np.reshape(features, (-1, N_SIZE, N_SIZE, 1))
        
        if len(x) == 0:
            x = features
            y = labels
        else:
            x = np.concatenate((x, features), axis=0)
            y = np.concatenate((y, labels), axis=0)
                                              
    return x, y


# In[4]:

val_features, val_labels = get_features_labels(N_FILES-1, group_n_games=N_VALIDATION_FILES, direc=PROCESSED_GAMES_DIR, validation=True)
print("Length  of val data: " + str(len(val_labels)))
n_file = N_FILES - N_VALIDATION_FILES - 1

model_name, model = create_model(index=(0 % N_MODELS))
callbacks = [ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)]
jumps = 10
while (n_file - jumps) >= 0:
    #features, labels = get_features_labels(n_file, group_n_games=3, direc=PROCESSED_GAMES_DIR, validation=True)
    features, labels = get_features_labels(n_file, group_n_games=jumps, direc=PROCESSED_GAMES_DIR)
    #model_name, model = create_model(index=(n_file % N_MODELS))
    #callbacks = [ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)]
    #print("\n\n\n\n\nSTARTED WITH GAME #" + str(n_file))
    
    if TRAIN_MODEL:
        history = model.fit(features, labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(val_features, val_labels),
                        callbacks=callbacks)
    else:
        print("Opted not to train the model as TRAIN_MODEL is set to False. May be because model is already trained and is now being used for validation")
    
    n_file = n_file - jumps

    score = model.evaluate(val_features, val_labels, verbose=0)
    print('Model Test loss:', score[0])
    print('Model Test accuracy:', score[1])


saved_model = load_model(MODEL_NAME)
score = saved_model.evaluate(val_features, val_labels, verbose=0)
print('Saved Model Test loss:', score[0])
print('Saved Model Test accuracy:', score[1])


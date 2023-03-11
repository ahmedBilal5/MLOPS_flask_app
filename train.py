#!/usr/bin/python3
# tain.py

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
#from sklearn.metrics import confusion_matrix
from joblib import dump, load

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump
from sklearn import preprocessing

print("this is the train file")
def train():

    # Load directory paths for persisting model

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_FILE_RFC = os.environ["MODEL_FILE_RFC"]
    MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
    MODEL_PATH_RFC = os.path.join(MODEL_DIR, MODEL_FILE_RFC)
    print(MODEL_PATH_RFC)
    # Load, read and normalize training data
    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')
    
    # Models training
    
    # Linear Discrimant Analysis (Default parameters)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)
    
    # Save model
    dump(clf_lda, MODEL_PATH_LDA)
    print("Training LDA score and classification:")
    print(clf_lda.score(X_train, y_train))
    predicted = clf_lda.predict(X_train)
    print(predicted)
    #print(confusion_matrix(y_train, predicted))
    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=1, max_iter=1000)
    clf_NN.fit(X_train, y_train)
    dump(clf_NN, MODEL_PATH_NN)
    print("Training NN score and classification:")
    print(clf_NN.score(X_train, y_train))
    predicted = clf_NN.predict(X_train)
    print(predicted)
    #random forest
    from sklearn.ensemble import RandomForestClassifier
    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train, y_train)
   # print(confusion_matrix(y_train, predicted))
    dump(clf_rfc, MODEL_PATH_RFC)
    print("Training RFC score and classification:")
    print(clf_rfc.score(X_train, y_train))
    predicted = clf_rfc.predict(X_train)
    print(predicted)

if __name__ == '__main__':
    train()

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 00:55:05 2021

@author: PC
"""
from data import load_data
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



def feature_extractor(images, labels):
    pre_train_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    pre_train_model.summary()
    features = pre_train_model.predict(images)
    features = features.reshape((features.shape[0], 7*7*512))
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_test, y_test)

def build_model(X_train, y_train):
    params = {'C' : [0.1, 1.0, 10.0, 100.0]}
    model = GridSearchCV(LogisticRegression(), params)
    model.fit(X_train, y_train)
    print('Best parameter for the model {}'.format(model.best_params_))
    return model
    
if __name__ == "__main__":
    images, labels = load_data()
    (X_train, y_train), (X_test, y_test) = feature_extractor(images, labels)
    model = build_model(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
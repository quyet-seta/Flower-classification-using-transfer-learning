# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 02:12:34 2021

@author: PC
"""
from data import load_data
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import CSVLogger


def build_model(shape=(224,224,3)):
    
    inputs = Input(shape=shape)    
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    #vgg16.summary()
    
    out_vgg16 = vgg16.output
    x = Flatten()(out_vgg16)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(17, activation='softmax')(x)
    return (vgg16, Model(inputs, outputs))

def train(pre_train, model):
    images, labels = load_data()
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    aug_test= ImageDataGenerator(rescale=1./255)
    
    callbacks_1 = [
        CSVLogger("flower_loss_acc1.csv")
    ]
    
    callbacks_2 = [
        CSVLogger("flower_loss_acc2.csv")
    ]
    
    # freeze VGG model
    for layer in pre_train.layers:
        layer.trainable = False
        
    opt = RMSprop(0.001)
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    numOfEpoch = 25
    H = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32), 
                            steps_per_epoch=len(X_train)//32,
                            validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),
                            validation_steps=len(X_test)//32,
                            epochs=numOfEpoch,
                            callbacks=callbacks_1)
    
    # unfreeze some last CNN layer:
    for layer in pre_train.layers[15:]:
        layer.trainable = True
    
    numOfEpoch = 35
    opt = SGD(0.001)
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    H = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32), 
                            steps_per_epoch=len(X_train)//32,
                            validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),
                            validation_steps=len(X_test)//32,
                            epochs=numOfEpoch,
                            callbacks=callbacks_2)
    
if __name__ == "__main__":
    vgg16, model = build_model()
    model.summary()
    train(vgg16, model)
    
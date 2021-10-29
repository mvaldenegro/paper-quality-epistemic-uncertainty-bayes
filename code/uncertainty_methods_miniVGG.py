import keras
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, AveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2

import keras_uncertainty
from keras_uncertainty.models import MCDropoutClassifier, DeepEnsembleClassifier, DeepSubEnsembleClassifier, StochasticClassifier, GradientClassificationConfidence
from keras_uncertainty.layers import DropConnectConv2D, DropConnectDense, RBFClassifier, add_gradient_penalty, add_l2_regularization, duq_training_loop
from keras_uncertainty.layers import BayesByBackpropDense, FlipoutDense

def minivgg_model(input_shape, num_classes=10):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation="softmax"))    

    return model

def minivgg_deepensemble(input_shape, dataset, num_epochs, batch_size, prob=0.25, num_classes=10, num_ensembles=5):
    def model_fn():
        model = minivgg_model(input_shape=input_shape, num_classes=num_classes)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model
    
    model = DeepEnsembleClassifier(model_fn=model_fn, num_estimators=num_ensembles)

    x_train, y_train = dataset
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)

    return model

def minivgg_dropout(input_shape, dataset, num_epochs, batch_size, prob=0.25, num_classes=10):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(prob))
    model.add(Dense(num_classes, activation="softmax")) 
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    x_train, y_train = dataset

    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
    mc_model = MCDropoutClassifier(model)

    return mc_model

def minivgg_dropconnect(input_shape, dataset, num_epochs, batch_size, prob=0.1, num_classes=10):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(DropConnectDense(num_classes, activation="softmax", prob=prob)) 
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    x_train, y_train = dataset

    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
    mc_model = MCDropoutClassifier(model)

    return mc_model

from keras.optimizers import SGD

def minivgg_duq(input_shape, dataset, num_epochs, batch_size, num_classes=10):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(RBFClassifier(num_classes, 0.1, gamma=0.999, centroid_dims=256))

    add_l2_regularization(model)
    add_gradient_penalty(model, lambda_coeff=0.05)

    sgd = SGD(learning_rate=0.05, momentum=0.9, decay=1e-4)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["categorical_accuracy"])
    
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    x_train, y_train = dataset

    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)

    return model

def minivgg_baseline(input_shape, dataset, num_epochs, batch_size, num_classes=10):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation="softmax"))    

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    x_train, y_train = dataset

    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    return model

def minivgg_flipout(input_shape, dataset, num_epochs, batch_size, num_classes=10):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(FlipoutDense(num_classes, kl_weight=0.0, activation="softmax", prior=False, bias_distribution=False))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    x_train, y_train = dataset
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=2 * batch_size, verbose=1)

    mc_model = StochasticClassifier(model, num_samples=50)

    return mc_model
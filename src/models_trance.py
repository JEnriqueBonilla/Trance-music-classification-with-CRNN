import os
import pickle
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping


def regularized_validation_logistic_regression(X_train, Y_train, X_val, Y_val, inverse_regularization_strength, n_classes):
    """

    :param X_train:
    :param Y_train:
    :param X_val:
    :param Y_val:
    :param inverse_regularization_strength:
    :return:
    """

    best_performance = {'best_accuracy': 0,
                        'inverse_constant': 0}
    
    if n_classes==4:
        solver = 'newton-cg'
        max_iter=100
    else:
        solver = 'lbfgs'
        max_iter=100

    for c in inverse_regularization_strength:
        model = LogisticRegression(multi_class='multinomial', solver=solver, penalty='l2', C=c, max_iter=max_iter)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_val)
        accuracy = accuracy_score(Y_val, Y_pred)

        if accuracy > best_performance['best_accuracy']:
            best_performance['best_accuracy'] = accuracy
            best_performance['inverse_constant'] = c

    logistic_model = LogisticRegression(multi_class='multinomial', solver=solver, penalty='l2', C=best_performance['inverse_constant'], max_iter=max_iter)

    logistic_model.fit(X_train, Y_train)

    return best_performance, logistic_model


def mfcc_ffnn(X_train, Y_train, X_val, Y_val, n_classes, save_weights_folder='mfcc_weights', random_state=42,
              verbose=True, early_stop=20, batch_size=16, epochs=200, train=True, load_checkpoint=False):
    """

    :param X_train:
    :param Y_train:
    :param X_val:
    :param Y_val:
    :param n_classes:
    :param save_weights_folder:
    :param random_state:
    :param verbose:
    :param early_stop:
    :param batch_size:
    :param epochs:
    :param train:
    :param load_checkpoint:
    :return:
    """
    warnings.filterwarnings('ignore')

    weights = os.path.join(save_weights_folder, str(n_classes) + '_' + str(random_state))
    os.makedirs(save_weights_folder, exist_ok=True)

    # transform data
    x_train = np.array([x.tolist() for x in X_train])
    y_train = Y_train
    x_val = np.array([x.tolist() for x in X_val])
    y_val = Y_val

    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_val = np_utils.to_categorical(lb.fit_transform(y_val))

    num_labels = y_train.shape[1]

    # build model
    model = Sequential()

    model.add(Dense(256, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.summary()

    # Initialize weights using checkpoint if it exists
    if load_checkpoint:
        if os.path.isfile(weights):
            print('Checkpoint file detected. Loading weights.')
            model.load_weights(weights)
        else:
            train = True

    checkpointer = ModelCheckpoint(filepath=weights, verbose=verbose, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop, verbose=verbose, mode='auto')

    # Train model
    if train:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                  callbacks=[checkpointer, earlystopper])

        model.load_weights(weights)

    return model


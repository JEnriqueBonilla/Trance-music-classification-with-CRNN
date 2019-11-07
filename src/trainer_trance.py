import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from .models import CRNN2D
from .utility import plot_history, predict_artist, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping
from .utility_trance import encode_labels, slice_songs,load_mel_dataset_previous_split


def train_model(train_songs, val_songs, test_songs, slice_length, song_folder, n_classes, plots=True, train=True,
                load_checkpoint=False, save_metrics=True, save_metrics_folder='metrics_plots_mel',
                save_weights_folder='weights_mel', batch_size=16, nb_epochs=200, early_stop=10, lr=0.0001,
                random_state=42):
    """
    Main function for training the model and testing

    :param train_songs:
    :param val_songs:
    :param test_songs:
    :param slice_length:
    :param song_folder:
    :param n_classes:
    :param plots:
    :param train:
    :param load_checkpoint:
    :param save_metrics:
    :param save_metrics_folder:
    :param save_weights_folder:
    :param batch_size:
    :param nb_epochs:
    :param early_stop:
    :param lr:
    :param random_state:
    :return:
    """
    warnings.filterwarnings('ignore')

    print("Training for slice length of {} \n".format(slice_length))
    weights = os.path.join(save_weights_folder, str(n_classes) +
                           '_' + str(slice_length) + '_' + str(random_state))
    os.makedirs(save_weights_folder, exist_ok=True)
    os.makedirs(save_metrics_folder, exist_ok=True)

    # song split
    Y_train, X_train, S_train, Y_test, X_test, S_test, Y_val, X_val, S_val = load_mel_dataset_previous_split(train_songs,
                                                                                                             val_songs,
                                                                                                             test_songs,
                                                                                                             song_folder_name=song_folder)

    # Create slices out of the songs
    X_train, Y_train, S_train = slice_songs(X_train, Y_train, S_train, length=slice_length)
    X_val, Y_val, S_val = slice_songs(X_val, Y_val, S_val, length=slice_length)
    X_test, Y_test, S_test = slice_songs(X_test, Y_test, S_test, length=slice_length)

    print('Training set label counts: {} \n'.format(np.unique(Y_train, return_counts=True)))

    # Encode the target vectors into one-hot encoded vectors
    Y_train, le, enc = encode_labels(Y_train)
    Y_test, le, enc = encode_labels(Y_test, le, enc)
    Y_val, le, enc = encode_labels(Y_val, le, enc)

    # Reshape data as 2d convolutional tensor shape
    X_train = X_train.reshape(X_train.shape + (1,))
    X_val = X_val.reshape(X_val.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    # build the model
    model = CRNN2D(X_train.shape, nb_classes=Y_train.shape[1])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    print('model summary')
    model.summary()
    print('')

    # Initialize weights using checkpoint if it exists
    if load_checkpoint:
        print("Looking for previous weights...")
        if os.path.isfile(weights):
            print('Checkpoint file detected. Loading weights.')
            model.load_weights(weights)
        else:
            print('No checkpoint file detected.  Starting from scratch.')
    else:
        print('Starting from scratch (no checkpoint)')

    checkpointer = ModelCheckpoint(filepath=weights, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop, verbose=0, mode='auto')

    # Train the model
    if train:
        print("Input Data Shape {} \n".format(X_train.shape))
        history = model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, epochs=nb_epochs, verbose=1,
                            validation_data=(X_val, Y_val), callbacks=[checkpointer, earlystopper])
        if plots:
            plot_history(history)

    # Load weights that gave best performance on validation set
    model.load_weights(weights)
    filename = os.path.join(save_metrics_folder, str(n_classes) + '_' + str(slice_length) + '_' + str(random_state) + '.txt')

    # Score test model
    score = model.evaluate(X_test, Y_test, verbose=0)
    y_score = model.predict_proba(X_test)

    # Calculate confusion matrix
    y_predict = np.argmax(y_score, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(y_true, y_predict)

    # Plot the confusion matrix
    class_names = np.arange(n_classes)
    class_names_original = le.inverse_transform(class_names)
    plt.figure(figsize=(14, 14))
    plot_confusion_matrix(cm, classes=class_names_original, normalize=True, title='Confusion matrix with normalization')
    if save_metrics:
        plt.savefig(filename + '.png', bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(14, 14))

    # print out metrics
    print('Test score/loss:', score[0])
    print('Test accuracy:', score[1])
    print('\nTest results on each slice:')
    scores = classification_report(y_true, y_predict, target_names=class_names_original)
    scores_dict = classification_report(y_true, y_predict, target_names=class_names_original, output_dict=True)
    print(scores)

    # Predict artist using pooling methodology
    pooling_scores, pooled_scores_dict = predict_artist(model, X_test, Y_test, S_test, le,
                                                        class_names=class_names_original, slices=None, verbose=False)

    print('')
    print('')
    print('')
    # Save metrics
    if save_metrics:
        plt.savefig(filename + '_pooled.png', bbox_inches="tight")
        plt.close()
        with open(filename, 'w') as f:
            f.write("Training data shape:" + str(X_train.shape))
            f.write('\nnb_classes: ' + str(n_classes) +
                    '\nslice_length: ' + str(slice_length))
            f.write('\nweights: ' + weights)
            f.write('\nlr: ' + str(lr))
            f.write('\nTest score/loss: ' + str(score[0]))
            f.write('\nTest accuracy: ' + str(score[1]))
            f.write('\nTest results on each slice:\n')
            f.write(str(scores))
            f.write('\n\n Scores when pooling song slices:\n')
            f.write(str(pooling_scores))

    return score, scores, scores_dict, pooling_scores, pooled_scores_dict


def train_slice_lengths_crnn(train_songs, val_songs, test_songs, slice_lengths, song_folder, n_classes, train=True,
                             load_check_point=True, save_metrics_folder='metrics_plots_mel',
                             save_weights_folder='weights_mel', batch_size=16, nb_epochs=200, early_stop=10, lr=0.0001,
                             random_state=42, summary_metrics_output_folder='summary_mel'):
    """

    :param train_songs:
    :param val_songs:
    :param test_songs:
    :param slice_lengths:
    :param song_folder:
    :param n_classes:
    :param train:
    :param load_check_point:
    :param save_metrics_folder:
    :param save_weights_folder:
    :param batch_size:
    :param nb_epochs:
    :param early_stop:
    :param lr:
    :param random_state:
    :param summary_metrics_output_folder:
    :return:
    """
    warnings.filterwarnings('ignore')

    best_slice_len = None
    best_accuracy = 0

    for slice_len in slice_lengths:

        score, scores, scores_dict, pooling_scores, pooled_scores_dict = train_model(train_songs=train_songs,
                                                                                     val_songs=val_songs,
                                                                                     test_songs=test_songs,
                                                                                     slice_length=slice_len,
                                                                                     song_folder=song_folder,
                                                                                     n_classes=n_classes, plots=True,
                                                                                     train=train,
                                                                                     load_checkpoint=load_check_point,
                                                                                     save_metrics=True,
                                                                                     save_metrics_folder=save_metrics_folder,
                                                                                     save_weights_folder=save_weights_folder,
                                                                                     batch_size=batch_size,
                                                                                     nb_epochs=nb_epochs,
                                                                                     early_stop=early_stop,
                                                                                     lr=lr,
                                                                                     random_state=random_state)

        if score[1] > best_accuracy:
            best_slice_len = slice_len
            best_accuracy = score[1]

        gc.collect()

        os.makedirs(summary_metrics_output_folder, exist_ok=True)

        score_df = pd.DataFrame(list(scores_dict['weighted avg']))
        score_df.to_csv('{}/{}_score.csv'.format(summary_metrics_output_folder, slice_len))

        pooling_df = pd.DataFrame(list(pooled_scores_dict['weighted avg']))
        pooling_df.to_csv('{}/{}_pooled_score.csv'.format(summary_metrics_output_folder, slice_len))

    return best_accuracy, best_slice_len

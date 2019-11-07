import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .models import CRNN2D
from keras.optimizers import Adam
from sklearn.manifold import TSNE
from .utility_trance import load_mel_dataset, slice_songs, encode_labels


def plot_mel_tsne(random_state=42, slice_length=2584, nb_classes=4, weights_path='weights_mel',
                  folder='song_mel_label_data', ensemble_visual=False, save_path='tSNE_mel', trance_df=None):
    # set these parameters

    # checkpoint
    checkpoint_path = weights_path + '/' + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_state)

    # Load the song data and split into train and test sets at song level
    # Aqui tal vez valga la pena dividir en train_test_val
    print("Loading data for {}".format(slice_length))
    Y, X, S = load_mel_dataset(song_folder_name=folder)
    X, Y, S = slice_songs(X, Y, S, length=slice_length)

    # Reshape data as 2d convolutional tensor shape
    X_shape = X.shape + (1,)
    X = X.reshape(X_shape)

    # encode Y
    Y_original = Y
    Y, le, enc = encode_labels(Y)

    # build the model
    model = CRNN2D(X.shape, nb_classes=Y.shape[1])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    # Initialize weights using checkpoint if it exists
    if os.path.isfile(checkpoint_path):
        print('Checkpoint file detected. Loading weights. \n')
        model.load_weights(checkpoint_path)
    else:
        raise Exception('no checkpoint for {} \n'.format(checkpoint_path))

    # drop final dense layer and activation
    print("Modifying model and predicting representation")
    model.pop()
    model.pop()
    model.summary()

    # predict representation
    print("\n Predicting \n")
    X_rep = model.predict(X)

    del X
    gc.collect()

    if ensemble_visual:

        if trance_df is None:
            raise Exception('No Trance_df, all ensemble plots cannot be done without trance_df')

        songs = np.unique(S)
        X_song = np.zeros((songs.shape[0], X_rep.shape[1]))
        Y_song = np.empty((songs.shape[0]), dtype="S10")
        BPM = np.empty((songs.shape[0]), dtype="S10")
        MM = np.empty((songs.shape[0]), dtype="S10")

        for i, song in enumerate(songs):
            # Label
            xs = X_rep[S == song]
            Y_song[i] = Y_original[S == song][0]
            X_song[i, :] = np.mean(xs, axis=0)

            # BPM
            bpm = int(list(trance_df[trance_df['Song_file'] == song]["BPM"])[0])
            if bpm >= 137:
                result = '137+'
            elif bpm <= 130:
                result = '130-'
            else:
                result = '131-136'

            BPM[i] = result

            # Min or major
            MM[i] = list(trance_df[trance_df['Song_file'] == song]["Min_or_Maj"])[0]

        X_rep = X_song
        Y_original = Y_song

        # fit tsne
        print("Fitting TSNE {} \n".format(X_rep.shape))
        tsne_model = TSNE()
        X_2d = tsne_model.fit_transform(X_rep)

        # save results
        print("Saving results")
        os.makedirs(save_path, exist_ok=True)
        save_path += '/' + str(checkpoint_path.split('_')[2])
        save_path += '_ensemble'

        pd.DataFrame({'x0': X_2d[:, 0], 'x1': X_2d[:, 1], 'label': Y_original, 'bpm': BPM, 'min_maj': MM}).to_csv(save_path + '.csv', index=False)

        # save label figure
        sns.set_palette("Paired", n_colors=nb_classes)
        plt.figure(figsize=(10, 10))
        # plt.figure(figsize=(nb_classes, nb_classes))
        sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=Y_original, palette=sns.color_palette(n_colors=nb_classes))
        plt.savefig(save_path + '_label' + '.png')
        plt.close()

        # save bpm figure
        sns.set_palette("Paired", n_colors=len(set(BPM)))
        plt.figure(figsize=(10, 10))
        # plt.figure(figsize=(nb_classes, nb_classes))
        sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=BPM, palette=sns.color_palette(n_colors=len(set(BPM))))
        plt.savefig(save_path + '_bpm' + '.png')
        plt.close()

        # save min or major figure
        sns.set_palette("Paired", n_colors=len(set(MM)))
        plt.figure(figsize=(10, 10))
        # plt.figure(figsize=(nb_classes, nb_classes))
        sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=MM, palette=sns.color_palette(n_colors=len(set(MM))))
        plt.savefig(save_path + '_mm' + '.png')
        plt.close()

    else:
        # fit tsne
        print("Fitting TSNE {} \n".format(X_rep.shape))
        tsne_model = TSNE()
        X_2d = tsne_model.fit_transform(X_rep)

        # save results
        print("Saving results")
        os.makedirs(save_path, exist_ok=True)
        save_path += '/' + str(checkpoint_path.split('_')[2])

        pd.DataFrame({'x0': X_2d[:, 0], 'x1': X_2d[:, 1], 'label': Y_original}).to_csv(save_path + '.csv', index=False)

        # save figure
        sns.set_palette("Paired", n_colors=nb_classes)
        plt.figure(figsize=(14, 14))
        # plt.figure(figsize=(nb_classes, nb_classes))
        sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=Y_original, palette=sns.color_palette(n_colors=nb_classes))
        plt.savefig(save_path + '_label' + '.png')
        plt.close()

    del Y, S, X_rep, X_2d, Y_original

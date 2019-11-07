import os
import dill
import pickle
import random
import librosa
import warnings
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
from mutagen.mp3 import MP3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .utility import plot_confusion_matrix


def load_dataset_song_split(X, Y, S, test_split_size=0.1, validation_split_size=0.1, random_state=42):
    """

    :param X:
    :param Y:
    :param S:
    :param test_split_size:
    :param validation_split_size:
    :param random_state:
    :return:
    """
    # train and test split
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, test_size=test_split_size, stratify=Y,
                                                                         random_state=random_state)

    # Create a validation to be used to track progress  # quisa quite una para dejar mi test aparte
    X_train, X_val, Y_train, Y_val, S_train, S_val = train_test_split(X_train, Y_train, S_train,
                                                                      test_size=validation_split_size, shuffle=True,
                                                                      stratify=Y_train, random_state=random_state)

    return Y_train, X_train, S_train, Y_test, X_test, S_test, Y_val, X_val, S_val


def get_songs_split(s_train, s_val, s_test):
    """

    :param s_train:
    :param s_val:
    :param s_test:
    :return:
    """

    train_songs = [val[0] for val in s_train.values]
    val_songs = [val[0] for val in s_val.values]
    test_songs = [val[0] for val in s_test.values]

    return train_songs, val_songs, test_songs


def logistic_test_performance(X_test, Y_test, n_labels, logistic_model, save_metrics=True, plot_folder='./metrics_logistic'):
    """

    :param X_test:
    :param Y_test:
    :param logistic_model:
    :param save_metrics:
    :param plot_folder:
    :return:
    """
    warnings.filterwarnings('ignore')
    y_predict = logistic_model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, y_predict)

    # Calculate confusion matrix
    cm = confusion_matrix(Y_test, y_predict)

    # Plot the confusion matrix
    class_names_original = logistic_model.classes_
    plt.figure(figsize=(14, 14))
    plot_confusion_matrix(cm, classes=class_names_original, normalize=True, title='Confusion matrix with normalization')

    if save_metrics:
        plt.savefig(plot_folder + '/logistic_' + str(n_labels) + '.png', bbox_inches="tight")

    plt.close()
    plt.figure(figsize=(14, 14))

    # TODO: print('Test score/loss:', score[0])

    scores = classification_report(Y_test, y_predict, target_names=class_names_original)
    scores_dict = classification_report(Y_test, y_predict, target_names=class_names_original, output_dict=True)

    return test_accuracy, scores, scores_dict


def create_mfcc_dataset(label_folder='./electronic_music/Trance_label/Train/', song_duration=180.0,
                        save_folder='song_mfccs_label_data', sr=44100, create_dataset=True):
    """

    :param label_folder:
    :param song_duration:
    :param save_folder:
    :param create_dataset:
    :param sr:
    :return:
    """
    if create_dataset:
        # load labels
        os.makedirs(save_folder, exist_ok=True)
        labels = [path for path in os.listdir(label_folder) if os.path.isdir(label_folder + path)]

        # iterate through all labels
        for label in labels:
            print('{} \n '.format(label))
            label_path = os.path.join(label_folder, label)
            label_songs = os.listdir(label_path)

            # iterate through  label songs
            for song in label_songs:
                # load song_duration seconds of a song
                song_path = os.path.join(label_path, song)
                audio = MP3(song_path)
                audio_length = int(audio.info.length)
                audio_middle = (audio_length - int(song_duration))/2
                X, sample_rate = librosa.load(song_path, offset=audio_middle, duration=song_duration, sr=sr)

                # extract Mel-frequency cepstral coefficients (mfcc) feature from data
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
                data = (label, mfccs, song)

                # Save each song
                save_name = label + '_%%-%%_' + song
                with open(os.path.join(save_folder, save_name), 'wb') as fp:
                    dill.dump(data, fp)


def load_dataset(song_folder_name):
    """
    This function loads the dataset based on a location; it returns a list of spectrograms and their corresponding
    artists/song names

    :param song_folder_name:
    :return:
    """

    # Get all songs saved as numpy arrays in the given folder
    song_list = os.listdir(song_folder_name)

    # Create empty lists
    label = []
    mfccs = []
    song_name = []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)

        label.append(loaded_song[0])
        mfccs.append(loaded_song[1])
        song_name.append(loaded_song[2])

    return label, mfccs, song_name


def load_dataset_previous_split(train_songs, val_songs, test_songs, song_folder_name):
    """

    :param train_songs:
    :param val_songs:
    :param test_songs:
    :param song_folder_name:
    :return:
    """
    Y_train = []
    X_train = []
    S_train = []
    Y_test = []
    X_test = []
    S_test = []
    Y_val = []
    X_val = []
    S_val = []

    Y, X, S = load_dataset(song_folder_name=song_folder_name)

    for x, y, song in zip(X, Y, S):
        if song in train_songs:
            Y_train.append(y)
            X_train.append(x)
            S_train.append(song)
        elif song in val_songs:
            Y_val.append(y)
            X_val.append(x)
            S_val.append(song)
        elif song in test_songs:
            Y_test.append(y)
            X_test.append(x)
            S_test.append(song)

    return Y_train, X_train, S_train, Y_test, X_test, S_test, Y_val, X_val, S_val


def mfcc_test_performance(X_test, Y_test, model, n_classes, save_metrics=True, plot_folder='./metrics_mfcc'):
    """

    :param X_test:
    :param Y_test:
    :param model:
    :param n_classes:
    :param save_metrics:
    :param plot_folder:
    :return:
    """
    lb = LabelEncoder()

    x_test = np.array([x.tolist() for x in X_test])
    y_test = Y_test
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    y_score = model.predict_proba(x_test)

    # Calculate confusion matrix
    y_predict = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_predict)

    # Plot the confusion matrix
    class_names = np.arange(n_classes)
    class_names_original = lb.inverse_transform(class_names)
    plt.figure(figsize=(14, 14))
    plot_confusion_matrix(cm, classes=class_names_original, normalize=True, title='Confusion matrix with normalization')

    if save_metrics:
        plt.savefig(plot_folder + '/mfcc_' + str(n_classes) + '.png', bbox_inches="tight")

    plt.close()
    plt.figure(figsize=(14, 14))

    # save scores
    scores = classification_report(y_true, y_predict, target_names=class_names_original)
    scores_dict = classification_report(y_true, y_predict, target_names=class_names_original, output_dict=True)

    return score, scores, scores_dict


def create_melspectrogram_dataset(label_folder='electronic_music/Trance_label/Train/', save_folder='song_mel_label_data',
                                  sr=44100, n_mels=128, n_fft=2048, hop_length=512, song_duration=180.0,
                                  create_data=False):
    """
    This function creates the dataset given a folder with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder.

    :param label_folder:
    :param save_folder:
    :param sr:
    :param n_mels:
    :param n_fft:
    :param hop_length:
    :param song_duration:
    :param create_data:
    :return:
    """
    if create_data:
        # get list of all labels
        os.makedirs(save_folder, exist_ok=True)
        labels = [path for path in os.listdir(label_folder) if os.path.isdir(label_folder + path)]

        # iterate through all lables, songs and find mel spectrogram
        for label in labels:
            print('{} \n'.format(label))
            label_path = os.path.join(label_folder, label)
            label_songs = os.listdir(label_path)

            for song in label_songs:
                print(song)
                song_path = os.path.join(label_path, song)

                # Create mel spectrogram for song_duration in the middle of the song and convert it to the log scale
                audio = MP3(song_path)
                audio_lenght = int(audio.info.length)
                audio_middle = (audio_lenght - int(song_duration))/2
                y, sr = librosa.load(song_path, sr=sr, offset=audio_middle, duration=song_duration)
                S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                log_S = librosa.logamplitude(S, ref_power=1.0)
                data = (label, log_S, song)

                # Save each song
                save_name = label + '_%%-%%_' + song
                with open(os.path.join(save_folder, save_name), 'wb') as fp:
                    dill.dump(data, fp)


def visualize_spectrogram(path, duration=None, offset=0, sr=44100, n_mels=128, n_fft=2048, hop_length=512):
    """
    This function creates a visualization of a spectrogram given the path to an audio file

    :param path:
    :param duration:
    :param offset:
    :param sr:
    :param n_mels:
    :param n_fft:
    :param hop_length:
    :return:
    """

    # Make a mel-scaled power (energy-squared) spectrogram
    y, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    # Convert to log scale (dB)
    log_S = librosa.logamplitude(S, ref_power=1.0)

    # song name
    _, _, _, _, song_name = path.split("/")

    # Render output spectrogram in the console
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram for ' + song_name)
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


def create_spectrogram_plots(label_folder='electronic_music/Trance_label/Train/', sr=44100, n_mels=128, n_fft=2048,
                             hop_length=512, song_duration=180.0, n_classes=4):
    """
    Create a spectrogram from a randomly selected song for each artist and plot"

    :param label_folder:
    :param sr:
    :param n_mels:
    :param n_fft:
    :param hop_length:
    :param song_duration:
    :param n_classes:
    :return:
    """

    # get list of all artists
    labels = os.listdir(label_folder)
    
    fig, ax = plt.subplots(nrows=2, ncols=int(n_classes/2), figsize=(14, 12), sharex=True, sharey=True)

    row = 0
    col = 0

    # iterate through labels and random songs and plot a spectrogram on a grid
    for label in labels:
        # Randomly select album and song
        label_path = os.path.join(label_folder, label)
        label_songs = os.listdir(label_path)
        song = random.choice(label_songs)
        song_path = os.path.join(label_path, song)

        # Create mel spectrogram
        audio = MP3(song_path)
        audio_lenght = int(audio.info.length)
        audio_middle = (audio_lenght - int(song_duration)) / 2

        y, sr = librosa.load(song_path, sr=sr, offset=audio_middle, duration=5)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S, ref_power=1.0)

        # Plot on grid
        plt.axes(ax[row, col])
        librosa.display.specshow(log_S, sr=sr)
        plt.title(label)
        col += 1
        if col == int(n_classes/2):
            row += 1
            col = 0

    fig.tight_layout()


def create_mel_vizualizations(song_path, label_folder, n_classes ,plot_folder='spectogram_plot', create_visuals=True, save_visuals=True, song_duration=180.0):
    """

    :param song_path:
    :param label_folder:
    :param plot_folder:
    :param create_visuals:
    :param save_visuals:
    :param song_duration:
    :return:
    """

    _, _, _, _, song_name = song_path.split("/")
    song_name, _ = song_name.split('.')

    if create_visuals:
        # Create spectrogram for a specific song
        audio = MP3(song_path)
        audio_lenght = int(audio.info.length)
        audio_middle = (audio_lenght - int(song_duration)) / 2

        visualize_spectrogram(song_path, offset=audio_middle, duration=song_duration)

        if save_visuals:
            os.makedirs(plot_folder, exist_ok=True)
            plt.savefig(os.path.join(plot_folder + '/' + song_name + '.png'), bbox_inches="tight")

        # Create spectrogram plots from randomly selected songs for each label
        create_spectrogram_plots(label_folder=label_folder, sr=44100, n_mels=128, n_fft=2048,
                                 hop_length=512, song_duration=180.0, n_classes=n_classes)

        if save_visuals:
            plt.savefig(os.path.join(plot_folder + '/spectrograms_' + str(n_classes) + '.png'), bbox_inches="tight")


def load_mel_dataset(song_folder_name):
    """
    This function loads the dataset based on a location; it returns a list of spectrograms and their corresponding
    artists/song names

    :param song_folder_name:
    :return:
    """

    # Get all songs saved as numpy arrays in the given folder
    song_list = os.listdir(song_folder_name)

    # Create empty lists
    label = []
    spectrogram = []
    song_name = []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)

        label.append(loaded_song[0])
        spectrogram.append(loaded_song[1])
        song_name.append(loaded_song[2])

    return label, spectrogram, song_name


def load_mel_dataset_previous_split(train_songs, val_songs, test_songs, song_folder_name):
    """

    :param train_songs:
    :param val_songs:
    :param test_songs:
    :param song_folder_name:
    :return:
    """
    Y_train = []
    X_train = []
    S_train = []
    Y_test = []
    X_test = []
    S_test = []
    Y_val = []
    X_val = []
    S_val = []

    Y, X, S = load_mel_dataset(song_folder_name=song_folder_name)

    for x, y, song in zip(X, Y, S):
        if song in train_songs:
            Y_train.append(y)
            X_train.append(x)
            S_train.append(song)
        elif song in val_songs:
            Y_val.append(y)
            X_val.append(x)
            S_val.append(song)
        elif song in test_songs:
            Y_test.append(y)
            X_test.append(x)
            S_test.append(song)

    return Y_train, X_train, S_train, Y_test, X_test, S_test, Y_val, X_val, S_val


def load_mel_dataset_song_split(song_folder_name, test_split_size=0.1, validation_split_size=0.1, random_state=42):
    """

    :param song_folder_name:
    :param test_split_size:
    :param validation_split_size:
    :param random_state:
    :return:
    """
    Y, X, S = load_dataset(song_folder_name=song_folder_name)
    # train and test split
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, test_size=test_split_size, stratify=Y,
                                                                         random_state=random_state)

    # Create a validation to be used to track progress  # quisa quite una para dejar mi test aparte
    X_train, X_val, Y_train, Y_val, S_train, S_val = train_test_split(X_train, Y_train, S_train,
                                                                      test_size=validation_split_size, shuffle=True,
                                                                      stratify=Y_train, random_state=random_state)

    return Y_train, X_train, S_train, Y_test, X_test, S_test, Y_val, X_val, S_val


def slice_songs(X, Y, S, length):
    """
    Slices the spectrogram into sub-spectrograms according to length

    :param X:
    :param Y:
    :param S:
    :param length:
    :return:
    """

    # Create empty lists for train and test sets
    label = []
    spectrogram = []
    song_name = []

    # Slice up songs using the length specified
    for i, song in enumerate(X):
        slices = int(song.shape[1] / length)
        for j in range(slices):
            spectrogram.append(song[:, length * j:length * (j + 1)])
            label.append(Y[i])
            song_name.append(S[i])

    return np.array(spectrogram), np.array(label), np.array(song_name)


def encode_labels(Y, le=None, enc=None):
    """
    Encodes target variables into numbers and then one hot encodings"

    :param Y:
    :param le:
    :param enc:
    :return:
    """

    # initialize encoders
    N = Y.shape[0]

    # Encode the labels
    if le is None:
        le = LabelEncoder()
        Y_le = le.fit_transform(Y).reshape(N, 1)
    else:
        Y_le = le.transform(Y).reshape(N, 1)

    # convert into one hot encoding
    if enc is None:
        enc = OneHotEncoder()
        Y_enc = enc.fit_transform(Y_le).toarray()
    else:
        Y_enc = enc.transform(Y_le).toarray()

    return Y_enc, le, enc


def create_trance_df(my_path):
    """

    :param my_path:
    :return:
    """
    trance_labels = [os.path.join(my_path, file) for file in os.listdir(my_path)]
    d = {'BPM': [], 'Genre': [], 'Label': [], 'Song_name': [], 'Key': [], 'Artists': [], 'Length': []}
    trance_df = pd.DataFrame(data=d)

    for label_path in trance_labels:
        trance_songs = [(os.path.join(label_path, file), file) for file in os.listdir(label_path) if
                        os.path.isfile(os.path.join(label_path, file))]

        for path_file, file in trance_songs:

            audio = MP3(path_file)
            audio_lenght = int(audio.info.length)
            
            key, bpm, artists, song = file.split('-')
            _, _, _, _, label = label_path.split('/')

            trance_df = trance_df.append({'BPM': bpm, 'Genre': 'Trance', 'Label': label, 'Song_name': song, 'Key': key,
                                          'Artists': artists, 'Length': audio_lenght, 'Song_file': file},
                                         ignore_index=True)

    trance_df["BPM"] = trance_df["BPM"].apply(int)
    trance_df["Length"] = trance_df["Length"].apply(float)
    trance_df["Length_minutes"] = trance_df["Length"] / 60
    trance_df["Min_or_Maj"] = ["Min" if "A" in key else "Maj" for key in trance_df["Key"]]

    return trance_df


def plot_label_frequencies(trance_df):
    """

    :param trance_df:
    :return:
    """
    data = trance_df.groupby(["Label"])["Label"].agg(["count"]).reset_index()

    plt.bar(data["Label"], data["count"])
    plt.xticks(data["Label"], rotation='vertical')
    plt.xlabel('Labels')
    plt.ylabel('No of tracks')
    plt.title("track frequency")

    for index, value in enumerate(data["count"]):
        plt.text(index + -.1, value + .03, value)

    plt.savefig('./exploratory_analysis/label_frequencies.png', bbox_inches="tight")
    plt.show()


def plot_keys(trance_df, sort_index):
    """

    :param trance_df:
    :param sort_index:
    :return:
    """
    keys = ["1A ", "2A ", "3A ", "4A ", "5A ", "6A ", "7A ", "8A ", "9A ", "10A ", "11A ", "12A ", "1B ", "2B ", "3B ",
            "4B ", "5B ", "6B ", "7B ", "8B ", "9B ", "10B ", "11B ", "12B "]

    df = trance_df.groupby(["Label"])["Key"].value_counts()

    plt.figure(figsize=(18, 18))
    for index, label in enumerate(trance_df["Label"].unique()):
        plt.subplot(321 + index)
        plt.title(label)
        df2 = df[label]
        df2 = df2.reindex(keys)
        df2 = df2/df2.sum()

        if sort_index:
            df2 = df2.sort_index()
        else:
            df2 = df2.sort_values(ascending=False)

        df2.plot(kind='bar')

    plt.savefig('./exploratory_analysis/group_keys.png', bbox_inches="tight")


def plot_min_major(trance_df):
    """

    :param trance_df:
    :return:
    """
    df = trance_df.groupby(["Label"])["Min_or_Maj"].value_counts()  # .reset_index() #.plot(kind='bar')

    plt.figure(figsize=(18, 18))
    for index, label in enumerate(trance_df["Label"].unique()):
        plt.subplot(321 + index)
        plt.title(label)
        df2 = df[label]
        df2 = df2/df2.sum()
        df2.plot(kind='bar')

    plt.savefig('./exploratory_analysis/group_min_major.png', bbox_inches="tight")


def print_logistic_coefficients(logistic_model, logistic_data):
    """

    :param logistic_model:
    :param logistic_data:
    :return:
    """
    coeficients = logistic_model.coef_

    for index, label in enumerate(logistic_model.classes_):
        print('{} \n'.format(label))

        label_coef = zip(logistic_data.columns, coeficients[index])
        label_coef = sorted(label_coef, key=lambda x: x[1], reverse=True)

        for zip_element in label_coef:
            print(zip_element)
        print('')
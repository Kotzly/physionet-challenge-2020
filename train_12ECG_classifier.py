#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from get_12ECG_features import get_12ECG_features
import tqdm
import multiprocessing as mp
from pathlib import Path
from itertools import chain
import re
import os
from os.path import join

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

N_JOBS = os.cpu_count() - 1

def create_nn(n_inputs=14, n_classes=111):
    neural_model = Sequential(
        [
            InputLayer(input_shape=(n_inputs,)),
            Dense(200, activation="relu"),
            Dropout(.2),
            Dense(n_classes, activation="sigmoid")
        ]
    )

    neural_model.compile(
        optimizer=Adam(lr=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return neural_model

def load_data(output_directory, header_files):
    if os.path.isfile(join(output_directory, "x_train.npy")):
        print("Found saved data. Loading...")
        x_train = np.load(join(output_directory, "x_train.npy"))
        y_train = np.load(join(output_directory, "y_train.npy"))
        header_files_train = np.load(join(output_directory, "header_files_train.npy"))

        x_val = np.load(join(output_directory, "x_val.npy"))
        y_val = np.load(join(output_directory, "y_val.npy"))
        header_files_val = np.load(join(output_directory, "header_files_val.npy"))

        x_test = np.load(join(output_directory, "x_test.npy"))
        y_test = np.load(join(output_directory, "y_test.npy"))
        header_files_test = np.load(join(output_directory, "header_files_test.npy"))

        classes = list(np.load(join(output_directory, "classes.npy")))
    else:
        print("Did not find saved data.")
        classes, features, labels = load_challenge_data(header_files)

        x_train, x_valtest, y_train, y_valtest, header_files_train, header_files_valtest = train_test_split(features, labels, header_files, test_size=0.5, random_state=1)
        x_val, x_test, y_val, y_test, header_files_val, header_files_test = train_test_split(x_valtest, y_valtest, header_files_valtest, test_size=0.5, random_state=1)

        del x_valtest, y_valtest, features, labels
        header_files_train = [str(Path(x).absolute()) for x in header_files_train]
        header_files_val = [str(Path(x).absolute()) for x in header_files_val]
        header_files_test = [str(Path(x).absolute()) for x in header_files_test]

        np.save(join(output_directory, "x_train.npy"), x_train)
        np.save(join(output_directory, "y_train.npy"), y_train)
        np.save(join(output_directory, "header_files_train.npy"), header_files_train)

        np.save(join(output_directory, "x_val.npy"), x_val)
        np.save(join(output_directory, "y_val.npy"), y_val)
        np.save(join(output_directory, "header_files_val.npy"), header_files_val)

        np.save(join(output_directory, "x_test.npy"), x_test)
        np.save(join(output_directory, "y_test.npy"), y_test)
        np.save(join(output_directory, "header_files_test.npy"), header_files_test)

        np.save(join(output_directory, "classes.npy"), classes)

    return x_train, y_train, x_val, y_val, classes

def train_12ECG_classifier(input_directory, output_directory, model="mlp"):
    # Load data.
    print('Listing files...')
    header_files = list(Path(input_directory).glob("*.hea"))

    print("Loading data...")

    x_train, y_train, x_val, y_val, classes = load_data(output_directory, header_files)


    # Train model.
    print('Training model...')
    # Replace NaN values with mean values
    imputer = SimpleImputer()
    scaler = RobustScaler()

    x_train = imputer.fit_transform(x_train)
    x_val = imputer.transform(x_val)

    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    
    # Train the classifier
    model = create_nn(
        n_inputs=x_train.shape[1],
        n_classes=y_train.shape[1]
    )
    model.summary()

    callbacks = EarlyStopping(
        monitor="val_loss",
        patience=30,
        min_delta=1e-3,
        restore_best_weights=True
    )
    history = model.fit(
        x_train, 
        y_train,
        callbacks=callbacks,
        validation_data=(x_val, y_val),
        epochs=1000,
        batch_size=32,
        workers=N_JOBS,
        use_multiprocessing=N_JOBS>1
    )

    # Save model.
    print('Saving model...')

    # final_model={'model': model, 'imputer': imputer,'classes': classes}
    # filename = os.path.join(output_directory, 'finalized_model.sav')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    artifacts = {
        'imputer': imputer,
        'classes': classes,
        'scaler': scaler
    }
    filepath = os.path.join(output_directory, 'artifacts.joblib')
    joblib.dump(artifacts, filepath, protocol=0)    
    joblib.dump(history.history, os.path.join(output_directory, 'history.joblib'))
    model.save(os.path.join(output_directory, "model"))

    # model_filename = os.path.join(output_directory, 'model.sav')
    # joblib.dump({'model': model}, model_filename, protocol=0)
    
    # imputer_filename = os.path.join(output_directory, 'imputer.sav')
    # joblib.dump({'imputer': imputer}, imputer_filename, protocol=0)

    # classes_filename = os.path.join(output_directory, 'classes.sav')
    # joblib.dump({'classes': classes}, classes_filename, protocol=0)
    
# Load challenge data.
def load_features_from_file(mat_file, header_file):

    with open(header_file, 'r') as f:
        header = f.readlines()

    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    feature = get_12ECG_features(recording, header)
    return feature

def load_features_from_files(mat_files, header_files):
    with mp.Pool(N_JOBS) as pool:
        features = pool.starmap(load_features_from_file, zip(mat_files, header_files))
    features = np.array(features)
    return features

def load_challenge_data(header_files):
    print("\tLoading labels")
    classes, labels = get_classes(header_files)

    mat_files = [str(header_file).replace('.hea', '.mat') for header_file in header_files]

    print("\tLoading features")
    features = load_features_from_files(mat_files, header_files)

    return classes, features, labels

# Find unique classes.
def get_classes_from_txt(txt):
    classes_txt = re.findall("#Dx:\s*([0-9a-zA-Z\,]*)\n", txt)[0]
    classes = list([c.strip() for c in classes_txt.split(",")])
    return classes

def get_classes_from_file(filename):
    with open(filename, 'r') as f:
        classes = get_classes_from_txt(f.read())
    return classes

def get_classes(filepaths, return_labels=True):
    with mp.Pool(N_JOBS) as pool:
        classes_list = pool.map(get_classes_from_file, filepaths)

    all_classes = sorted(set(list(chain(*classes_list))))

    labels = np.zeros((len(filepaths), len(all_classes)))

    for i, file_classes in enumerate(classes_list):
        for c in file_classes:
            labels[i, all_classes.index(c)] = 1

    if return_labels:
        return all_classes, labels
    return all_classes, None

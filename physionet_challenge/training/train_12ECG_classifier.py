#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from get_12ECG_features import get_12ECG_features
import tqdm
import multiprocessing as mp
from pathlib import Path
from itertools import chain
import re
import os
from os.path import join

from sklearn.model_selection import train_test_split

from physionet_challenge.model.baseline import BaselineMultibranch
from physionet_challenge.data.load import load_challenge_data, load_dataset
from physionet_challenge.data.save import save_dataset

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

N_JOBS = os.cpu_count()

def load_data(output_directory, header_files):
    if os.path.isfile(join(output_directory, "x_train.npy")):
        print("Found saved data. Loading...")
        x_train, y_train, header_files_train, \
        x_val, y_val, header_files_val, \
        x_test, y_test, header_files_test, classes = load_dataset(output_directory)
    else:
        print("Did not find saved data.")
        classes, features, labels = load_challenge_data(header_files)

        x_train, x_valtest, y_train, y_valtest, header_files_train, header_files_valtest = train_test_split(features, labels, header_files, test_size=0.5, random_state=1)
        x_val, x_test, y_val, y_test, header_files_val, header_files_test = train_test_split(x_valtest, y_valtest, header_files_valtest, test_size=0.5, random_state=1)

        del x_valtest, y_valtest, features, labels
        header_files_train = [str(Path(x).absolute()) for x in header_files_train]
        header_files_val = [str(Path(x).absolute()) for x in header_files_val]
        header_files_test = [str(Path(x).absolute()) for x in header_files_test]

        save_dataset(
            x_train, y_train, header_files_train,
            x_val, y_val, header_files_val,
            x_test, y_test, header_files_test,
            classes, output_directory=output_directory
        )

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
    K.clear_session()
    np.random.seed(1)
    tf.random.set_seed(1)

    model = BaselineMultibranch(
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
        use_multiprocessing=N_JOBS > 1
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

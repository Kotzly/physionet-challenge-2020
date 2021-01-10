#!/usr/bin/env python

import numpy as np
import joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from get_12ECG_features import get_12ECG_features
import tqdm
from pathlib import Path
import re
import os
from os.path import join

from sklearn.model_selection import train_test_split

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from physionet_challenge.model.tabnet import TabNetModel
from physionet_challenge.data.load import get_split_subjects, load_dataset
from physionet_challenge.data.save import save_dataset
from physionet_challenge.model.baseline import BaselineMultibranch
import warnings
from physionet_challenge.utils.data import CLASSES

warnings.simplefilter("ignore")

N_JOBS = os.cpu_count()

def train(input_dir, output_dir, classes=CLASSES, split_filepath=None, checkpoint_folder=None, model="mlp", seed=1, monitor="val_loss"):

    classes = CLASSES
    train_subjects = get_split_subjects(split_filepath, split="train", dataset_directory=input_dir)
    val_subjects = get_split_subjects(split_filepath, split="validation", dataset_directory=input_dir)

    if checkpoint_folder is not None:
        checkpoint_folder = Path(checkpoint_folder)
        if checkpoint_folder.exists():
            print("Found checkpoint")
            x_train = np.load(checkpoint_folder / "x_train.npy")
            y_train = np.load(checkpoint_folder / "y_train.npy")
            x_val = np.load(checkpoint_folder / "x_val.npy")
            y_val = np.load(checkpoint_folder / "y_val.npy")
        else:
            print("No checkpoint. Loading data")
            x_train, y_train = load_dataset(input_dir, classes, subjects=train_subjects)
            x_val, y_val = load_dataset(input_dir, classes, subjects=val_subjects)

            checkpoint_folder.mkdir(parents=True, exist_ok=True)
            save_dataset(x_train, y_train, checkpoint_folder, split="train")
            save_dataset(x_val, y_val, checkpoint_folder, split="val")
    else:
        x_train, y_train = load_dataset(input_dir, classes, subjects=train_subjects)
        x_val, y_val = load_dataset(input_dir, classes, subjects=val_subjects)
    
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
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_class = {
        "tabnet": TabNetModel,
        "mlp": BaselineMultibranch
    }[model]
    model = model_class(
        n_inputs=x_train.shape[1],
        n_classes=y_train.shape[1]
    )

    callbacks = EarlyStopping(
        monitor=monitor,
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

    model.summary()

    # Save model.
    print('Saving model...')

    # final_model={'model': model, 'imputer': imputer,'classes': classes}
    # filename = os.path.join(output_directory, 'finalized_model.sav')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    artifacts = {
        'imputer': imputer,
        'classes': classes,
        'scaler': scaler
    }
    filepath = os.path.join(output_dir, 'artifacts.joblib')
    joblib.dump(artifacts, filepath, protocol=0)    
    joblib.dump(history.history, os.path.join(output_dir, 'history.joblib'))
    model.save(
        os.path.join(output_dir, "model"),
        include_optimizer=False
    )

    # model_filename = os.path.join(output_directory, 'model.sav')
    # joblib.dump({'model': model}, model_filename, protocol=0)
    
    # imputer_filename = os.path.join(output_directory, 'imputer.sav')
    # joblib.dump({'imputer': imputer}, imputer_filename, protocol=0)

    # classes_filename = os.path.join(output_directory, 'classes.sav')
    # joblib.dump({'classes': classes}, classes_filename, protocol=0)
    
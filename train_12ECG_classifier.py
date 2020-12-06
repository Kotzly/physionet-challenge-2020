#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from get_12ECG_features import get_12ECG_features
import tqdm
import multiprocessing as mp
from pathlib import Path
from itertools import chain
import re

N_JOBS = os.cpu_count()

def train_12ECG_classifier(input_directory, output_directory, n_jobs=4):
    # Load data.
    header_files = []
    print('Listing files...')
    for f in Path(input_directory).glob("*.hea"):
        header_files.append(f)
    print("Loading data...")
    classes, features, labels = load_challenge_data(header_files)

    # Train model.
    print('Training model...')

    # Replace NaN values with mean values
    imputer=SimpleImputer().fit(features)
    features=imputer.transform(features)

    # Train the classifier
    model = RandomForestClassifier(max_depth=8).fit(features,labels)

    # Save model.
    print('Saving model...')

    final_model={'model': model, 'imputer': imputer,'classes': classes}
    filename = os.path.join(output_directory, 'finalized_model.sav')
    joblib.dump(final_model, filename, protocol=0)
    
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
    classes, labels = get_classes(header_files)

    mat_files = [str(header_file).replace('.hea', '.mat') for header_file in header_files]

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

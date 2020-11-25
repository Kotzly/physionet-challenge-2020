#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from get_12ECG_features import get_12ECG_features
import tqdm
import multiprocessing as mp

def train_12ECG_classifier(input_directory, output_directory, n_jobs=4):
    # Load data.
    header_files = []
    print('Listing files...')
    for f in tqdm.tqdm(os.listdir(input_directory)):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    classes = get_classes(input_directory, header_files)
    num_classes = len(classes)
    num_files = len(header_files)

    features = list()
    labels = list()
    
    print("Loading data...")
    
    for i in tqdm.tqdm(range(num_files)):
        recording, header = load_challenge_data(header_files[i])

        tmp = get_12ECG_features(recording, header)
        features.append(tmp)

        for l in header:
            if l.startswith('#Dx:'):
                labels_act = np.zeros(num_classes)
                arrs = l.strip().split(' ')
                for arr in arrs[1].split(','):
                    class_index = classes.index(arr.rstrip()) # Only use first positive index
                    labels_act[class_index] = 1
        labels.append(labels_act)

    features = np.array(features)
    labels = np.array(labels)

    # Train model.
    print('Training model...')

    # Replace NaN values with mean values
    imputer=SimpleImputer().fit(features)
    features=imputer.transform(features)

    # Train the classifier
    model = RandomForestClassifier(max_depth=15).fit(features,labels)

    # Save model.
    print('Saving model...')

    final_model={'model': model, 'imputer': imputer,'classes': classes}
    filename = os.path.join(output_directory, 'finalized_model.sav')
    joblib.dump(final_model, filename, protocol=0)
    
    model_filename = os.path.join(output_directory, 'model.sav')
    joblib.dump({'model': model}, model_filename, protocol=0)
    
    imputer_filename = os.path.join(output_directory, 'imputer.sav')
    joblib.dump({'imputer': imputer}, imputer_filename, protocol=0)

    classes_filename = os.path.join(output_directory, 'classes.sav')
    joblib.dump({'classes': classes}, classes_filename, protocol=0)
    

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    print("Loading classes...")
    for filename in tqdm.tqdm(filenames):
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)

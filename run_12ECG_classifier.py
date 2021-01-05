#!/usr/bin/env python

import numpy as np, os, sys
import joblib
from get_12ECG_features import get_12ECG_features
from tensorflow.keras.models import load_model
from os.path import join

def run_12ECG_classifier(data, header_data, artifacts):


    # Use your classifier here to obtain a label and score for each class.
    model = artifacts['model']
    imputer = artifacts['imputer']
    classes = artifacts['classes']
    scaler = artifacts['scaler']

    features=np.asarray(get_12ECG_features(data, header_data))
    feats_reshape = features.reshape(1, -1)
    feats_reshape = imputer.transform(feats_reshape)
    feats_reshape = scaler.transform(feats_reshape)
    pred = model.predict(feats_reshape)[0]
    current_label = (pred > .5)
    current_label = current_label.astype(int)
    
    current_score = pred
    return current_label, current_score,classes

def load_12ECG_artifacts(files_path):
    # load the model from disk 
    filename = join(files_path, "artifacts.joblib")

    artifacts = joblib.load(filename)

    model = load_model(join(files_path, "model"))
    
    artifacts["model"] = model
    
    return artifacts

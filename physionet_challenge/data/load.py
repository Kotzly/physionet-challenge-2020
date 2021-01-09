import numpy as np
import os
from scipy.io import loadmat
from os.path import join
from physionet_challenge.processing.features import baseline_features, get_metadata_from_file
from physionet_challenge.data.split import split_dataset
from tensorflow.keras.models import load_model
import multiprocessing as mp
import joblib
import re
import json
from pathlib import Path
from itertools import chain

N_JOBS = os.cpu_count() - 1


def load_data_and_header(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


def load_recording_from_file(mat_filepath):
    data = loadmat(mat_filepath)
    recording = np.asarray(data['val'], dtype=np.float64)
    return recording


def load_features_from_file(mat_filepath, header_filepath):
    # Principal function.
    recording = load_recording_from_file(mat_filepath)
    metadata = get_metadata_from_file(header_filepath)
    features = baseline_features(recording, metadata)
    return features


def load_features_from_files(mat_files, header_files):
    with mp.Pool(N_JOBS) as pool:
        features = pool.starmap(
            load_features_from_file,
            zip(mat_files, header_files)
        )
    features = np.array(features)
    return features


def load_dataset(dataset_directory, subjects=None):
    if subjects is None:
        subjects = [os.path.splitext(filepath.name) for filepath in Path(dataset_directory).glob(".hea")]

    header_filepaths = [Path(dataset_directory) / (subject + ".hea") for subject in subjects]
    mat_filepaths = [Path(dataset_directory) / (subject + ".hea") for subject in subjects]

    for i in range(len(subjects)):
        assert header_filepaths[i].exists()
        assert mat_filepaths[i].exists()

    print("\tLoading labels")
    classes, labels = get_labels(header_filepaths)

    print("\tLoading features")
    features = load_features_from_files(mat_filepaths, header_filepaths)

    return classes, features, labels


def get_labels_from_txt(txt):
    # Find unique classes.
    classes_txt = re.findall("#Dx:\s*([0-9a-zA-Z\,]*)\n", txt)[0]
    classes = list([c.strip() for c in classes_txt.split(",")])
    return classes


def get_labels_from_file(filename):
    with open(filename, 'r') as f:
        classes = get_labels_from_txt(f.read())
    return classes


def get_labels(filepaths, all_classes_ordered):
    with mp.Pool(N_JOBS) as pool:
        classes_list = pool.map(get_labels_from_file, filepaths)

    all_classes_ordered = sorted(set(list(chain(*classes_list))))

    labels = np.zeros((len(filepaths), len(all_classes_ordered)))

    for i, file_classes in enumerate(classes_list):
        for c in file_classes:
            labels[i, all_classes_ordered.index(c)] = 1

    return labels


def load_artifacts(files_path):
    # load the model from disk 
    filename = join(files_path, "artifacts.joblib")

    artifacts = joblib.load(filename)

    model = load_model(join(files_path, "model"))
    
    artifacts["model"] = model
    
    return artifacts


def get_split_subjects(split_filepath, split="train", dataset_directory=None):
    with open(split_filepath, "r") as file:
        subjects = json.load(file)[split]
    if dataset_directory is not None:
        filepaths = Path(dataset_directory).glob("*")
        dir_subjects = [os.path.splitext(filepath.name) for filepath in filepaths]
        for subject in subjects:
            assert subject in dir_subjects, f"Subject {subject} is not in {dataset_directory}"
    return subjects


def load_features_labels(dataset_directory, split_filepath=None, split=None, processing=None):

    subjects = get_split_subjects(split_filepath, split=split, extension=".hea")
    classes, features, labels = load_dataset(dataset_directory, subjects)

    return features, labels


def load_all_classes(dataset_directory):
    classes, _ = split_dataset(dataset_directory)
    return classes
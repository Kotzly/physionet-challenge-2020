import numpy as np
import os
from scipy.io import loadmat
from os.path import join
from physionet_challenge.processing.features import baseline_features, get_metadata_from_file, spectogram_feature
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


def load_features_from_file(mat_filepath, header_filepath, processing_fn=None):
    # Principal function.
    if processing_fn is None:
        processing_fn = baseline_features
    recording = load_recording_from_file(mat_filepath)
    metadata = get_metadata_from_file(header_filepath)
    features = processing_fn(recording, metadata)
    return features


def load_spectogram_features_from_file(mat_filepath, header_filepath):
    recording = load_recording_from_file(mat_filepath)
    metadata = get_metadata_from_file(header_filepath)
    features = spectogram_feature(recording, metadata)
    return features

def load_baseline_features_from_file(mat_filepath, header_filepath):
    recording = load_recording_from_file(mat_filepath)
    metadata = get_metadata_from_file(header_filepath)
    features = baseline_features(recording, metadata)
    return features


def load_features_from_files(mat_files, header_files, file_processing_fn=None):
    if file_processing_fn is None:
        file_processing_fn = load_baseline_features_from_file
    with mp.Pool(N_JOBS) as pool:
        features = pool.starmap(
            file_processing_fn,
            zip(mat_files, header_files)
        )
    features = np.stack(features, axis=0)
    return features


def load_dataset(dataset_directory, classes=None, subjects=None, processing="baseline"):
    if subjects is None:
        subjects = [os.path.splitext(filepath.name) for filepath in Path(dataset_directory).glob(".hea")]

    PROCESSING_FN_DICT = {
        "baseline": load_baseline_features_from_file,
        "transfer": load_spectogram_features_from_file
    }

    if classes is None:
        classes = load_all_classes(dataset_directory)

    if isinstance(processing, str) or processing is None:
        processing = PROCESSING_FN_DICT[processing]
    

    header_filepaths = [Path(dataset_directory) / (subject + ".hea") for subject in subjects]
    mat_filepaths = [Path(dataset_directory) / (subject + ".mat") for subject in subjects]

    for i in range(len(subjects)):
        assert header_filepaths[i].exists()
        assert mat_filepaths[i].exists()

    print("\tLoading labels")
    labels = get_labels(header_filepaths, classes)

    print("\tLoading features")
    features = load_features_from_files(mat_filepaths, header_filepaths, file_processing_fn=processing)

    return features, labels


def get_labels_from_txt(txt):
    # Find unique classes.
    classes_txt = re.findall("#Dx:\s*([0-9a-zA-Z\,]*)\n", txt)[0]
    classes = list([c.strip() for c in classes_txt.split(",")])
    return classes


def get_labels_from_file(filename, all_classes_ordered=None):
    with open(filename, 'r') as f:
        labels_txt = get_labels_from_txt(f.read())

    if all_classes_ordered is None:
        return labels_txt

    labels = np.zeros((1, len(all_classes_ordered)))
    for c in labels_txt:
        labels[0, all_classes_ordered.index(c)] = 1

    return labels


def get_labels(filepaths, all_classes_ordered):
    with mp.Pool(N_JOBS) as pool:
        labels_txt = pool.map(get_labels_from_file, filepaths)

    labels = np.zeros((len(labels_txt), len(all_classes_ordered)))
    for i, label in enumerate(labels_txt):
        for c in label:
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
        dir_subjects = [os.path.splitext(filepath.name)[0] for filepath in filepaths]
        for subject in subjects:
            assert subject in dir_subjects, f"Subject {subject} is not in {dataset_directory}"
    return subjects


def load_features_labels(dataset_directory, classes, split_filepath=None, split=None):

    subjects = get_split_subjects(split_filepath, split=split)
    features, labels = load_dataset(dataset_directory, subjects, classes)

    return features, labels


def load_all_classes(dataset_directory):
    classes, _ = split_dataset(dataset_directory)
    return classes


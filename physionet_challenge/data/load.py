import numpy as np, os, sys
from scipy.io import loadmat
import tqdm
from os.path import join
from pathlib import Path
from processing.features import baseline_features
from physionet_challenge.load import load_model
import multiprocessing as mp
import joblib

N_JOBS = os.cpu_count() - 1


def load_data_and_header(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data

# Load challenge data.
def load_features_from_file(mat_filepath, header_filepath):

    with open(header_filepath, 'r') as f:
        header = f.readlines()

    x = loadmat(mat_filepath)
    recording = np.asarray(x['val'], dtype=np.float64)
    feature = baseline_features(recording, header)
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

def load_dataset(output_directory):
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

    return x_train, y_train, header_files_train, x_val, y_val, header_files_val, x_test, y_test, header_files_test, classes
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


def load_artifacts(files_path):
    # load the model from disk 
    filename = join(files_path, "artifacts.joblib")

    artifacts = joblib.load(filename)

    model = load_model(join(files_path, "model"))
    
    artifacts["model"] = model
    
    return artifacts

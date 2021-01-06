#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from physionet_challenge.data.load import load_artifacts, load_data_and_header
from physionet_challenge.model.inference import run_classifier_for_subject
import tqdm
from os.path import join
from pathlib import Path
from physionet_challenge.data.load import load_data_and_header
from physionet_challenge.data.save import save_challenge_predictions


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 4:
        raise Exception('Include the model, input and output directories as arguments, e.g., python driver.py model input output.')

    model_input = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]

    # Find files.
    if not os.path.isfile(join(model_input, "header_files_test.npy")):
        input_files = []
        for f in os.listdir(input_directory):
            if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
                input_files.append(f)
    else:
        input_files = list(np.load(join(model_input, "header_files_test.npy")))
        input_files = [f.replace(".hea", ".mat") for f in input_files if f.lower().endswith(".hea") and not f.startswith('.')]

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading 12ECG model...')
    artifacts = load_artifacts(model_input)

    print("Making predictions...")

    num_files = len(input_files)

    for i, f in enumerate(input_files):
        if i % 1000 == 1:
            print('\t{}/{}...'.format(i+1, num_files))
        data, header_data = load_data_and_header(f)
        current_label, current_score, classes = run_classifier_for_subject(data, header_data, artifacts)
        # Save results.
        save_challenge_predictions(output_directory, Path(f).name, current_score, current_label, classes)

    print('Done.')

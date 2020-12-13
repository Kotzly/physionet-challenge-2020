#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_artifacts, run_12ECG_classifier
import tqdm
from os.path import join

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data


def save_challenge_predictions(output_directory, filename, scores, labels, classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')



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

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading 12ECG model...')
    artifacts = load_12ECG_artifacts(model_input)

    print("Making predictions...")

    num_files = len(input_files)

    for i, f in enumerate(input_files):
        if i % 1000 == 1:
            print('\t{}/{}...'.format(i+1, num_files))
        data,header_data = load_challenge_data(f)
        current_label, current_score, classes = run_12ECG_classifier(data, header_data, artifacts)
        # Save results.
        save_challenge_predictions(output_directory, f, current_score, current_label, classes)


    print('Done.')

import numpy as np, os, sys
from scipy.io import loadmat
import tqdm
from os.path import join
from pathlib import Path

def save_challenge_predictions(output_directory, filename, scores, labels, classes):

    recording = os.path.splitext(filename)[0]
    new_filename = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory, new_filename)
    
    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

def save_dataset(
    x_train, y_train, header_files_train,
    x_val, y_val, header_files_val,
    x_test, y_test, header_files_test,
    classes, output_directory="."):

    np.save(join(output_directory, "x_train.npy"), x_train)
    np.save(join(output_directory, "y_train.npy"), y_train)
    np.save(join(output_directory, "header_files_train.npy"), header_files_train)

    np.save(join(output_directory, "x_val.npy"), x_val)
    np.save(join(output_directory, "y_val.npy"), y_val)
    np.save(join(output_directory, "header_files_val.npy"), header_files_val)

    np.save(join(output_directory, "x_test.npy"), x_test)
    np.save(join(output_directory, "y_test.npy"), y_test)
    np.save(join(output_directory, "header_files_test.npy"), header_files_test)
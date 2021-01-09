import numpy as np
import os
from os.path import join


def save_challenge_predictions(
    output_directory,
    filename,
    scores,
    labels,
    classes
):

    recording = os.path.splitext(filename)[0]
    new_filename = filename.replace('.mat', '.csv')
    output_file = os.path.join(output_directory, new_filename)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')


def save_dataset(
    features, labels,
    output_directory=".",
    split="none"
):

    np.save(join(output_directory, f"x_{split}.npy"), features)
    np.save(join(output_directory, f"y_{split}.npy"), labels)

import os
from scipy.io import loadmat
from physionet_challenge.data.load import load_artifacts, load_data_and_header
from physionet_challenge.model.inference import run_classifier_for_subject
import tqdm
from os.path import join
from pathlib import Path
from data.split import load_split_json
from physionet_challenge.data.save import save_challenge_predictions
import argparse

if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Input model directory.")
    parser.add_argument("input_dir", help="Input data directory.")
    parser.add_argument("output_dir", help="Output predictions directory.")
    parser.add_argument(
        "--split_filepath",
        help="Split filepath.",
        default=None
    )
    parser.add_argument("--split", help="Split name.", default="train")
    
    args = parser.parse_args()

    model_input = args.model_dir
    input_directory = args.input_dir
    output_directory = args.output_dir

    if args.split_filepath is not None:
        split_dict = load_split_json(args.split_filepath)
        split_subjects = split_dict[args.split]

    # Find files.
    _input_files = Path(input_directory).glob("*.mat")
    _input_files = [x.name for x in input_files]

    input_files = []
    for subject in split_subjects:
        mat_file = subject + ".mat"
        assert mat_file in _input_files
        input_files.append(join(input_directory, mat_file)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading 12ECG model...')
    artifacts = load_artifacts(model_input)

    print("Making predictions...")

    num_files = len(input_files)

    for i, filepath in enumerate(input_files):
        if i % 1000 == 1:
            print('\t{}/{}...'.format(i+1, num_files))
        data, header_data = load_data_and_header(filepath)
        current_label, current_score, classes = run_classifier_for_subject(
            data, header_data, artifacts
        )
        # Save results.
        save_challenge_predictions(
            output_directory, 
            Path(filepath).name,
            current_score,
            current_label,
            classes
        )

    print('Done.')

import os
from scipy.io import loadmat
from physionet_challenge.data.load import load_artifacts, load_data_and_header, load_features_from_files
from physionet_challenge.model.inference import run_classifier_for_subject
import tqdm
from os.path import join
from pathlib import Path
from physionet_challenge.data.split import load_split_json
from physionet_challenge.data.save import save_challenge_predictions
import argparse

# python physionet_challenge\command\evaluation.py model_new ..\..\Datasets\Physionet2020Challenge\all inference_new --split_filepath split.json --split train
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

    if args.split is not None:
        split_dict = load_split_json(args.split_filepath)
        split_subjects = split_dict[args.split]

    # Find files.
    _input_files = Path(input_directory).glob("*.mat")
    _input_files = [x.name for x in _input_files]

    mat_filepaths = []
    header_filepaths = []
    input_files = []
    for subject in split_subjects:
        mat_file = subject + ".mat"
        h_file = subject + ".hea"
        assert mat_file in _input_files
        input_files.append(
            (
                join(input_directory, mat_file),
                join(input_directory, h_file),
            )
        )
        mat_filepaths.append(join(input_directory, mat_file))
        header_filepaths.append(join(input_directory, h_file))

    print("Loaded {} subjects from split {}".format(len(input_files), args.split))

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading 12ECG model...')
    artifacts = load_artifacts(model_input)

    print("Making predictions...")

    num_files = len(input_files)

    #############################################
    
    model = artifacts['model']
    imputer = artifacts['imputer']
    classes = artifacts['classes']
    scaler = artifacts['scaler']

    features = load_features_from_files(mat_filepaths, header_filepaths)
    feats_reshape = imputer.transform(features)
    feats_reshape = scaler.transform(feats_reshape)
    current_score = model.predict(feats_reshape)
    current_label = (current_score > .5).astype(int)

    for i in range(len(features)):
        if i % 1000 == 1:
            print('\t{}/{}...'.format(i+1, num_files))
        save_challenge_predictions(
            output_directory, 
            Path(mat_filepaths[i]).name,
            current_score[i],
            current_label[i],
            classes
        )

    # for i, (mat_filepath, header_filepath) in enumerate(input_files):
    #     if i % 1000 == 1:
    #         print('\t{}/{}...'.format(i+1, num_files))
    #     # data, header_data = load_data_and_header(mat_filepath, header_filepath)
    #     current_label, current_score, classes = run_classifier_for_subject(
    #         mat_filepath, header_filepath, artifacts
    #     )
    #     # Save results.
    #     save_challenge_predictions(
    #         output_directory, 
    #         Path(mat_filepath).name,
    #         current_score,
    #         current_label,
    #         classes
    #     )

    print('Done.')

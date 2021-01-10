from pathlib import Path
from physionet_challenge.processing.features import get_metadata_from_file
import multiprocessing as mp
# from sklearn.model_selection import train_test_split
import json
import numpy as np
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)

FOLDS_NAMES = ["train", "test", "validation"]
PROPORTIONS = (.55, .2, .25)

N_JOBS = os.cpu_count()


def load_split_json(filepath):
    with open(filepath, "r") as file:
        content = json.load(file)
    return content


def readlines(filepath):
    with open(filepath, "r") as file:
        content = file.readlines()
    return content


def train_test_split(arr, test_size=.2, random_state=1):
    arr = np.array(arr)
    assert test_size <= 1 and test_size >= 0
    train_size = 1 - test_size
    np.random.seed(random_state)
    N = len(arr)
    idx = np.random.permutation(np.arange(N))
    N_split = int(np.ceil(N * train_size))
    x1 = arr[idx[:N_split]]
    x2 = arr[idx[N_split:]]
    return x1, x2

#python physionet_challenge\data\split.py ..\..\Datasets\Physionet2020Challenge\all --split_filepath split.json --classes_filepath classes.npy
def split_dataset(
    input_folder,
    split_filepath=None,
    classes_filepath=None,
    proportions=PROPORTIONS,
    folds_names=FOLDS_NAMES,
    seed=1
):

    assert sum(proportions) == 1, "Sum of proportions must be 1."
    if folds_names is None:
        folds_names = [f"split_{i}" for i in range(len(proportions))]
    assert len(proportions) == len(folds_names)
    
    logging.info("Listing dataset files.")
    header_filepaths = list(
        sorted(
            Path(input_folder).glob("*.hea"),
            key=lambda x: x.name
        )
    )

    logging.info("Reading files metadata.")
    
    with mp.Pool(N_JOBS) as pool:
        datapool = pool.map(get_metadata_from_file, header_filepaths)
    # datapool = []
    # for i in range(len(header_filepaths)):
    #     print(header_filepaths[i])
    #     datapool.append(get_metadata_from_file(header_filepaths[i]))

    # Creates a dictionary of keys=diagnoses and items=list of subject with
    # the diagnose.
    logging.info("Creating subject list.")
    
    all_diagnoses = {}
    np.random.seed(seed)
    for data in datapool:
        assert len(data["dx"]) > 0
        
        put = False
        for d in data["dx"]:
            if d not in all_diagnoses or len(all_diagnoses[d]) < 3:
                if d not in all_diagnoses:
                    all_diagnoses[d] = []
                all_diagnoses[d].append(data["subject_id"])
                put = True
                break
        if not put:
            all_diagnoses[np.random.choice(data["dx"])].append(data["subject_id"])

    print(sum(len(x) for x in all_diagnoses.values()))    
    # Creates splits
    splits = {fold: [] for fold in folds_names}

    if split_filepath is not None:
        logging.info("Creating splits")
        for diagnose, subject_list in all_diagnoses.items():
            for i in range(len(proportions) - 1):
                p = sum(proportions[i+1:])/sum(proportions[i:])
                s_this, subject_list = train_test_split(
                    subject_list, test_size=p, random_state=seed
                )
                splits[folds_names[i]].extend(s_this)
            splits[folds_names[-1]].extend(subject_list)
            # Save splits
        with open(split_filepath, "w") as file:
            json.dump(splits, file, indent=4)

        with open(Path(split_filepath).parent / "found_classes.json", "w") as file:
            json.dump(all_diagnoses, file, indent=4)

    classes = list(sorted(all_diagnoses.keys()))
    if classes_filepath is not None:
        np.save(classes_filepath, classes)

    logging.info("Done.")

    return splits, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Input directory.")
    parser.add_argument("--split_filepath", help="Filepath to save split json.")
    parser.add_argument("--classes_filepath", help="Filepath to save classes.")
    parser.add_argument("--proportions", help="Proportions, separated by comma.", default=None)
    parser.add_argument("--names", help="Name of folds.", default=None)
    parser.add_argument("--seed", help="Seed.", default=1)
    args = parser.parse_args()
    
    if args.proportions is None:
        proportions = PROPORTIONS
    else:
        proportions = [float(x.strip()) for x in args.proportions.split(",")]

    if args.names is None:
        names = FOLDS_NAMES
    else:
        names = [x.strip().lower() for x in args.names.split(",")]

    split_dataset(
        args.input_folder,
        args.split_filepath,
        args.classes_filepath,
        proportions=proportions,
        folds_names=names,
        seed=1
    )
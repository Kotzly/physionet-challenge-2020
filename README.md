# PhysioNet/CinC Challenge 2020  Classification code

The current important files are:
    - data/split.py - Create the train/val/test splits.
    - command/train_model.py - Run the training script.
    - command/inference.py - Create the inference files.
    - evaluation/evaluation.py - Use the inference files to calculate the challenge metrics.

They need to be run in this order.

## Contents

This code uses two main scripts to train the model and classify the data:

* `train_model.py` Train your model. Add your model code to the `train_12ECG_model` function. It also performs all file input and output. **Do not** edit this script or we will be unable to evaluate your submission.
* `driver.py` is the classifier which calls the output from your `train_model` script. It also performs all file input and output. **Do not** edit this script or we will be unable to evaluate your submission.

Check the code in these files for the input and output formats for the `train_model` and `driver` scripts.

To create and save your model, you should edit `train_12ECG_classifier.py` script. Note that you should not change the input arguments of the `train_12ECG_classifier` function or add output arguments. The needed models and parameters should be saved in a separated file. In the sample code, an additional script, `get_12ECG_features.py`, is used to extract hand-crafted features. 

To run your classifier, you should edit the `run_12ECG_classifier.py` script, which takes a single recording as input and outputs the predicted classes and probabilities. Please, keep the formats of both outputs as they are shown in the example. You should not change the inputs and outputs of the `run_12ECG_classifier` function.

## Use

You can run this classifier code by installing the requirements and running

    python train_model.py training_data model   
    python driver.py model test_data test_outputs

where `training_data` is a directory of training data files, `model` is a directory of files for the model, `test_data` is the directory of test data files, and `test_outputs` is a directory of classifier outputs.  The [PhysioNet/CinC 2020 webpage](https://physionetchallenges.github.io/2020/) provides a training database with data files and a description of the contents and structure of these files.

## Submission

The `driver.py`, `get_12ECG_score.py`, and `get_12ECG_features.py` scripts must be in the root path of your repository. If they are inside a folder, then the submission will be unsuccessful.

## Details

See the [PhysioNet/CinC 2020 webpage](https://physionetchallenges.github.io/2020/) for more details, including instructions for the other files in this repository.


# PhysioNet/CinC Challenge 2020 Evaluation Metrics

This repository contains the Python and MATLAB evaluation code for the PhysioNet/Computing in Cardiology Challenge 2020. The `evaluate_12ECG_score` script evaluates the output of your algorithm using the evaluation metric that is described on the [webpage](https://physionetchallenges.github.io/2020/) for the PhysioNet/CinC Challenge 2020. While this script reports multiple evaluation metric, we use the last score (`Challenge Metric`) to evaluate your algorithm.

## Python

You can run the Python evaluation code by installing the NumPy Python package and running

    python evaluate_12ECG_score.py labels outputs scores.csv class_scores.csv

where `labels` is a directory containing files with one or more labels for each 12-lead ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a directory containing files with outputs produced by your algorithm for those recordings; `scores.csv` (optional) is a collection of scores for your algorithm; and `class_scores.csv` (optional) is a collection of per-class scores for your algorithm.

## MATLAB

You can run the MATLAB evaluation code by installing Python and the NumPy Python package and running

    evaluate_12ECG_score(labels, outputs, scores.csv, class_scores.csv)

where `labels` is a directory containing files with one or more labels for each 12-lead ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a directory containing files with outputs produced by your algorithm for those recordings; `scores.csv` (optional) is a collection of scores for your algorithm; and `class_scores.csv` (optional) is a collection of per-class scores for your algorithm.

## Troubleshooting

Unable to run this code with your code? Try one of the [baseline classifiers](https://physionetchallenges.github.io/2020/#submissions) on the [training data](https://physionetchallenges.github.io/2020/#data). Unable to install or run Python? Try  [Python](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/products/individual), or your package manager.

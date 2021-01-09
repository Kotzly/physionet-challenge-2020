#!/usr/bin/env python

import os
from physionet_challenge.training import train
import argparse

if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument(
        "--split",
        help="Split json filepath.",
        default=None
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint folder.",
        default=None
    )
    parser.add_argument(
        "--seed",
        help="Seed for reproducibility.",
        default=1
    )
    parser.add_argument("--model", help="Model type.", default="mlp")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    print('Running training code...')

    train(
        args.input_dir,
        args.output_dir,
        split_filepath=args.split,
        checkpoint_folder=args.checkpoint,
        model=args.model,
        seed=args.seed
    )

    print('Done.')

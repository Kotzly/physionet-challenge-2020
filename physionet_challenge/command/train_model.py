#!/usr/bin/env python

import os
from physionet_challenge.training.train import train
import argparse

def main(input_dir, output_dir, checkpoint_folder, split_filepath, model, seed, monitor):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print('Running training code...')

    train(
        input_dir,
        output_dir,
        split_filepath=split_filepath,
        checkpoint_folder=checkpoint_folder,
        model=model,
        seed=seed,
        monitor=monitor
    )

    print('Done.')


if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory.", type=str)
    parser.add_argument("output_dir", help="Output directory.", type=str)
    parser.add_argument(
        "--split",
        help="Split json filepath.",
        default=None,
        type=str
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint folder.",
        default=None,
        type=str
    )
    parser.add_argument(
        "--seed",
        help="Seed for reproducibility.",
        default=1,
        type=int
    )
    parser.add_argument("--model", help="Model type.", default="mlp", type=str)
    parser.add_argument("--monitor", help="Monitor variable for training.", default="val_loss", type=str)
    
    args = parser.parse_args()

    main(
        args.input_dir,
        args.output_dir,
        checkpoint_folder=args.checkpoint,
        split_filepath=args.split,
        model=args.model,
        seed=args.seed,
        monitor=args.monitor
    )
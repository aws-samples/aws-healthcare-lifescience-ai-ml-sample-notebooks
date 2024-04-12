# Adapted from
# https://github.com/awslabs/neptuneml-toolkit/blob/main/examples/custom-models/introduction/movie-lens-rgcn/link-predict/src/transform.py

import argparse
from neptuneml_toolkit.transform import get_transform_config
from train import transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Whether script is running locally",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.local:
        hyperparameters = {
            "num-neighbors": 30,
            "batch_size": 1024,
            "lr": 0.0015355425376242019,
            "task": "link_predict",
            "model": "custom",
            "name": "custom-link_predict",
            "weight-decay": 0.0,
            "n-epochs": 2,
            "hidden-size": 128,
            "num-bases": 4,
            "num-encoder-layers": 1,
            "num-negs": 10,
        }
        data_path, model_path, devices = "./data", "./output", [-1]
    else:
        data_path, model_path, devices, hyperparameters = get_transform_config()

    transform(data_path, model_path, devices, hyperparameters)

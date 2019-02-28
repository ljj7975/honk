from collections import ChainMap
import argparse
# import datetime
import numpy as np
import os
import random
import sys

import torch
import torch.utils.data as data

import json

from . import model as mod
from .manage_audio import AudioPreprocessor

TEXT_COLOR = {
    'HEADER':'\033[95m',
    'OKBLUE':'\033[94m',
    'OKGREEN':'\033[92m',
    'WARNING':'\033[93m',
    'FAIL':'\033[91m',
    'ENDC':'\033[0m',
    'BOLD':'\033[1m',
    'UNDERLINE':'\033[4m'
}

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def generate_data_loaders(config, datasets):
    train_set = datasets["train_set"]
    dev_set = datasets["dev_set"]
    test_set = datasets["test_set"]

    train_loader = data.DataLoader(
        datasets["train_set"],
        batch_size=config["batch_size"],
        # shuffle=True,
        collate_fn=train_set.collate_fn)
    dev_loader = data.DataLoader(
        datasets["dev_set"],
        batch_size=min(len(dev_set), 16),
        shuffle=True,
        collate_fn=dev_set.collate_fn)
    test_loader = data.DataLoader(
        datasets["test_set"],
        batch_size=min(len(test_set), 16),
        shuffle=True,
        collate_fn=test_set.collate_fn)

    return {"train_loader":train_loader, "dev_loader":dev_loader, "test_loader":test_loader}


def reset_config(config, size_per_word, lr, n_epochs):
    config["size_per_word"] = size_per_word
    config["lr"] = lr
    config["n_epochs"] = n_epochs

def main():
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False,
                         n_epochs=500,
                         lr=[0.001],
                         schedule=[np.inf],
                         batch_size=64,
                         dev_every=10,
                         seed=0,
                         use_nesterov=False,
                         input_file="",
                         output_file=output_file,
                         gpu_no=1,
                         cache_size=32768,
                         momentum=0.9,
                         weight_decay=0.00001,
                         personalized=False)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.PersonalizedSpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval"], default="train", type=str)
    parser.add_argument("--exp_type", choices=["lr", "epochs", "data_size", "all"], default="all", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)

    config['model_dir'] = 'model/'

    default_lr = [0.01]
    default_size_per_word = 10
    default_n_epochs = 20

    reset_config(config, default_size_per_word, default_lr, default_n_epochs)

    print(TEXT_COLOR['OKGREEN'])
    print("default size_per_word :", config["size_per_word"])
    print("default lr :", config["lr"])
    print("default n_epochs :", config["n_epochs"])
    print(TEXT_COLOR['ENDC'])

    print("personalized = True, keep_original = False, batch_size =", config['batch_size'])
    config["personalized"] = True
    config["keep_original"] = False

    train_set, dev_set, test_set = mod.PersonalizedSpeechDataset.splits(config)
    datasets = {
        "train_set": train_set,
        "dev_set": dev_set,
        "test_set": test_set
    }

    print("\ntrain_set size = ", len(datasets["train_set"]))
    print("dev_set size = ", len(datasets["dev_set"]))
    print("test_set size = ", len(datasets["test_set"]))

    data_loaders = generate_data_loaders(config, datasets)

    print("\ntrain_loader size = (", len(data_loaders["train_loader"]), len(data_loaders["train_loader"].dataset), ")")
    print("dev_loader size = (", len(data_loaders["dev_loader"]), len(data_loaders["dev_loader"].dataset), ")")
    print("test_loader size = (", len(data_loaders["test_loader"]), len(data_loaders["test_loader"].dataset), ")")

    for i in range(3):
        print(TEXT_COLOR['OKGREEN'])
        print("iteration :", i)
        print(TEXT_COLOR['ENDC'])

        index = 0
        for data, label in data_loaders["train_loader"]:
            print('\n\tindex = ', index)
            print('\tdata size = ', data.size())
            print('\tlabel size = ', label.size())
            print('\t4th label = ', label[4].item())
            index += 1

            if i == 1:
                break

if __name__ == "__main__":
    main()

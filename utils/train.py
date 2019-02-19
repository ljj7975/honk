from collections import ChainMap
import argparse
import datetime
import os
import random
import sys

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

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

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()
    # print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)
    return accuracy

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def evaluate(config, test_loader=None, model=None):
    if not test_loader:
        if config["personalized"]:
            _, _, test_set = mod.PersonalizedSpeechDataset.splits(config)
        else:
            _, _, test_set = mod.SpeechDataset.splits(config)

        test_loader = data.DataLoader(
            test_set,
            batch_size=len(test_set),
            shuffle=True,
            collate_fn=test_set.collate_fn)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0

    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(print_eval("test", scores, labels, loss) * model_in.size(0))
        total += model_in.size(0)
    accuracy = sum(results) / total
    print("final test accuracy: {}".format(accuracy))
    return accuracy.item()

def train(config, data_loaders=None):
    if not data_loaders:
        if config["personalized"]:
            train_set, dev_set, test_set = mod.PersonalizedSpeechDataset.splits(config)
        else:
            train_set, dev_set, test_set = mod.SpeechDataset.splits(config)

        train_loader = data.DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True, drop_last=True,
            collate_fn=train_set.collate_fn)
        dev_loader = data.DataLoader(
            dev_set,
            batch_size=min(len(dev_set), 16),
            shuffle=True,
            collate_fn=dev_set.collate_fn)
        test_loader = data.DataLoader(
            test_set,
            batch_size=min(len(test_set), 16),
            shuffle=True,
            collate_fn=test_set.collate_fn)
    else:
        train_loader = data_loaders['train_loader']
        dev_loader = data_loaders['dev_loader']
        test_loader = data_loaders['test_loader']

    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    step_no = 0

    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
            print_eval("train step #{}".format(step_no), scores, labels, loss)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.item()
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            if avg_acc > max_acc:
                print("saving best model...")
                print("final dev accuracy: {}".format(avg_acc))
                max_acc = avg_acc
                model.save(config["output_file"])
    return evaluate(config, test_loader, model)

def evaluate_personalization(config, test_loader, acc_map=None):

    config["personalized"] = False
    config["keep_original"] = False
    acc = round(evaluate(config), 4)
    print("\n< accuracy on original data : ", acc, ">")
    if acc_map:
        acc_map['original'].append(acc)

    # config["personalized"] = True
    # config["keep_original"] = True
    # acc = round(evaluate(config), 4)
    # print("\n< accuracy on mixed personalized data : ", acc, ">")
    # if acc_map:
    #     acc_map['mixed'].append(acc)
    
    config["personalized"] = True
    config["keep_original"] = False
    acc = round(evaluate(config, test_loader), 4)
    print("\n< accuracy on pure personalized data : ", acc, ">")
    if acc_map:
        acc_map['personalized'].append(acc)

def evaluate_data_size(config):
    print(TEXT_COLOR['WARNING'] + "\n~~ personalization (size of personalized data set) ~~" + TEXT_COLOR['ENDC'])

    data_size = []
    acc_map = {
        'original':[],
        # 'mixed':[],
        'personalized':[]
    }

    for i in range(1, 11):
        config["size_per_word"] = i
        train_set, dev_set, test_set = mod.PersonalizedSpeechDataset.splits(config)
        datasets = {
            "train_set": train_set,
            "dev_set": dev_set,
            "test_set": test_set
        }
        data_loaders = generate_data_loaders(config, datasets)

        data_size.append(config["size_per_word"])
        new_model_file_name = config['model_dir'] + 'datasize_' + str(config["size_per_word"]) + '_' + config['model_file_suffix']
        print("\n\n~~ Size per keyword : " + str(config["size_per_word"]) + " ~~")
        print("~~ Model path : " + new_model_file_name + " ~~")
        config["input_file"] = config['original_model']
        config["output_file"] = new_model_file_name

        print("\n< train further only with personalized data >")
        config["personalized"] = True
        config["keep_original"] = False
        train(config, data_loaders)

        config["input_file"] = new_model_file_name
        evaluate_personalization(config, data_loaders['test_loader'], acc_map)

        print(TEXT_COLOR['WARNING'] + str(data_size[-1]) +' - ' + str(acc_map['personalized'][-1]) + TEXT_COLOR['ENDC'])

    best_index = np.argmax(acc_map['personalized'])
    return data_size, acc_map, best_index

def evaluate_lr(config, datasets):
    print(TEXT_COLOR['WARNING'] + "\n~~ personalization (learning rate) ~~" + TEXT_COLOR['ENDC'])

    lr = []
    acc_map = {
        'original':[],
        # 'mixed':[],
        'personalized':[]
    }

    for i in range(1, 5):
        data_loaders = generate_data_loaders(config, datasets)
        config["lr"] = [round(0.1**i, i)]
        lr.append(config["lr"][0])
        new_model_file_name = config['model_dir'] + 'lr_' + str(config["lr"][0]) + '_' + config['model_file_suffix']
        print("\n\n~~ learning rate : " + str(config["lr"][0]) + " ~~")
        print("~~ Model path : " + new_model_file_name + " ~~")
        config["input_file"] = config['original_model']
        config["output_file"] = new_model_file_name

        print("\n< train further only with personalized data >")
        config["personalized"] = True
        config["keep_original"] = False
        train(config, data_loaders)

        config["input_file"] = new_model_file_name
        evaluate_personalization(config, data_loaders['test_loader'], acc_map)

        print(TEXT_COLOR['WARNING'] + str(lr[-1]) +' - ' + str(acc_map['personalized'][-1]) + TEXT_COLOR['ENDC'])

    best_index = np.argmax(acc_map['personalized'])
    return lr, acc_map, best_index

def evaluate_epochs(config, datasets):
    print(TEXT_COLOR['WARNING'] + "\n~~ personalization (epochs) ~~" + TEXT_COLOR['ENDC'])

    epochs = []
    acc_map = {
        'original':[],
        # 'mixed':[],
        'personalized':[]
    }

    for i in range(5, 50, 5):
        data_loaders = generate_data_loaders(config, datasets)
        config["n_epochs"] = i
        epochs.append(config["n_epochs"])
        new_model_file_name = config['model_dir'] + 'lr_' + str(config["lr"][0]) + '_' + config['model_file_suffix']
        print("\n\n~~ number of epcohs : " + str(config["n_epochs"]) + " ~~")
        print("~~ Model path : " + new_model_file_name + " ~~")
        config["input_file"] = config['original_model']
        config["output_file"] = new_model_file_name

        print("\n< train further only with personalized data >")
        config["personalized"] = True
        config["keep_original"] = False
        train(config, data_loaders)

        config["input_file"] = new_model_file_name
        evaluate_personalization(config, data_loaders['test_loader'], acc_map)

        print(TEXT_COLOR['WARNING'] + str(epochs[-1]) +' - ' + str(acc_map['personalized'][-1]) + TEXT_COLOR['ENDC'])

    best_index = np.argmax(acc_map['personalized'])
    return epochs, acc_map, best_index

def generate_data_loaders(config, datasets):
    train_set = datasets["train_set"]
    dev_set = datasets["dev_set"]
    test_set = datasets["test_set"]

    train_loader = data.DataLoader(
        datasets["train_set"],
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
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

    if config["personalized"]:
        print("\npersonalization\n")

        default_lr = [0.01]
        default_size_per_word = 6
        default_n_epochs = 20

        if config["type"] == "train":
            config['model_file_suffix'] = '{:%d_%H_%M}.pt'.format(datetime.datetime.now())
            # retraining base model
            # original_model_file_name = config['model_dir'] + 'original_' + config['model_file_suffix']
            # config["output_file"] = original_model_file_name

            # print(TEXT_COLOR['WARNING'] + "\n~~ Training base model ~~" + TEXT_COLOR['ENDC'])
            # config["personalized"] = False
            # config["keep_original"] = False
            # train(config)

            # reusing pretrained base model
            original_model_file_name = "model/res8-narrow.pt"

            reset_config(config, default_size_per_word, default_lr, default_n_epochs)

            print(TEXT_COLOR['OKGREEN'])
            print("default size_per_word :", config["size_per_word"])
            print("default lr :", config["lr"])
            print("default n_epochs :", config["n_epochs"])
            print(TEXT_COLOR['ENDC'])

            config["personalized"] = True
            config["keep_original"] = False
            train_set, dev_set, test_set = mod.PersonalizedSpeechDataset.splits(config)
            datasets = {
                "train_set": train_set,
                "dev_set": dev_set,
                "test_set": test_set
            }

            print(TEXT_COLOR['WARNING'] + "\n~~ pre personalization evaluation ~~" + TEXT_COLOR['ENDC'])

            config["input_file"] = original_model_file_name
            pre_trained_acc_map = {
                'original':[],
                # 'mixed':[],
                'personalized':[]
            }

            data_loaders = generate_data_loaders(config, datasets)

            evaluate_personalization(config, data_loaders['test_loader'], pre_trained_acc_map)
            print(TEXT_COLOR['OKGREEN'])
            print('original - ', pre_trained_acc_map['original'])
            # print('mixed - ', pre_trained_acc_map['mixed'])
            print('personalized - ', pre_trained_acc_map['personalized'])
            print(TEXT_COLOR['ENDC'])

            # personalization experiment
            config['original_model'] = original_model_file_name

            print('experiment type : ', config["exp_type"])


            if config["exp_type"] == "all":
                reset_config(config, default_size_per_word, default_lr, default_n_epochs)
                data_size, data_size_acc_map, best_data_size_index = evaluate_data_size(config)

                best_data_size = data_size[best_data_size_index]
                best_data_size_acc = data_size_acc_map['personalized'][best_data_size_index]

                print(TEXT_COLOR['OKGREEN'])
                print("\n~~~~~~~~~~ best data size is " + str(best_data_size) + " with acc of " + str(best_data_size_acc) + "~~~~~~")
                print('datasize - ', data_size)
                print('original - ', data_size_acc_map['original'])
                # print('mixed - ', data_size_acc_map['mixed'])
                print('personalized - ', data_size_acc_map['personalized'])
                print(TEXT_COLOR['ENDC'])

                reset_config(config, default_size_per_word, default_lr, default_n_epochs)
                lr, lr_acc_map, best_lr_index = evaluate_lr(config, datasets)

                best_lr = lr[best_lr_index]
                best_lr_acc = lr_acc_map['personalized'][best_lr_index]

                print(TEXT_COLOR['OKGREEN'])
                print("\n~~~~~~~~~~ best learning rate is " + str(best_lr) + " with acc of " + str(best_lr_acc) + "~~~~~~")
                print('lr - ', lr)
                print('original - ', lr_acc_map['original'])
                # print('mixed - ', lr_acc_map['mixed'])
                print('personalized - ', lr_acc_map['personalized'])
                print(TEXT_COLOR['ENDC'])

                reset_config(config, default_size_per_word, default_lr, default_n_epochs)
                epochs, epochs_acc_map, best_epochs_index = evaluate_epochs(config, datasets)

                best_epochs = epochs[best_epochs_index]
                best_epochs_acc = epochs_acc_map['personalized'][best_epochs_index]

                print(TEXT_COLOR['OKGREEN'])
                print("\n~~~~~~~~~~ best number of epochs is " + str(best_epochs) + " with acc of " + str(best_epochs_acc) + "~~~~~~")
                print('epochs - ', epochs)
                print('original - ', epochs_acc_map['original'])
                # print('mixed - ', epochs_acc_map['mixed'])
                print('personalized - ', epochs_acc_map['personalized'])
                print(TEXT_COLOR['ENDC'])

                print(TEXT_COLOR['FAIL'])
                print("\nsummary :")
                print("best datasize :", best_data_size, best_data_size_acc)
                print("best lr :", best_lr, best_lr_acc)
                print("best epcohs :", best_epochs, best_epochs_acc)
                print(TEXT_COLOR['ENDC'])

            elif config["exp_type"] == "data_size":
                reset_config(config, default_size_per_word, default_lr, default_n_epochs)
                data_size, data_size_acc_map, best_data_size_index = evaluate_data_size(config)

                print(TEXT_COLOR['OKGREEN'])
                print("\n~~~~~~~~~~ best data size is " + str(data_size[best_data_size_index]) + " with acc of " + str(data_size_acc_map['personalized'][best_data_size_index]) + "~~~~~~")
                print('datasize - ', data_size)
                print('original - ', data_size_acc_map['original'])
                # print('mixed - ', data_size_acc_map['mixed'])
                print('personalized - ', data_size_acc_map['personalized'])
                print(TEXT_COLOR['ENDC'])

            elif config["exp_type"] == "lr":
                reset_config(config, default_size_per_word, default_lr, default_n_epochs)
                lr, lr_acc_map, best_lr_index = evaluate_lr(config, datasets)

                print(TEXT_COLOR['OKGREEN'])
                print("\n~~~~~~~~~~ best learning rate is " + str(lr[best_lr_index]) + " with acc of " + str(lr_acc_map['personalized'][best_lr_index]) + "~~~~~~")
                print('lr - ', lr)
                print('original - ', lr_acc_map['original'])
                # print('mixed - ', lr_acc_map['mixed'])
                print('personalized - ', lr_acc_map['personalized'])
                print(TEXT_COLOR['ENDC'])

            elif config["exp_type"] == "epochs":
                reset_config(config, default_size_per_word, default_lr, default_n_epochs)
                epochs, epochs_acc_map, best_epochs_index = evaluate_epochs(config, datasets)

                print(TEXT_COLOR['OKGREEN'])
                print("\n~~~~~~~~~~ best number of epochs is " + str(epochs[best_epochs_index]) + " with acc of " + str(epochs_acc_map['personalized'][best_epochs_index]) + "~~~~~~")
                print('epochs - ', epochs)
                print('original - ', epochs_acc_map['original'])
                # print('mixed - ', epochs_acc_map['mixed'])
                print('personalized - ', epochs_acc_map['personalized'])
                print(TEXT_COLOR['ENDC'])

        elif config["type"] == "eval":
            reset_config(config, default_size_per_word, default_lr, default_n_epochs)

            train_set, dev_set, test_set = mod.PersonalizedSpeechDataset.splits(config)
            datasets = {
                "train_set": train_set,
                "dev_set": dev_set,
                "test_set": test_set
            }

            pre_trained_acc_map = {
                'original':[],
                # 'mixed':[],
                'personalized':[]
            }

            data_loaders = generate_data_loaders(config, datasets)

            evaluate_personalization(config, data_loaders['test_loader'], pre_trained_acc_map)
    else:
        if config["type"] == "train":
            train(config)
        elif config["type"] == "eval":
            evaluate(config)

if __name__ == "__main__":
    main()

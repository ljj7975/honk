import subprocess
import re
import pprint
import json
import datetime
import os
import ast
import random
import sys
import tqdm

iteration = int(sys.argv[1])

command_template = 'python -m utils.train --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --n_epochs 26 --weight_decay 0.00001 --lr 0.1 0.01 0.001 --schedule 3000 6000 --model res8-narrow --data_folder /media/brandon/SSD/data/speech_dataset --seed {0} --gpu_no 0 --personalized --personalized_data_folder /media/brandon/SSD/data/personalized_speech_data/{1} --type eval --exp_type lr'

people = ["brandon", "jay", "jack", "max", "kevin", "joyce", "lee", "kang"]

def search(regex, line, index):
    out = None
    search = re.match(regex, line)
    if search:
        out = search.group(index).strip()
    return out

def get_defaults(lines):
    base_model = None
    dataset = None
    size_per_word = None
    lr = None
    epochs = None
    original_acc = None
    personlized_acc = None
    optimizers = None

    for line in lines:
        if not base_model:
            base_model = search('base model : (.*)', line, 1)

        if not dataset:
            dataset = search('personlized dataset : (.*)', line, 1)

        if not size_per_word:
            size_per_word = search('default size_per_word : (.*)', line, 1)

        if not lr:
            lr = search('default lr : (.*)', line, 1)

        if not epochs:
            epochs = search('default n_epochs : (.*)', line, 1)

        if not original_acc:
            original_acc = search('original - (.*)', line, 1)

        if not personlized_acc:
            personlized_acc = search('personalized - (.*)', line, 1)

        if not optimizers:
            optimizers = search('optimizers :  (.*)', line, 1)
    
    optimizers = optimizers.split(' ')

    output = {
        "base_model" : base_model,
        "dataset" : dataset,
        "size_per_word" : int(size_per_word),
        "lr" : float(lr),
        "epochs" : int(epochs),
        "original_acc" : float(original_acc),
        "personlized_acc" : float(personlized_acc),
        "optimizers" : optimizers
    }

    for key, value in output.items():
        if not value:
            print("failed to find", key)

    return output


def get_learning_rate(target_opt, lines):
    results = {}

    flag = False
    data_size = None
    lr = None
    original = None
    optimizer = None
    personlized = None

    for line in lines:
        if not flag and "~~~~~~~~~~ best learning rate" in line:
            flag = True
            data_size = None
            lr = None
            original = None
            optimizer = None
            personlized = None
        else:
            if not data_size:
                data_size = search('datasize = (.*)', line, 1)

            if not optimizer:
                optimizer = search('optimizer = (.*)', line, 1)

            if not lr:
                lr = search('lr = (.*)', line, 1)

            if not original:
                original = search('original = (.*)', line, 1)

            if not personlized:
                personlized = search('personalized = (.*)', line, 1)

            if data_size and lr and original and personlized:
                if optimizer == target_opt:
                    results[data_size] = {
                        "lr" : ast.literal_eval(lr),
                        "original" : ast.literal_eval(original),
                        "personlized" : ast.literal_eval(personlized)
                    }
                flag = False
    return results

def get_epochs(target_opt, lines):
    results = {}

    flag = False
    data_size = None
    epochs = None
    original = None
    optimizer = None
    personlized = None

    for line in lines:
        if not flag and "~~~~~~~~~ best number of epochs " in line:
            flag = True
            data_size = None
            epochs = None
            original = None
            optimizer = None
            personlized = None
        else:
            if not data_size:
                data_size = search('datasize = (.*)', line, 1)

            if not optimizer:
                optimizer = search('optimizer = (.*)', line, 1)

            if not epochs:
                epochs = search('epochs = (.*)', line, 1)

            if not original:
                original = search('original = (.*)', line, 1)

            if not personlized:
                personlized = search('personalized = (.*)', line, 1)

            if data_size and epochs and original and personlized:
                if optimizer == target_opt:
                    results[data_size] = {
                        "epochs" : ast.literal_eval(epochs),
                        "original" : ast.literal_eval(original),
                        "personlized" : ast.literal_eval(personlized)
                    }
                flag = False
    return results

for i in tqdm.tqdm(range(iteration)):
    dir_name = 'results/' + datetime.datetime.now().strftime('%m%d_%H%M%S')
    random_seed = random.randint(1,1001)
    print(i, dir_name)

    for person in people:
        command = command_template.format(random_seed, person)
        print(command)
        sys.stdout.flush()
        result = subprocess.run(command.split(), stdout=subprocess.PIPE)
        outputs = result.stdout.decode("utf-8").split("\n")

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        raw_file = os.path.join(dir_name, person + '_raw.txt')
        with open(raw_file, 'w') as file:
            for line in outputs:
                file.write("%s\n" % line)

        summary = {
            "command" : command,
            "setting" : get_defaults(outputs),
        }

        for optimizer in summary["setting"]["optimizers"]:
            summary[optimizer] = {
                "lr" : get_learning_rate(optimizer, outputs),
                "epochs" : get_epochs(optimizer, outputs)
                }

        summary_file = os.path.join(dir_name, person + '_summary.txt')
        with open(summary_file, 'w') as file:
            pprint.pprint(summary, stream=file)

        print("experiment for " + person + "is completed")
        print("\tsummary :", summary_file)
        print("\traw_file :", raw_file)
        sys.stdout.flush()

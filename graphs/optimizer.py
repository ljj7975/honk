import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from pprint import pprint
import pylab

fig, axs = plt.subplots(4, 2, figsize=[12,16])

axs_mapping = {
    "brandon" : axs[0][0],
    "jay" : axs[0][1],
    "jack" : axs[1][0],
    "max" : axs[1][1],
    "lee" : axs[2][0],
    "kevin" : axs[2][1],
    "kang" : axs[3][0],
    "joyce" : axs[3][1]
}

per_acc = {
    "brandon" : [],
    "jay" : [],
    "jack" : [],
    "max" : [],
    "kevin" : [],
    "lee" : [],
    "kang" : [],
    "joyce" : []
}

optimizers = ["RMSprop", "adam", "adagrad", "SGD"]

def init_dict():
    template = {}

    template['1'] = {"RMSprop":[], "adam":[], "adagrad":[], "SGD": []}
    template['3'] = {"RMSprop":[], "adam":[], "adagrad":[], "SGD": []}
    template['5'] = {"RMSprop":[], "adam":[], "adagrad":[], "SGD": []}

    return template


original = {}
personalized = {}
for name in axs_mapping.keys():
    original[name] = init_dict()
    personalized[name] = init_dict()


result_dir = "../results"
metric = "optimizer"
learning_rate = None
base_model_acc = []



print("total exp : ", len(os.listdir(result_dir)))

for exp in os.listdir(result_dir):
    if exp.startswith("."):
        continue

    print(exp)
    for file_name in os.listdir(os.path.join(result_dir, exp)):
        if file_name.startswith("."):
            continue

        if "summary" in file_name:
            person = file_name.split("_")[0]

            file_path = os.path.join(result_dir, exp, file_name)
            summary = ast.literal_eval(open(file_path, 'r').read())
            try:
                epochs = summary['setting']['epochs']
            except Exception:
                epochs = 20
            base_model_acc.append(summary['setting']['original_acc'])
            per_acc[person].append(summary['setting']['personlized_acc'])
            results = summary[metric]

            for key in results.keys():
                for ind, optimizer in enumerate(optimizers):
                    original[person][key][optimizer].append(results[key]['original'][ind])
                    personalized[person][key][optimizer].append(results[key]['personlized'][ind])

base_model_acc = list(set(base_model_acc))
base_model_acc = np.mean(base_model_acc)

for key, value in per_acc.items():
    per_acc[key] = np.mean(value)

line_color = [ '#abdda4', '#fdae61', '#2b83ba', '#ffffbf', '#d7191c']

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def process_data(arr):
    mean_arr = []
    for optim in optimizers:
        mean_arr.append(np.mean(arr[optim]))

    return np.array(mean_arr)

legends = []
x_ticks = np.arange(1,5)

name_mapping = {
    "BRANDON":"A",
    "JAY":"B",
    "JACK":"C",
    "MAX":"D",
    "LEE":"E",
    "KEVIN":"F",
    "KANG":"G",
    "JOYCE":"H"
}

# acc_mapping = {
#     "BRANDON":"89.79",
#     "JAY":"88.75",
#     "JACK":"80.83",
#     "MAX":"78.12",
#     "KEVIN":"91.67",
#     "LEE":"89.17",
#     "KANG":"76.88",
#     "JOYCE":"90.83"
# }

width = 0.15

loc = (np.arange(6) * width) - (width * 3)
x_ticks = x_ticks[1:]

def plot_mean_and_CI(ind, axis, mean, color_mean=None, label=None):

    mean = mean[1:]

    legend = None
    if ind % 2 == 0:
        legend = axis.bar(x_ticks+loc[ind], mean, width, color=color_mean, alpha=.5, label=label)
    else:
        legend = axis.bar(x_ticks+loc[ind], mean, width, color=color_mean, label=label)
    return legend

for person in name_mapping.keys():
    person = person.lower()
    processed_original = []
    processed_personalized = []

    original_dict = original[person]
    personalized_dict = personalized[person]

    fig, axis = plt.subplots(1, 1, figsize=[7,5])

    axis.set_title('{0} - {1:.1f} %'.format("User " + name_mapping[person.upper()], round(per_acc[person] * 100, 1)), fontsize=23, y=1.04)

    axis.xaxis.set_ticks(x_ticks)
    axis.set_xticklabels(optimizers[1:])
    yl = axis.set_xlabel('Optimizer', fontsize=20, labelpad=14)
    xl = axis.set_ylabel('Accuracy', fontsize=20, labelpad=14)

    axis.set_ylim(0.8, 1.0)
    axis.set_yticks(np.arange(0.8, 1.03, 0.05))
    axis.yaxis.grid()

    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)


    # axis.grid()

    for i in original[person].keys():
        original_mean = process_data(original_dict[i])
        personalized_mean = process_data(personalized_dict[i])

        original_line = plot_mean_and_CI(int(i)-1, axis, original_mean, color_mean=line_color[int(i)-1], label='original')

        personalized_line = plot_mean_and_CI(int(i), axis, personalized_mean, color_mean=line_color[int(i)-1], label='personalized')

        legends.append(original_line)
        legends.append(personalized_line)


    plt.rcParams.update({'font.size': 40})

    fig.subplots_adjust(bottom=0.20, top=0.87, wspace=0.4, hspace=0.35, right=0.95, left=0.20)

    file_name = '{0}-{1:.1f} %'.format("User " + name_mapping[person.upper()], round(per_acc[person] * 100, 1))
    plt.savefig(f"optimizer-{name_mapping[person.upper()]}")

# create a second figure for the legend
figLegend = pylab.figure(figsize = (18.6,2.4))

# produce a legend for the objects in the other figure
pylab.figlegend(*axis.get_legend_handles_labels(), loc='upper left', ncol=3)

# save the two figures to files
figLegend.savefig("optimizer-legend.png")

import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

fig, axs = plt.subplots(4, 2, figsize=[12,24])

axs_mapping = {
    "brandon" : axs[0][0],
    "jay" : axs[0][1],
    "jack" : axs[1][0],
    "max" : axs[1][1],
    "kevin" : axs[2][0],
    "lee" : axs[2][1],
    "kang" : axs[3][0],
    "joyce" : axs[3][1]
}

per_acc = {}

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
base_model_acc = None



print("total exp : ", len(os.listdir(result_dir)))

for exp in os.listdir(result_dir):
    print(exp)
    for file_name in os.listdir(os.path.join(result_dir, exp)):
        if "summary" in file_name:
            person = file_name.split("_")[0]

            file_path = os.path.join(result_dir, exp, file_name)
            summary = ast.literal_eval(open(file_path, 'r').read())
            try:
                epochs = summary['setting']['epochs']
            except Exception:
                epochs = 20
            base_model_acc = summary['setting']['original_acc']
            per_acc[person] = summary['setting']['personlized_acc']
            results = summary[metric]

            for key in results.keys():
                for ind, optimizer in enumerate(optimizers):
                    original[person][key][optimizer].append(results[key]['original'][ind])
                    personalized[person][key][optimizer].append(results[key]['personlized'][ind])



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
    "KEVIN":"E",
    "LEE":"F",
    "KANG":"G",
    "JOYCE":"H"
}

acc_mapping = {
    "BRANDON":"89.79",
    "JAY":"88.75",
    "JACK":"80.83",
    "MAX":"78.12",
    "KEVIN":"91.67",
    "LEE":"89.17",
    "KANG":"76.88",
    "JOYCE":"90.83"
}

width = 0.15

loc = (np.arange(6) * width) - (width * 3)

def plot_mean_and_CI(ind, axis, mean, color_mean=None, label=None):
    legend = None
    if ind % 2 == 0:
        legend = axis.bar(x_ticks+loc[ind], mean, width, color=color_mean, alpha=.5, label=label)
    else:
        legend = axis.bar(x_ticks+loc[ind], mean, width, color=color_mean, label=label)
    return legend

for person, axis in axs_mapping.items():
    processed_original = []
    processed_personalized = []

    original_dict = original[person]
    personalized_dict = personalized[person]

    axis.set_title('{0} - {1} %'.format("User " + name_mapping[person.upper()], acc_mapping[person.upper()]))

    axis.set(xlabel='optimizer', ylabel='accuracy', xticks=x_ticks, xticklabels=optimizers)
    axis.set_ylim(0.0, 1.0)
    # axis.grid()

    for i in original[person].keys():
        original_mean = process_data(original_dict[i])
        personalized_mean = process_data(personalized_dict[i])

        original_line = plot_mean_and_CI(int(i)-1, axis, original_mean, color_mean=line_color[int(i)-1], label='original')

        personalized_line = plot_mean_and_CI(int(i), axis, personalized_mean, color_mean=line_color[int(i)-1], label='personalized')

        legends.append(original_line)
        legends.append(personalized_line)

fig.legend(legends,     # The line objects
           labels=["original_1", "personalized_1", "original_3", "personalized_3", "original_5", "personalized_5"],   # The labels
           loc=8,  # Position of legend
           ncol=3
           )


fig.subplots_adjust(bottom=0.05, top=0.94, wspace=0.18, hspace=0.25, right=0.95, left=0.15)
# fig.suptitle('Learning Rate ( epochs = {0} )\nbase model accuracy : {1} %'.format(epochs, round(base_model_acc * 100, 2)))
fig.suptitle('Optimizer\nbase model accuracy - {0} %'.format(91.36))

fig.savefig("optimizer.png")
# plt.show()

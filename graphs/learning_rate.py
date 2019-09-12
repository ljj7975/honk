import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from pprint import pprint

optimizer = "SGD"

if not os.path.exists(optimizer):
    os.makedirs(optimizer)

print(optimizer)

fig, axs = plt.subplots(4, 2, figsize=[12,16])
fig, axs = plt.subplots(2, 2, figsize=[8,8])

# axs_mapping = {
#     "brandon" : axs[0][0],
#     "jay" : axs[0][1],
#     "jack" : axs[1][0],
#     "max" : axs[1][1],
#     "lee" : axs[2][0],
#     "kevin" : axs[2][1],
#     "kang" : axs[3][0],
#     "joyce" : axs[3][1]
# }

axs_mapping = {
    "brandon" : axs[0][0],
    "jay" : axs[0][1],
    "jack" : axs[1][0],
    "kang" : axs[1][1]
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

original = {
    "brandon" : {'1':[], '3':[], '5':[]},
    "jay" : {'1':[], '3':[], '5':[]},
    "jack" : {'1':[], '3':[], '5':[]},
    "max" : {'1':[], '3':[], '5':[]},
    "kevin" : {'1':[], '3':[], '5':[]},
    "lee" : {'1':[], '3':[], '5':[]},
    "kang" : {'1':[], '3':[], '5':[]},
    "joyce" : {'1':[], '3':[], '5':[]}
}

personalized = {
    "brandon" : {'1':[], '3':[], '5':[]},
    "jay" : {'1':[], '3':[], '5':[]},
    "jack" : {'1':[], '3':[], '5':[]},
    "max" : {'1':[], '3':[], '5':[]},
    "kevin" : {'1':[], '3':[], '5':[]},
    "lee" : {'1':[], '3':[], '5':[]},
    "kang" : {'1':[], '3':[], '5':[]},
    "joyce" : {'1':[], '3':[], '5':[]}
}


result_dir = "../results"
metric = "lr"
learning_rate = None
base_model_acc = []
lr = None

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
            base_model_acc.append(summary['setting']['original_acc'])
            per_acc[person].append(summary['setting']['personlized_acc'])
            results = summary[optimizer][metric]

            for key in results.keys():
                lr = results[key]['lr']
                original[person][key].append(results[key]['original'])
                personalized[person][key].append(results[key]['personlized'])

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
    processed = [[],[],[],[]]

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            processed[j].append(arr[i][j])

    mean_arr = []
    lower_arr = []
    upper_arr = []

    for data in processed:
        mean, lower, upper = mean_confidence_interval(data)
        mean_arr.append(mean)
        lower_arr.append(lower)
        upper_arr.append(upper)

    return np.array(mean_arr), np.array(lower_arr), np.array(upper_arr)

legends = []
x_ticks = np.arange(1,5)

name_mapping = {
    "BRANDON":"A",
    "JAY":"B",
    "JACK":"C",
    "MAX":"D",
    "LEE":"E",
    "KEVIN":"F",
    "KANG":"D",
    "JOYCE":"H"
}

def plot_mean_and_CI(axis, mean, lb, ub, fmt=None, color_mean=None, color_shading=None, label=None):
    # plot the shaded range of the confidence intervals
    axis.fill_between(x_ticks, ub, lb, color=color_shading, alpha=.2)

    # plot the mean on top
    return axis.plot(x_ticks, mean, fmt, color=color_mean, label=label)

for person, axis in axs_mapping.items():
    processed_original = []
    processed_personalized = []

    original_dict = original[person]
    personalized_dict = personalized[person]

    axis.set_title('{0} - {1:.1f} %'.format("User " + name_mapping[person.upper()], round(per_acc[person] * 100, 1)))

    axis.set(xlabel='learning rate', ylabel='accuracy', xticks=x_ticks, xticklabels=lr)
    axis.set_ylim(0.7, 1.0)
    axis.grid()

    for i in original[person].keys():
        original_mean, original_lower, original_upper = process_data(original_dict[i])
        original_error = [original_lower, original_upper]
        personalized_mean, personalized_lower, personalized_upper = process_data(personalized_dict[i])
        personalized_error = [personalized_lower, personalized_upper]

        original_line = plot_mean_and_CI(axis, original_mean, original_lower, original_upper, fmt='--', color_mean=line_color[int(i)-1], color_shading=line_color[int(i)-1], label='original')

        personalized_line = plot_mean_and_CI(axis, personalized_mean, personalized_lower, personalized_upper, fmt='-', color_mean=line_color[int(i)-1], color_shading=line_color[int(i)-1], label='personalized')

#         original_line = axis.plot(x_ticks, original_mean, '--', color=line_color[int(i)-1], label='original')
#         personalized_line = axis.plot(x_ticks, personalized_mean, '-', color=line_color[int(i)-1], label='personalized')

        legends.append(original_line)
        legends.append(personalized_line)

fig.legend(legends,     # The line objects
           labels=["original_1", "personalized_1", "original_3", "personalized_3", "original_5", "personalized_5"],   # The labels for each line
           loc=8,  # Position of legend
           ncol=3, bbox_to_anchor=(0.5, 0.03)
           )

fig.subplots_adjust(bottom=0.17, top=0.88, wspace=0.4, hspace=0.35, right=0.9, left=0.15)
fig.suptitle('Learning Rate\nbase model accuracy - {0:.1f} %'.format(round(base_model_acc * 100, 1)))

fig.savefig(optimizer+"/learning_rate.png")

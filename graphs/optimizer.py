import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

fig, axs = plt.subplots(5, 1, figsize=[6,24])

axs_mapping = {
    "brandon" : axs[0],
    "jay" : axs[1],
    "jack" : axs[2],
    "max" : axs[3],
    "kevin" : axs[4]
}

per_acc = {}

original = {
    "brandon" : {'1':[], '3':[], '5':[]},
    "jay" : {'1':[], '3':[], '5':[]},
    "jack" : {'1':[], '3':[], '5':[]},
    "max" : {'1':[], '3':[], '5':[]},
    "kevin" : {'1':[], '3':[], '5':[]}
}

personalized = {
    "brandon" : {'1':[], '3':[], '5':[]},
    "jay" : {'1':[], '3':[], '5':[]},
    "jack" : {'1':[], '3':[], '5':[]},
    "max" : {'1':[], '3':[], '5':[]},
    "kevin" : {'1':[], '3':[], '5':[]}
}


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
                original[person][key].append(results[key]['original'])
                personalized[person][key].append(results[key]['personlized'])

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
optimizers = ["RMSprop", "adam", "adagrad", "SGD"]

name_mapping = {
    "BRANDON":"A",
    "JAY":"B",
    "JACK":"C",
    "MAX":"D",
    "KEVIN":"E"
}

acc_mapping = {
    "BRANDON":"89.79",
    "JAY":"88.75",
    "JACK":"80.83",
    "MAX":"78.12",
    "KEVIN":"91.67"
}

width = 0.15

loc = (np.arange(6) * width) - (width * 3)

def plot_mean_and_CI(ind, axis, mean, lb, ub, fmt=None, color_mean=None, color_shading=None, label=None):
    # plot the shaded range of the confidence intervals
    # axis.fill_between(x_ticks, ub, lb, color=color_shading, alpha=.2)

    # plot the mean on top
    # return axis.plot(x_ticks, mean, fmt, color=color_mean, label=label)
    legend = None
    if ind % 2 == 0:
        # legend = axis.bar(x_ticks-0.15, mean, 0.30,  color=color_mean, label=label)
        legend = axis.bar(x_ticks+loc[ind], mean, width, color=color_mean, alpha=.5, label=label)
    else:
        # legend = axis.bar(x_ticks+0.15, mean, 0.35,  color=color_mean, label=label)
        legend = axis.bar(x_ticks+loc[ind], mean, width, color=color_mean, label=label)
    return legend

for person, axis in axs_mapping.items():
    processed_original = []
    processed_personalized = []

    original_dict = original[person]
    personalized_dict = personalized[person]

    axis.set_title('{0} - {1} %'.format("User " + name_mapping[person.upper()], acc_mapping[person.upper()]))

    axis.set(xlabel='optimizer', ylabel='accuracy', xticks=x_ticks, xticklabels=optimizers)
    axis.set_ylim(0.8, 1.0)
    # axis.grid()

    for i in original[person].keys():
        original_mean, original_lower, original_upper = process_data(original_dict[i])
        original_error = [original_lower, original_upper]
        personalized_mean, personalized_lower, personalized_upper = process_data(personalized_dict[i])
        personalized_error = [personalized_lower, personalized_upper]

        original_line = plot_mean_and_CI(int(i)-1, axis, original_mean, original_lower, original_upper, fmt='--', color_mean=line_color[int(i)-1], color_shading=line_color[int(i)-1], label='original')

        personalized_line = plot_mean_and_CI(int(i), axis, personalized_mean, personalized_lower, personalized_upper, fmt='-', color_mean=line_color[int(i)-1], color_shading=line_color[int(i)-1], label='personalized')

#         original_line = axis.plot(x_ticks, original_mean, '--', color=line_color[int(i)-1], label='original')
#         personalized_line = axis.plot(x_ticks, personalized_mean, '-', color=line_color[int(i)-1], label='personalized')

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

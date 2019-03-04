import os
import ast
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=[7,7])

axs_mapping = {
    "brandon" : axs[0][0],
    "jay" : axs[0][1],
    "jack" : axs[1][0],
    "max" : axs[1][1]
}

per_acc = {}

original = {
    "brandon" : {'1':[], '3':[], '5':[]},
    "jay" : {'1':[], '3':[], '5':[]},
    "jack" : {'1':[], '3':[], '5':[]},
    "max" : {'1':[], '3':[], '5':[]}
}

personalized = {
    "brandon" : {'1':[], '3':[], '5':[]},
    "jay" : {'1':[], '3':[], '5':[]},
    "jack" : {'1':[], '3':[], '5':[]},
    "max" : {'1':[], '3':[], '5':[]}
}

result_dir = "../results"
metric = "lr"
learning_rate = None
base_model_acc = None
lr = None

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

            original[person]
            personalized[person]

            for key in results.keys():
                lr = results[key]['lr']
                original[person][key].append(results[key]['original'])
                personalized[person][key].append(results[key]['personlized'])

line_color = ['#fd0000', '#D6B748', '#45A320', '#5ADABE', '#4867D6']

def get_average(arr):
    return np.mean(np.array(arr), axis=0)

legends = []
x_ticks = np.arange(1,5)

for person, axis in axs_mapping.items():
    processed_original = []
    processed_personalized = []

    original_dict = original[person]
    personalized_dict = personalized[person]

    axis.set_title('{0} (per : {1})'.format(person, per_acc[person]))

    axis.set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
    axis.set_ylim(0.7, 1.0)
    axis.grid()

    for i in original[person].keys():
        original_line = axis.plot(x_ticks, get_average(original_dict[i]), '--', color=line_color[int(i)-1], label='original')
        personalized_line = axis.plot(x_ticks, get_average(personalized_dict[i]), '-', color=line_color[int(i)-1], label='personalized')

        legends.append(original_line)
        legends.append(personalized_line)


fig.legend(legends,     # The line objects
           labels=["original_1", "personalized_1", "original_3", "personalized_3", "original_5", "personalized_5"],   # The labels for each line
           loc=8,  # Position of legend
           ncol=3
           )

fig.subplots_adjust(bottom=0.15, wspace=0.28, hspace=0.35)
fig.suptitle('Learning Rate ( epochs = {0} )\nbase model acc : {1}'.format(epochs, base_model_acc))

fig.savefig("learning_rate.png")
plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(2, 2, figsize=[7,7])

fig.suptitle('Learning rate ( n_epochs = 20 )\nbase model acc : 0.91296')
fig.subplots_adjust(hspace=0.35)

x_ticks = np.arange(1,5)
lr = [0.1, 0.01, 0.001, 0.0001]

line_color = ['#fd0000', '#D6B748', '#4867D6', '#5ADABE', '#45A320']

def plot_data(axis, original, personalized):
    legends = []
    for i in range(0,5,4):
        original_line = axis.plot(x_ticks, original[i], '--', color=line_color[i], label='original')
        personalized_line = axis.plot(x_ticks, personalized[i], '-', color=line_color[i], label='personalized')
        legends.append(original_line)
        legends.append(personalized_line)

    return legends

# brandon 

original = []
personalized = []

original.append([0.50081, 0.82267, 0.8951, 0.89542]) # 1
personalized.append([0.6, 0.81667, 0.8375, 0.81667]) # 1

original.append([0.89672, 0.85287, 0.8873, 0.89152]) # 2
personalized.append([0.65833, 0.92917, 0.8625, 0.84583]) # 2

original.append([0.88146, 0.86034, 0.89152, 0.89282]) # 3
personalized.append([0.825, 0.9375, 0.875, 0.83333]) # 3

original.append([0.67749, 0.86424, 0.8925, 0.8938]) # 4
personalized.append([0.875, 0.92917, 0.85833, 0.84167]) # 4

original.append([0.72978, 0.87301, 0.88828, 0.89022]) # 5
personalized.append([0.9, 0.93333, 0.87917, 0.8625]) # 5

axs[0][0].set_title('Brandon (per : 0.70417)')
legends = plot_data(axs[0][0], original, personalized)
axs[0][0].set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
axs[0][0].set_ylim(0.5, 1.0)
axs[0][0].grid()

# jay

original = []
personalized = []

original.append([0.90874, 0.90841, 0.90127, 0.89899])
personalized.append([0.82917, 0.90417, 0.90417, 0.89583])

original.append([0.90776, 0.90679, 0.90419, 0.89964])
personalized.append([0.89583, 0.92083, 0.93333, 0.9125])

original.append([0.8204, 0.89899, 0.90224, 0.90354])
personalized.append([0.91667, 0.9375, 0.92917, 0.925])

original.append([0.90971, 0.90257, 0.89932, 0.89997])
personalized.append([0.91667, 0.94583, 0.92083, 0.92083])

original.append([0.90257, 0.90484, 0.90874, 0.90809])
personalized.append([0.875, 0.93333, 0.92083, 0.9125])


axs[0][1].set_title('Jay (per : 0.89583)')
legends = plot_data(axs[0][1], original, personalized)
axs[0][1].set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
axs[0][1].set_ylim(0.5, 1.0)
axs[0][1].grid()

# jack 

original = []
personalized = []


original.append([0.6619, 0.82299, 0.88113, 0.89347])
personalized.append([0.725, 0.87917, 0.8625, 0.82917])

original.append([0.89445, 0.85385, 0.89445, 0.89445])
personalized.append([0.6375, 0.85417, 0.86667, 0.85417])

original.append([0.89022, 0.88698, 0.88535, 0.89769])
personalized.append([0.82083, 0.8875, 0.87917, 0.84167])

original.append([0.90224, 0.88438, 0.89217, 0.89152])
personalized.append([0.86667, 0.87083, 0.88333, 0.85])

original.append([0.7444, 0.87756, 0.89315, 0.89347])
personalized.append([0.92083, 0.925, 0.88333, 0.84583])


axs[1][0].set_title('Jack (per : 0.84167)')
legends = plot_data(axs[1][0], original, personalized)
axs[1][0].set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
axs[1][0].set_ylim(0.5, 1.0)
axs[1][0].grid()

# max

original = []
personalized = []

original.append([0.79474, 0.85645, 0.8873, 0.89217])
personalized.append([0.6375, 0.89583, 0.89583, 0.86667])

original.append([0.91004, 0.89087, 0.90516, 0.90484])
personalized.append([0.78333, 0.83333, 0.8875, 0.875])

original.append([0.90711, 0.85158, 0.89964, 0.89737])
personalized.append([0.85417, 0.91667, 0.9, 0.8875])

original.append([0.65086, 0.89477, 0.90029, 0.89964])
personalized.append([0.83333, 0.8625, 0.9, 0.87083])

original.append([0.71224, 0.89087, 0.89477, 0.89639])
personalized.append([0.90833, 0.89167, 0.90417, 0.88333])


axs[1][1].set_title('Max (per : 0.79167)')
legends = plot_data(axs[1][1], original, personalized)
axs[1][1].set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
axs[1][1].set_ylim(0.5, 1.0)
axs[1][1].grid()


fig.legend(legends,     # The line objects
           labels=["original_1", "personalized_1", "original_5", "personalized_5"],   # The labels for each line
           loc=8,  # Position of legend
           ncol=2
           )

fig.subplots_adjust(bottom=0.15, wspace=0.25)
fig.savefig("learning_rate.png")
plt.show()
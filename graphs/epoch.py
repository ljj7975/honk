import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(2, 2, figsize=[7,7])

fig.suptitle('Epochs ( learning rate = 0.01 )\nbase model acc : 0.91296')
fig.subplots_adjust(hspace=0.35)

epochs = np.arange(0, 55, 5)

line_color = ['#fd0000', '#D6B748', '#4867D6', '#5ADABE', '#45A320']

def plot_data(axis, original, personalized):
    legends = []
    for i in range(0,5,2):
        original_line = axis.plot(epochs, original[i], '--', color=line_color[i], label='original')
        personalized_line = axis.plot(epochs, personalized[i], '-', color=line_color[i], label='personalized')
        legends.append(original_line)
        legends.append(personalized_line)

    return legends

# brandon 

original = []
personalized = []


original.append([0.91296, 0.91361, 0.88892, 0.86002, 0.82267, 0.82202, 0.82267, 0.82202, 0.71095, 0.72881, 0.73011])
personalized.append([0.70417, 0.72917, 0.72083, 0.80833, 0.81667, 0.77917, 0.775, 0.80833, 0.80833, 0.84167, 0.84583])

original.append([0.91296, 0.90419, 0.88081, 0.85872, 0.85287, 0.83144, 0.85287, 0.83144, 0.7824, 0.7824, 0.77882])
personalized.append([0.70417, 0.83333, 0.9125, 0.9375, 0.92917, 0.9, 0.9, 0.92083, 0.93333, 0.925, 0.93333])

original.append([0.91296, 0.90322, 0.886, 0.86814, 0.86034, 0.8584, 0.8584, 0.8584, 0.8584, 0.8584, 0.8584])
personalized.append([0.70417, 0.83333, 0.89583, 0.90833, 0.9375, 0.94583, 0.9375, 0.9375, 0.93333, 0.9375, 0.94167])

original.append([0.91296, 0.90581, 0.89055, 0.87821, 0.86424, 0.86002, 0.84638, 0.84703, 0.8493, 0.84735, 0.84768])
personalized.append([0.70417, 0.80833, 0.87083, 0.90417, 0.92917, 0.9375, 0.92917, 0.925, 0.91667, 0.93333, 0.94167])

original.append([0.91296, 0.90809, 0.89087, 0.8821, 0.87301, 0.87301, 0.87301, 0.86587, 0.86781, 0.86781, 0.86781])
personalized.append([0.70417, 0.81667, 0.8625, 0.89583, 0.93333, 0.92917, 0.94583, 0.94583, 0.93333, 0.92917, 0.92917])



axs[0][0].set_title('Brandon (per : 0.70417)')
legends = plot_data(axs[0][0], original, personalized)
axs[0][0].set(xlabel='Epochs', ylabel='Accuracy', xticks=epochs[::2])
axs[0][0].set_ylim(0.7, 1.0)
axs[0][0].grid()

# jay

original = []
personalized = []

original.append([0.91296, 0.90841, 0.90841, 0.90841, 0.90841, 0.90841, 0.81617, 0.81455, 0.80676, 0.80741, 0.82234])
personalized.append([0.89583, 0.9125, 0.925, 0.89583, 0.90417, 0.925, 0.93333, 0.9375, 0.94583, 0.95, 0.95417])

original.append([0.91296, 0.90679, 0.90679, 0.90679, 0.90679, 0.90679, 0.90679, 0.84638, 0.84313, 0.82722, 0.82722])
personalized.append([0.89583, 0.92083, 0.9125, 0.91667, 0.92083, 0.92917, 0.92917, 0.9375, 0.9375, 0.95417, 0.96667])

original.append([0.91296, 0.90581, 0.89899, 0.89899, 0.89899, 0.89899, 0.85645, 0.8454, 0.84151, 0.84151, 0.83988])
personalized.append([0.89583, 0.925, 0.925, 0.925, 0.9375, 0.93333, 0.9375, 0.94583, 0.94167, 0.9375, 0.93333])

original.append([0.91296, 0.91004, 0.90257, 0.90257, 0.90257, 0.90257, 0.90257, 0.90257, 0.90257, 0.90257, 0.85612])
personalized.append([0.89583, 0.90833, 0.93333, 0.94583, 0.94583, 0.93333, 0.92083, 0.91667, 0.90833, 0.9125, 0.91667])

original.append([0.91296, 0.90906, 0.90484, 0.90484, 0.90484, 0.90484, 0.90484, 0.90484, 0.90484, 0.85677, 0.8558])
personalized.append([0.89583, 0.90417, 0.92917, 0.92917, 0.93333, 0.9375, 0.93333, 0.9375, 0.92917, 0.92917, 0.925])



axs[0][1].set_title('Jay (per : 0.89583)')
legends = plot_data(axs[0][1], original, personalized)
axs[0][1].set(xlabel='Epochs', ylabel='Accuracy', xticks=epochs[::2])
axs[0][1].set_ylim(0.7, 1.0)
axs[0][1].grid()

# jack 

original = []
personalized = []


original.append([0.91296, 0.90971, 0.90971, 0.82494, 0.82299, 0.82462, 0.82462, 0.81747, 0.81747, 0.80058, 0.79896])
personalized.append([0.84167, 0.85, 0.825, 0.84583, 0.87917, 0.89167, 0.8875, 0.89583, 0.90833, 0.8875, 0.9])

original.append([0.91296, 0.90322, 0.87983, 0.86359, 0.85385, 0.85385, 0.85385, 0.85385, 0.85385, 0.85385, 0.85385])
personalized.append([0.84167, 0.8875, 0.875, 0.85, 0.85417, 0.87083, 0.85, 0.8625, 0.85833, 0.88333, 0.87917])

original.append([0.91296, 0.90322, 0.8873, 0.8873, 0.8873, 0.85677, 0.85677, 0.85677, 0.85677, 0.81812, 0.81812])
personalized.append([0.84167, 0.88333, 0.89583, 0.87917, 0.8875, 0.91667, 0.9125, 0.89583, 0.88333, 0.89583, 0.88333])

original.append([0.91296, 0.91036, 0.88438, 0.88438, 0.88438, 0.84768, 0.82559, 0.82559, 0.82559, 0.82559, 0.82559])
personalized.append([0.84167, 0.89167, 0.87917, 0.85, 0.87083, 0.9, 0.90417, 0.90417, 0.90833, 0.9, 0.91667])

original.append([0.91296, 0.90809, 0.89445, 0.88243, 0.87756, 0.87756, 0.87756, 0.8532, 0.8532, 0.85255, 0.8532])
personalized.append([0.84167, 0.87917, 0.8875, 0.925, 0.925, 0.94167, 0.95, 0.94583, 0.95, 0.92917, 0.91667])


axs[1][0].set_title('Jack (per : 0.84167)')
legends = plot_data(axs[1][0], original, personalized)
axs[1][0].set(xlabel='Epochs', ylabel='Accuracy', xticks=epochs[::2])
axs[1][0].set_ylim(0.7, 1.0)
axs[1][0].grid()

# max

original = []
personalized = []

original.append([0.91296, 0.90906, 0.88892, 0.86197, 0.85645, 0.84703, 0.83631, 0.83404, 0.83404, 0.83176, 0.82657])
personalized.append([0.79167, 0.79167, 0.89167, 0.9, 0.89583, 0.9125, 0.9125, 0.925, 0.93333, 0.94167, 0.93333])

original.append([0.91296, 0.90386, 0.90029, 0.89087, 0.89087, 0.89087, 0.89087, 0.84865, 0.84216, 0.83988, 0.83436])
personalized.append([0.79167, 0.90417, 0.9125, 0.90833, 0.83333, 0.84167, 0.88333, 0.92917, 0.94167, 0.95417, 0.95833])

original.append([0.91296, 0.90679, 0.89672, 0.86554, 0.85158, 0.85028, 0.84833, 0.84411, 0.84411, 0.84475, 0.84411])
personalized.append([0.79167, 0.89167, 0.9125, 0.90833, 0.91667, 0.95, 0.95833, 0.9625, 0.95833, 0.95833, 0.95833])

original.append([0.91296, 0.90971, 0.89575, 0.89477, 0.89477, 0.82754, 0.82624, 0.8165, 0.8165, 0.8165, 0.8165])
personalized.append([0.79167, 0.8625, 0.90417, 0.88333, 0.8625, 0.89167, 0.92917, 0.94167, 0.94167, 0.94583, 0.93333])

original.append([0.91296, 0.91231, 0.90224, 0.89087, 0.89087, 0.85482, 0.83566, 0.82754, 0.8204, 0.82007, 0.83014])
personalized.append([0.79167, 0.84583, 0.8875, 0.89167, 0.89167, 0.90833, 0.94167, 0.95, 0.95, 0.95, 0.95417])



axs[1][1].set_title('Max (per : 0.79167)')
legends = plot_data(axs[1][1], original, personalized)
axs[1][1].set(xlabel='Epochs', ylabel='Accuracy', xticks=epochs[::2])
axs[1][1].set_ylim(0.7, 1.0)
axs[1][1].grid()


fig.legend(legends,     # The line objects
           labels=["original_1", "personalized_1", "original_3", "personalized_3", "original_5", "personalized_5"],   # The labels for each line
           loc=8,  # Position of legend
           ncol=3
           )

fig.subplots_adjust(bottom=0.15, wspace=0.25)
fig.savefig("epoch.png")
plt.show()
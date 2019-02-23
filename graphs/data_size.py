import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(2, 2, figsize=[7,7])

fig.suptitle('Size of audio per keyword\n( learning rate = 0.01, n_epochs = 20 )')
fig.subplots_adjust(hspace=0.35)

datasize = np.arange(0, 11)

# brandon 

original = [0.9204, 0.9219, 0.9208, 0.9202, 0.923, 0.9235, 0.8761, 0.893, 0.9076, 0.9083, 0.8818]
personalized = [0.7083, 0.7083, 0.7083, 0.7083, 0.7083, 0.7083, 0.9583, 0.9417, 0.925, 0.8917, 0.9667]

axs[0][0].set_title('Brandon')
original_line = axs[0][0].plot(datasize, original, '--', label='original')
personalized_line = axs[0][0].plot(datasize, personalized, '-', label='personalized')
axs[0][0].set(xlabel='Size per keyword', ylabel='Accuracy', xticks=datasize)
axs[0][0].set_ylim(0.7, 1.0)
axs[0][0].grid()

# jay

original = [0.9204, 0.9219, 0.9208, 0.9202, 0.923, 0.9235, 0.8943, 0.9018, 0.8994, 0.888, 0.9008]
personalized = [0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.9417, 0.925, 0.95, 0.9167, 0.9333]

axs[0][1].set_title('Jay')
original_line = axs[0][1].plot(datasize, original, '--', label='original')
personalized_line = axs[0][1].plot(datasize, personalized, '-', label='personalized')
axs[0][1].set(xlabel='Size per keyword', ylabel='Accuracy', xticks=datasize)
axs[0][1].set_ylim(0.7, 1.0)
axs[0][1].grid()


# jack

original = [0.9204, 0.9219, 0.9208, 0.9202, 0.923, 0.9235, 0.9157, 0.8992, 0.9018, 0.895, 0.8994]
personalized = [0.7667, 0.7667, 0.7667, 0.7667, 0.7667, 0.7667, 0.8417, 0.8667, 0.8583, 0.85, 0.7917]

axs[1][0].set_title('Jack')
original_line = axs[1][0].plot(datasize, original, '--', label='original')
personalized_line = axs[1][0].plot(datasize, personalized, '-', label='personalized')
axs[1][0].set(xlabel='Size per keyword', ylabel='Accuracy', xticks=datasize)
axs[1][0].set_ylim(0.7, 1.0)
axs[1][0].grid()


# max

original = [0.9204, 0.9219, 0.9208, 0.9202, 0.923, 0.9235, 0.9056, 0.9019, 0.8921, 0.8997, 0.9018]
personalized = [0.7833, 0.7833, 0.7833, 0.7833, 0.7833, 0.7833, 0.9333, 0.9417, 0.925, 0.9417, 0.95]

axs[1][1].set_title('Max')
original_line = axs[1][1].plot(datasize, original, '--', label='original')
personalized_line = axs[1][1].plot(datasize, personalized, '-', label='personalized')
axs[1][1].set(xlabel='Size per keyword', ylabel='Accuracy', xticks=datasize)
axs[1][1].set_ylim(0.7, 1.0)
axs[1][1].grid()


fig.legend([original_line, personalized_line],     # The line objects
           labels=["original", "personalized"],   # The labels for each line
           loc=8,  # Position of legend
           ncol=2
           )
fig.savefig("data_size.png")
# plt.show()
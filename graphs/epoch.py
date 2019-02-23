import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(2, 2, figsize=[7,7])

fig.suptitle('Epochs\n( size_per_word : 6, learning rate = 0.01 )')
fig.subplots_adjust(hspace=0.35)

epochs = np.arange(0, 50, 5)

# brandon 

original = [0.9204, 0.9188, 0.9023, 0.8946, 0.8825, 0.861, 0.8977, 0.8595, 0.8555, 0.8752]
personalized = [0.7083, 0.8417, 0.9083, 0.9667, 0.9583, 0.9583, 0.9417, 0.975, 0.95, 0.95]

axs[0][0].set_title('Brandon')
original_line = axs[0][0].plot(epochs, original, '--', label='original')
personalized_line = axs[0][0].plot(epochs, personalized, '-', label='personalized')
axs[0][0].set(xlabel='Size per keyword', ylabel='Accuracy', xticks=epochs)
axs[0][0].set_ylim(0.7, 1.0)
axs[0][0].grid()

# jay

original = [0.9204, 0.9201, 0.906, 0.8965, 0.8825, 0.8911, 0.8765, 0.8729, 0.8866, 0.8624]
personalized = [0.875, 0.8833, 0.9417, 0.9417, 0.95, 0.9333, 0.9417, 0.9583, 0.95, 0.9583]

axs[0][1].set_title('Jay')
original_line = axs[0][1].plot(epochs, original, '--', label='original')
personalized_line = axs[0][1].plot(epochs, personalized, '-', label='personalized')
axs[0][1].set(xlabel='Size per keyword', ylabel='Accuracy', xticks=epochs)
axs[0][1].set_ylim(0.7, 1.0)
axs[0][1].grid()


# jack 

original = [0.9204, 0.9191, 0.914, 0.9089, 0.8911, 0.8879, 0.8889, 0.884, 0.8771, 0.8851]
personalized = [0.7667, 0.7917, 0.85, 0.8417, 0.8667, 0.825, 0.875, 0.8417, 0.9, 0.8833]

axs[1][0].set_title('Jack')
original_line = axs[1][0].plot(epochs, original, '--', label='original')
personalized_line = axs[1][0].plot(epochs, personalized, '-', label='personalized')
axs[1][0].set(xlabel='Size per keyword', ylabel='Accuracy', xticks=epochs)
axs[1][0].set_ylim(0.7, 1.0)
axs[1][0].grid()

# max

original = [0.9204, 0.9219, 0.9151, 0.9127, 0.9069, 0.8968, 0.8838, 0.8838, 0.8835, 0.8869]
personalized = [0.7833, 0.8417, 0.8917, 0.9167, 0.95, 0.9333, 0.95, 0.9583, 0.95, 0.9417]

axs[1][1].set_title('Max')
original_line = axs[1][1].plot(epochs, original, '--', label='original')
personalized_line = axs[1][1].plot(epochs, personalized, '-', label='personalized')
axs[1][1].set(xlabel='Size per keyword', ylabel='Accuracy', xticks=epochs)
axs[1][1].set_ylim(0.7, 1.0)
axs[1][1].grid()

fig.legend([original_line, personalized_line],     # The line objects
           labels=["original", "personalized"],   # The labels for each line
           loc=8,  # Position of legend
           ncol=2
           )

fig.savefig("epoch.png")
# plt.show()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(1, 1, figsize=[7,7])

fig.suptitle('Epochs\n( size_per_word : 6, learning rate = 0.01 )')
fig.subplots_adjust(hspace=0.35)

epochs = np.arange(0, 105, 5)

# jack 

original = [0.9204, 0.918, 0.9098, 0.9012, 0.9034, 0.8911, 0.891, 0.8849, 0.8904, 0.875, 0.8873, 0.8782, 0.8813, 0.8668, 0.8791, 0.8685, 0.8635, 0.8421, 0.8796, 0.8644, 0.8816]
personalized = [0.7667, 0.8167, 0.8583, 0.8583, 0.8583, 0.8417, 0.825, 0.8833, 0.85, 0.8917, 0.8583, 0.8667, 0.9, 0.9083, 0.8917, 0.875, 0.9083, 0.8917, 0.8583, 0.9083, 0.9083]

axs.set_title('Jack')
original_line = axs.plot(epochs, original, '--', label='original')
personalized_line = axs.plot(epochs, personalized, '-', label='personalized')
axs.set(xlabel='Size per keyword', ylabel='Accuracy', xticks=epochs[::2])
axs.set_ylim(0.7, 1.0)
axs.grid()


fig.legend([original_line, personalized_line],     # The line objects
           labels=["original", "personalized"],   # The labels for each line
           loc=8,  # Position of legend
           ncol=2
           )

# fig.savefig("epoch.png")
plt.show()
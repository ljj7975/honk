import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[7,7])
epochs = np.arange(0, 50, 5)

# brandon 

original = [0.9204, 0.9188, 0.9023, 0.8946, 0.8825, 0.861, 0.8977, 0.8595, 0.8555, 0.8752]
personalized = [0.7083, 0.8417, 0.9083, 0.9667, 0.9583, 0.9583, 0.9417, 0.975, 0.95, 0.95]

ax1.plot(epochs, original, '--', label='original')
ax1.plot(epochs, personalized, '-', label='personalized')
ax1.set(xlabel='Size per keyword', ylabel='Accuracy', xticks=epochs)
ax1.set_ylim(0.7, 1.0)
ax1.grid()

# Shrink current axis's height by 10% on the bottom
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                  box.width, box.height * 0.8])

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
           fancybox=True, shadow=True, ncol=4, prop={'size': 10})


# jay

original = [0.9204, 0.9201, 0.906, 0.8965, 0.8825, 0.8911, 0.8765, 0.8729, 0.8866, 0.8624]
personalized = [0.875, 0.8833, 0.9417, 0.9417, 0.95, 0.9333, 0.9417, 0.9583, 0.95, 0.9583]

ax2.plot(epochs, original, '--', label='original')
ax2.plot(epochs, personalized, '-', label='personalized')
ax2.set(xlabel='Size per keyword', ylabel='Accuracy', xticks=epochs)
ax2.set_ylim(0.7, 1.0)
ax2.grid()

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.2,
                  box.width, box.height * 0.8])

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
           fancybox=True, shadow=True, ncol=4, prop={'size': 10})

# fig.savefig("data_size.png")
plt.show()
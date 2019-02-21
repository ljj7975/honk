import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[7,7])
datasize = np.arange(0, 11)

# brandon 

original = [0.9204, 0.9219, 0.9208, 0.9202, 0.923, 0.9235, 0.8761, 0.893, 0.9076, 0.9083, 0.8818]
personalized = [0.7083, 0.7083, 0.7083, 0.7083, 0.7083, 0.7083, 0.9583, 0.9417, 0.925, 0.8917, 0.9667]

ax1.plot(datasize, original, '--', label='original')
ax1.plot(datasize, personalized, '-', label='personalized')
ax1.set(xlabel='Size per keyword', ylabel='Accuracy', xticks=datasize)
ax1.set_ylim(0.7, 1.0)
ax1.grid()

# Shrink current axis's height by 10% on the bottom
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                  box.width, box.height * 0.8])

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
           fancybox=True, shadow=True, ncol=4, prop={'size': 10})


# jay

original = [0.9204, 0.9219, 0.9208, 0.9202, 0.923, 0.9235, 0.8943, 0.9018, 0.8994, 0.888, 0.9008]
personalized = [0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.9417, 0.925, 0.95, 0.9167, 0.9333]

ax2.plot(datasize, original, '--', label='original')
ax2.plot(datasize, personalized, '-', label='personalized')
ax2.set(xlabel='Size per keyword', ylabel='Accuracy', xticks=datasize)
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
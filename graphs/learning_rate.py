import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[7,7])
x_ticks = np.arange(1,5)
lr = [0.1, 0.01, 0.001, 0.0001]

# brandon 

original = [0.6906, 0.8644, 0.9069, 0.9061]
personalized = [0.925, 0.9667, 0.8833, 0.85]

ax1.plot(x_ticks, original, '--', label='original')
ax1.plot(x_ticks, personalized, '-', label='personalized')
ax1.set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
ax1.set_ylim(0.7, 1.0)
ax1.grid()

# Shrink current axis's height by 10% on the bottom
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                  box.width, box.height * 0.8])

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
           fancybox=True, shadow=True, ncol=4, prop={'size': 10})


# jay

original = [0.8608, 0.8996, 0.9107, 0.9124]
personalized = [0.925, 0.9583, 0.925, 0.9]

ax2.plot(x_ticks, original, '--', label='original')
ax2.plot(x_ticks, personalized, '-', label='personalized')
ax2.set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
ax2.set_ylim(0.7, 1.0)
ax2.grid()

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.2,
                  box.width, box.height * 0.8])

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
           fancybox=True, shadow=True, ncol=4, prop={'size': 10})

# fig.savefig("learning_rate.png")
plt.show()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(2, 2, figsize=[7,7])

fig.suptitle('Learning rate\n( size_per_word : 6, n_epochs = 20 )')
fig.subplots_adjust(hspace=0.35)

x_ticks = np.arange(1,5)
lr = [0.1, 0.01, 0.001, 0.0001]

# brandon 

original = [0.6906, 0.8644, 0.9069, 0.9061]
personalized = [0.925, 0.9667, 0.8833, 0.85]

axs[0][0].set_title('Brandon')
original_line = axs[0][0].plot(x_ticks, original, '--', label='original')
personalized_line = axs[0][0].plot(x_ticks, personalized, '-', label='personalized')
axs[0][0].set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
axs[0][0].set_ylim(0.7, 1.0)
axs[0][0].grid()


# jay

original = [0.8608, 0.8996, 0.9107, 0.9124]
personalized = [0.925, 0.9583, 0.925, 0.9]

axs[0][1].set_title('Jay')
original_line = axs[0][1].plot(x_ticks, original, '--', label='original')
personalized_line = axs[0][1].plot(x_ticks, personalized, '-', label='personalized')
axs[0][1].set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
axs[0][1].set_ylim(0.7, 1.0)
axs[0][1].grid()

# jack 

original = [0.7742, 0.9032, 0.9023, 0.9107]
personalized = [0.85, 0.8583, 0.8083, 0.7833]

axs[1][0].set_title('Jack')
original_line = axs[1][0].plot(x_ticks, original, '--', label='original')
personalized_line = axs[1][0].plot(x_ticks, personalized, '-', label='personalized')
axs[1][0].set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
axs[1][0].set_ylim(0.7, 1.0)
axs[1][0].grid()

# max

original = [0.7753, 0.9118, 0.9083, 0.9116]
personalized = [0.95, 0.9333, 0.9167, 0.875]

axs[1][1].set_title('Max')
original_line = axs[1][1].plot(x_ticks, original, '--', label='original')
personalized_line = axs[1][1].plot(x_ticks, personalized, '-', label='personalized')
axs[1][1].set(xlabel='Learning rate', ylabel='Accuracy', xticks=x_ticks, xticklabels=lr)
axs[1][1].set_ylim(0.7, 1.0)
axs[1][1].grid()


fig.legend([original_line, personalized_line],     # The line objects
           labels=["original", "personalized"],   # The labels for each line
           loc=8,  # Position of legend
           ncol=2
           )
fig.savefig("learning_rate.png")
# plt.show()
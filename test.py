from matplotlib import pyplot as plt


fig, axs = plt.subplots(2, 2)
fig.suptitle('Vertically stacked subplots')
axs[0, 0].plot(range(10), range(10))
axs[1, 0].plot(range(10), range(10))
plt.show()
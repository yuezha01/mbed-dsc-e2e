import numpy as np
from matplotlib import pyplot as plt
    
def plot_mnist(images, labels):
    # plot mnist images
    # images [num_images, 28, 28]
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_xlabel(np.matmul(labels[i], np.arange(10)))
    plt.show()

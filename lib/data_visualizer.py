import matplotlib.pyplot as plt
import numpy as np


class DataVisualizer():

    def __init__(self, y, title):
        self.y = y
        self.x = np.random.rand(len(y))
        self.title = title

    def scatter(self):
        plt.scatter(self.x, self.y)
        plt.title(self.title)
        plt.xlabel("Index")
        plt.ylabel(self.title)
        plt.grid(True)
        plt.show()

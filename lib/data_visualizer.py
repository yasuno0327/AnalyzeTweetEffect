import matplotlib.pyplot as plt
import numpy as np


class DataVisualizer():

    def __init__(self, x, y, title):
        self.y = y
        self.x = x
        self.title = title

    def scatter(self):
        plt.scatter(self.x, self.y)
        plt.title(self.title)
        plt.xlabel("Index")
        plt.ylabel(self.title)
        plt.grid(True)
        plt.show()

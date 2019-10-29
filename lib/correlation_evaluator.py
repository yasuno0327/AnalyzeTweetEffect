import pandas as pd
import numpy as np


class CorrelationEvaluator():
    def __init__(self, l1, l2):
        self.s1 = pd.Series(l1)
        self.s2 = pd.Series(l2)

    def evaluate(self):
        return self.s1.corr(self.s2)

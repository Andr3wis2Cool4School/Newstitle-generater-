import numpy as np
import pandas as pd



class DataLoder():
    def __init__(self, datapath, x_col, y_col):
        self.df = pd.read_csv(datapath)
        self.X = self.df[x_col]
        self.Y = self.df[y_col]
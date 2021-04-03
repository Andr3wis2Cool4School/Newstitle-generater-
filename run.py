import numpy as np
import pandas as pd 
from core.utils.dataprocer import DataLoader


def main():
    data = DataLoader('./data/fake_or_real_news.csv', 'text', 'title')
    X, Y = data.get_X_and_Y()
    print('DATA LOADING.....')
    print(X.head())
    print(Y.head())

    config = data.fit_text()
    # print(config)

    

if __name__ == '__main__':
    main()
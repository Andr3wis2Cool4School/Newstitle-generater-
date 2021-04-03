import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from core.utils.dataprocer import DataLoader
from core.model.seq2seq import Seq2SeqSummarizer

LOAD_EXISTING_WEIGHTS = False


def main():
    np.random.seed(42)
    data = DataLoader('./data/fake_or_real_news.csv', 'text', 'title')
    X, Y = data.get_X_and_Y()
    print('DATA LOADING.....')
    print(X.head())
    print(Y.head())

    config = data.fit_text()
    # print(config)
    summarizer = Seq2SeqSummarizer(config)

    if LOAD_EXISTING_WEIGHTS:
        summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    


    

    

if __name__ == '__main__':
    main()
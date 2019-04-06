# -*- coding: utf-8 -*-
import os
import dataset
import pandas as pd


class Database:
    """ This class facilitates the manipulation of the database. """

    def __init__(self, path):
        """ Constructor.

            Args:
                path the path to the directory of the database
        """
        self.directory = path

    def load(self):
        """ Loads the content of the database in memory. """
        train_dataframe = pd.read_csv(os.path.join(path, 'raw/train.csv'))
        test_dataframe = pd.read_csv(os.path.join(path, 'raw/test.csv'))

        self.train_dataset = Dataset(test_dataframe.drop('species', axis=1), test_dataframe['species'])
        self.test_dataset = Dataset(test_dataframe.values)

    def get_train_dataset():
        """ Returns the training dataset. """
        return self.train_dataset

    def get_test_dataset():
        """ Returns the test dataset. """
        return self.test_dataset

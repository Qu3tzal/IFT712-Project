# -*- coding: utf-8 -*-
import os
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
        train_dataframe = pd.read_csv(os.path.join(self.directory, 'raw/train.csv'))
        test_dataframe = pd.read_csv(os.path.join(self.directory, 'raw/test.csv'))

        self.train_dataset = train_dataframe
        self.test_dataset = test_dataframe

    def get_train_dataset(self):
        """ Returns the training dataset. """
        return self.train_dataset

    def get_test_dataset(self):
        """ Returns the test dataset. """
        return self.test_dataset

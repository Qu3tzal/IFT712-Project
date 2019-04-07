# -*- coding: utf-8 -*-
import pandas as pd


class Preparator:
    """ This class prepares a Pandas's DataFrame to be used. """

    def __init__(self):
        """ Constructor. """
        pass

    def prepare(self, dataset, columns):
        """ Prepares the dataset.

            Arg:
                dataset the dataset to prepare
                columns a list of the name of the columns to prepare
        """
        raise NotImplementedError()

class NullPreparator(Preparator):
    """ A preparator that does nothing. """

    def prepare(self, dataset, columns):
        pass

class StandardizerPreparator(Preparator):
    """ A preparator that centers and divide all columns by their respective standard deviation. """

    def prepare(self, dataset, columns):
        """ Prepares the dataset.

            Arg:
                dataset the dataset to prepare
                columns a list of the name of the columns to prepare
        """
        for column_name in columns:
            dataset[column_name] = (dataset[column_name] - dataset[column_name].mean()) / dataset[column_name].std()

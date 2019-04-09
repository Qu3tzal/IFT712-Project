# -*- coding: utf-8 -*-
import pandas as pd
import sklearn.decomposition
import sklearn.utils

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

class Name2IntPreparator(Preparator):
    """ A preparator that converts name to integers. """

    def prepare(self, dataset, columns):
        """ Prepares the dataset.

            Arg:
                dataset the dataset to prepare
                columns a list of the name of the columns to prepare
        """
        for column_name in columns:
            values_dict = dict()
            last_number_used = -1

            for i, value in enumerate(dataset[column_name]):
                # Force the creation of the entry in the dictionnary
                numeric_replacement = 0

                # Check if the dictionnary has the value already.
                if value in values_dict:
                    numeric_replacement = values_dict[value]
                else:
                    # Increment
                    last_number_used += 1
                    # Set as the numeric replacement for the current string.
                    numeric_replacement = last_number_used
                    # Register in the dictionnary.
                    values_dict[value] = numeric_replacement

                # Replace the string by its numeric value.
                dataset.loc[i, column_name] = numeric_replacement

            # Force conversion to int.
            dataset[column_name] = dataset[column_name].astype(int)

class PCAPreparator(Preparator):
    """ A preparator that performs a PCA.

        Keeps the same number of columns and does not rename the columns.
        The name of the columns after the preparation therefore lose their meaning.
    """

    def prepare(self, dataset, columns):
        """ Prepares the dataset.

            Arg:
                dataset the dataset to prepare
                columns a list of the name of the columns to prepare
        """
        values = dataset[columns]

        PCA = sklearn.decomposition.PCA(n_components=len(columns)) # We keep all the columns.
        PCA.fit_transform(values)

class ShufflePreparator(Preparator):
    """ A preparator that shuffles the dataset. """

    def prepare(self, dataset, columns):
        """ Prepares the dataset.

            Arg:
                dataset the dataset to prepare
                columns a list of the name of the columns to prepare
        """
        sklearn.utils.shuffle(dataset[columns])

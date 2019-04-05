# -*- coding: utf-8 -*-
import random
import numpy as np


class Dataset:
    """ This class facilitates the manipulation of the dataset. """

    def __init__(self, inputs, targets=None):
        """ Constructor.

            Args:
                inputs the list of the inputs
                targets the list of the targets
        """
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(targets)

    def head(self, k):
        """ Returns the k first elements.

            If the dataset has targets returns the k first inputs and targets,
            otherwise returns the k first inputs.

            Arg:
                k the number of elements to return from the start
        """
        if not self.targets is None:
            return (self.inputs[:k], self.targets[:k])
        else:
            return self.input[:k]

    def tail(self, k):
        """ Returns the k last elements.

            If the dataset has targets returns the k last inputs and targets,
            otherwise returns the k last inputs.

            Arg:
                k the number of elements to return from the end
        """
        if not self.targets is None:
            return (self.inputs[-k:], self.targets[-k:])
        else:
            return self.input[-k:]

    def split(self, index):
        """ Splits the dataset in two parts.

            If the index is an integer then splits the dataset on the index-th
            element. If the index is a float then splits in two parts with the
            first part being of size index% of the total dataset.

            Arg:
                index the index where to cut the dataset
        """
        # Convert the index in an integer if necessary.
        if isinstance(index, (float, np.floating)):
            index = int(self.inputs.shape[0] * index)

        return self.head(index), self.tail(self.inputs.shape[0] - index)

    def shuffle(self):
        """ Shuffles the dataset. """
        if not self.targets is None:
            # To keep the inputs with their associated targets during the shuffle
            # we shuffle tuples.
            data = list(zip(self.inputs, self.targets))
            random.shuffle(data)
            self.inputs, self.targets = zip(*data)

            # Convert back to numpy arrays.
            self.inputs = np.asarray(self.inputs)
            self.targets = np.asarray(self.targets)
        else:
            random.shuffle(self.inputs)

    def __getitem__(self, key):
        """ Returns the key-th element of the dataset.

            If the dataset has targets, returns the key-th input with its associated
            target, otherwise returns the key-th input.

            Arg:
                key the index of the element to set
        """
        if not self.targets is None:
            return (self.inputs[key], self.targets[key])
        else:
            return self.inputs[key]

# -*- coding: utf-8 -*-


class Classifier:
    """ This class is the abstract version of a classifier.
        All classifiers in this project should inherit this class to offer a
        uniform API.
    """

    def __init__(self, name):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        self.parameters = dict({"name": name})

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        raise NotImplementedError()

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        raise NotImplementedError()

    def score(self, inputs, targets):
        """ Computes the accuracy and loss on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy and loss
        """
        raise NotImplementedError()

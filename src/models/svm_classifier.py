# -*- coding: utf-8 -*-
import models.classifier
from sklearn import svm


class SVMClassifier(models.classifier.Classifier):
    """ This class is the abstract version of a classifier.
        All classifiers in this project should inherit this class to offer a
        uniform API.
    """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("SVM")
        self.svm = svm.SVC(gamma='auto')

    def get_underlying_classifier(self):
        """ Returns the underlying classifier object. """
        return self.svm

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        self.svm.fit(inputs, targets)

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        return self.svm.predict(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.svm.score(inputs, targets)

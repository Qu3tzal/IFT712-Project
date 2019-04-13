# -*- coding: utf-8 -*-
import models.classifier
from sklearn import svm


class SVMClassifier(models.classifier.Classifier):
    """ This class implements a SVM classifier. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("SVM")
        self.svm = svm.SVC(gamma='auto', probability=True)

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

    def predict_proba(self, dataset):
        """ Predicts the probabilities of each class for the dataset inputs.

            Arg:
                dataset the inputs to predict

            Returns: the probabilities of all inputs
        """
        return self.svm.predict_proba(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.svm.score(inputs, targets)

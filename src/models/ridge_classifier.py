# -*- coding: utf-8 -*-
import models.classifier
from sklearn.linear_model import RidgeClassifier as ridge

class RidgeClassifier(models.classifier.Classifier):
    """ This class implements the ridge classifier. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("ridge")
        self.ridge = ridge()

    def get_underlying_classifier(self):
        """ Returns the underlying classifier object. """
        return self.ridge

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        self.ridge.fit(inputs, targets)

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
        return self.ridge.score(inputs, targets)

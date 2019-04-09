# -*- coding: utf-8 -*-
import models.classifier
from sklearn import ensemble

class RandomForestClassifier(models.classifier.Classifier):
    """ This class implements a random forest classifier. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("RandomForest")
        self.random_forest = ensemble.RandomForestClassifier(n_estimators=100)

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        self.random_forest.fit(inputs, targets)

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        return self.random_forest.predict(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.random_forest.score(inputs, targets)

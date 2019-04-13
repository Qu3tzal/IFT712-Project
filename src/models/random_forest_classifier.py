# -*- coding: utf-8 -*-
import models.classifier
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


class RandomForestClassifier(models.classifier.Classifier):
    """ This class implements a random forest classifier. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("RandomForest")
        self.random_forest = GridSearchCV(ensemble.RandomForestClassifier(n_estimators=100), param_grid={'n_estimators': [45, 99, 100, 99 * 2, 200]}, cv=3, iid=False)

    def get_underlying_classifier(self):
        """ Returns the underlying classifier object. """
        return self.random_forest

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

    def predict_proba(self, dataset):
        """ Predicts the probabilities of each class for the dataset inputs.

            Arg:
                dataset the inputs to predict

            Returns: the probabilities of all inputs
        """
        return self.random_forest.predict_proba(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.random_forest.score(inputs, targets), metrics.log_loss(targets, self.predict_proba(inputs), labels=[str(x) for x in range(0,99)])

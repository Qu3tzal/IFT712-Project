# -*- coding: utf-8 -*-
import models.classifier
from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier(models.classifier.Classifier):
    """ This class implements the logistic regression classifier. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("logistic_regression")
        self.logistic_regression = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000)

    def get_underlying_classifier(self):
        """ Returns the underlying classifier object. """
        return self.logistic_regression

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        self.logistic_regression.fit(inputs, targets)

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        return self.logistic_regression.predict(dataset)

    def predict_proba(self, dataset):
        """ Predicts the probabilities of each class for the dataset inputs.

            Arg:
                dataset the inputs to predict

            Returns: the probabilities of all inputs
        """
        return self.logistic_regression.predict_proba(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.logistic_regression.score(inputs, targets)

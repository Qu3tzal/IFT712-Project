# -*- coding: utf-8 -*-
import models.classifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

class LogisticRegressionClassifier(models.classifier.Classifier):
    """ This class is the abstract version of a classifier.
       All classifiers in this project should inherit this class to offer a
       uniform API.
    """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("logistic_regression")
        self.logistic_regression = GridSearchCV(LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000), param_grid={'C': [0.0001, 0.001, 0.01, 0.1, 1, 5]}, cv=3, iid=False)

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
        return self.logistic_regression.score(inputs, targets), metrics.log_loss(targets, self.predict_proba(inputs), labels=[str(x) for x in range(0,99)])

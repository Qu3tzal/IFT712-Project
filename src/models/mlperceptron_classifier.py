# -*- coding: utf-8 -*-
import models.classifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class MLPerceptronClassifier(models.classifier.Classifier):
    """ This class implements the multilayer perceptron classifier. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("MLPerceptronClassifier")
        self.mlp = GridSearchCV(
                        MLPClassifier(
                            solver='adam',
                            activation='relu',
                            hidden_layer_sizes=(99,), # We have 99 classes.
                            learning_rate='adaptive', # Makes the learning rate smaller when the loss cease to decrease.
                            max_iter=1000 # Make sure we have enough time to converge.
                        ),
                        {"alpha": 10.0 ** -np.arange(1, 7)},
                        cv=3,
                        iid=False
                    )

    def get_underlying_classifier(self):
        """ Returns the underlying classifier object. """
        return self.mlp

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        self.mlp.fit(inputs, targets)

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        return self.mlp.predict(dataset)

    def predict_proba(self, dataset):
        """ Predicts the probabilities of each class for the dataset inputs.

            Arg:
                dataset the inputs to predict

            Returns: the probabilities of all inputs
        """
        return self.mlp.predict_proba(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.mlp.score(inputs, targets)

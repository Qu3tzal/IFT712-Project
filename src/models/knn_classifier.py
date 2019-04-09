# -*- coding: utf-8 -*-
import models.classifier
from sklearn.neighbors import KNeighborsClassifier as knn

class KNNClassifier(models.classifier.Classifier):
    """ This class is the abstract version of a classifier.
        All classifiers in this project should inherit this class to offer a
        uniform API.
    """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("KNN")
        self.knn = knn(n_neighbors=3)

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        self.knn.fit(inputs, targets)

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        return self.knn.predict(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.knn.score(inputs, targets)

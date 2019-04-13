# -*- coding: utf-8 -*-
import models.classifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

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
        grid_parameters = {'n_neighbors': range(2, 15)}
        self.knn = GridSearchCV(knn(), grid_parameters, cv=3, iid=False)   # least populated class in y has only 3 members, so cv is set to 3

    def get_underlying_classifier(self):
        """ Returns the underlying classifier object. """
        return self.knn

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

    def predict_proba(self, dataset):
        """ Predicts the probabilities of each class for the dataset inputs.

            Arg:
                dataset the inputs to predict

            Returns: the probabilities of all inputs
        """
        return self.knn.predict_proba(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.knn.score(inputs, targets), metrics.log_loss(targets, self.predict_proba(inputs), labels=[str(x) for x in range(0,99)])

    def bestparams(self):
        """ Computes the accuracy on the given dataset.

            Arg:
                None

            Returns: parameters chosen by the GridSearchCV cross validation
        """
        return self.knn.best_params_

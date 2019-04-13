# -*- coding: utf-8 -*-
import models.classifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import GridSearchCV

class KNNClassifier(models.classifier.Classifier):
    """ This class implements the k-nearest neighbors classifier. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("KNN")
        grid_parameters = {'n_neighbors': range(2, 15)}
        self.knn = GridSearchCV(knn(), grid_parameters, cv=3, iid=False)   # least populated class in y has only 3 members, so cv is set to 3

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

    def bestparams(self):
        """ Computes the accuracy on the given dataset.

            Arg:
                None

            Returns: parameters chosen by the GridSearchCV cross validation
        """
        return self.knn.best_params_

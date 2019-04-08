# -*- coding: utf-8 -*-
import models.classifier
from sklearn.linear_model import LogisticRegression

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
        print("logistic regression classifier")
        # param as tests
        # TODO iteration problem while initializing logistic regression classifier
        self.logistic_regression = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000)

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        print("logistic regression training...")
        self.logistic_regression.fit(inputs, targets)

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        return self.logistic_regression.predict(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        print("logistic regression scoring...")
        return self.logistic_regression.score(inputs, targets)




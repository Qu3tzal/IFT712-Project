# -*- coding: utf-8 -*-
import models.classifier
import sklearn.ensemble
import models.svm_classifier
import models.knn_classifier
import models.ridge_classifier
import models.logistic_regression_classifier
import models.mlperceptron_classifier
import models.random_forest_classifier
from sklearn import metrics

class VotingClassifier(models.classifier.Classifier):
    """ This class implements a bagging classifier using all classifiers implemented so far. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("Voting Classifier")

        svm = models.svm_classifier.SVMClassifier().get_underlying_classifier()
        knn = models.knn_classifier.KNNClassifier().get_underlying_classifier()
        #ridge = models.ridge_classifier.RidgeClassifier().get_underlying_classifier()
        logistic_regression = models.logistic_regression_classifier.LogisticRegressionClassifier().get_underlying_classifier()
        random_forest = models.random_forest_classifier.RandomForestClassifier().get_underlying_classifier()
        mlp = models.mlperceptron_classifier.MLPerceptronClassifier().get_underlying_classifier()

        #self.voting = sklearn.ensemble.VotingClassifier(estimators=[('SVM_Classifier', svm), ('KNN_Classifier', knn), ('Ridge_Classifier', ridge), ('Logistic_Regression_Classifier', logistic_regression), ('Random_Forest_Classifier', random_forest), ('Multilayer_Perceptron_Classifier', mlp)], voting='soft')
        self.voting = sklearn.ensemble.VotingClassifier(estimators=[('SVM_Classifier', svm), ('KNN_Classifier', knn), ('Logistic_Regression_Classifier', logistic_regression), ('Random_Forest_Classifier', random_forest), ('Multilayer_Perceptron_Classifier', mlp)], voting='soft')

    def get_underlying_classifier(self):
        """ Returns the underlying classifier object. """
        return self.voting

    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        self.voting.fit(inputs, targets)

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        return self.voting.predict(dataset)

    def predict_proba(self, dataset):
        """ Predicts the probabilities of each class for the dataset inputs.

            Arg:
                dataset the inputs to predict

            Returns: the probabilities of all inputs
        """
        return self.voting.predict_proba(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.voting.score(inputs, targets), metrics.log_loss(targets, self.predict_proba(inputs), labels=[str(x) for x in range(0,99)])

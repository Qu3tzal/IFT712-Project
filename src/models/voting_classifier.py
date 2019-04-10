# -*- coding: utf-8 -*-
import models.classifier
from sklearn.ensemble import VotingClassifier
import models.svm_classifier
import models.knn_classifier
import models.ridge_classifier
import models.logistic_regression_classifier
import models.mlperceptron_classifier
import models.random_forest_classifier

class VotingClassifier(models.classifier.Classifier):
    """ This class implements a bagging classifier using all classifiers implemented so far. """

    def __init__(self):
        """ Constructor.

            Arg:
                name the name of the classifier
        """
        super().__init__("Voting Classifier")
        svm = models.svm_classifier.SVMClassifier()
        knn = models.knn_classifier.KNNClassifier()
        ridge = models.ridge_classifier.RidgeClassifier()
        logistic_regression = models.logistic_regression_classifier.LogisticRegressionClassifier()
        random_forest = models.random_forest_classifier.RandomForestClassifier()
        mlp = models.mlperceptron_classifier.MLPerceptronClassifier()
        #self.voting = VotingClassifier([('SVM Classifier', svm), ('KNN Classifier', knn), ('Ridge Classifier', ridge), ('Logistic Regression Classifier', logistic_regression), ('Random Forest Classifier', random_forest), ('Multilayer Perceptron Classifier', mlp)],'hard')
        self.voting = VotingClassifier(estimators=[('SVM_Classifier', svm), ('KNN_Classifier', knn), ('Ridge_Classifier', ridge), ('Logistic_Regression_Classifier', logistic_regression), ('Random_Forest_Classifier', random_forest), ('Multilayer_Perceptron_Classifier', mlp)], voting='hard')


    def train(self, inputs, targets):
        """ Trains the model on the given dataset.

            Arg:
                inputs the inputs
                targets the targets
        """
        self.VotingClassifier.fit(inputs, targets)

    def predict(self, dataset):
        """ Predicts the dataset.

            Arg:
                dataset the inputs to predict

            Returns: the prediction of all inputs
        """
        return self.VotingClassifier.predict(dataset)

    def score(self, inputs, targets):
        """ Computes the accuracy on the given dataset.

            Arg:
                inputs the inputs
                targets the targets

            Returns: the accuracy
        """
        return self.VotingClassifier.score(inputs, targets)

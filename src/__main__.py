# -*- coding: utf-8 -*-
import sys
import data.bootstrap, data.database
import features.preparator as preparator
import models.classifier
import models.svm_classifier
import models.knn_classifier
import models.ridge_classifier
import models.logistic_regression_classifier
import models.mlperceptron_classifier
import models.random_forest_classifier
import models.voting_classifier


def main():

    """ Main function.
    """

    training_methods = ["ridge", "SVM", "KNN", "logistic_regression", "random_forest", "multi_layer_neural_network", "bagging"]
    methods_chosen = []
    methods_authorized = []

    if len(sys.argv) < 2:
        usage = "\n----------------\n" \
                "\nUsage: " \
                "\n\tpython src training_method" \
                "\n\tpython src training_method_1 training_method_2 training_method_N" \
                "\n\n\ttraining_method: all, ridge, SVM, KNN, logistic_regression, random_forest, multi_layer_neural_network, bagging" \
                "\n\n----------------\n\n"
        print(usage)
        return
    if len(sys.argv) == 2 and sys.argv[1] == "all":
        methods_authorized = training_methods.copy()
        print(methods_authorized)
    elif len(sys.argv) >= 2 and len(sys.argv) < 8:
        methods_chosen = sys.argv[1:]
        for method in methods_chosen:
            if method not in training_methods:
                print(method + " is not a valid method")
            else:
                methods_authorized.append(method)
                print(method)
    else:
        print("Please enter between 1 and 7 methods.")
        return

    # Load the database.
    db = data.database.Database('data')
    db.load()

    # Get the datasets.
    dataset = db.get_train_dataset().drop('id', axis=1)

    # Prepare the datasets.
    shuffler = preparator.ShufflePreparator()
    shuffler.prepare(dataset, dataset.columns[1:])

    pca = preparator.PCAPreparator()
    pca.prepare(dataset, dataset.columns[1:])

    standardizer = preparator.StandardizerPreparator()
    standardizer.prepare(dataset, dataset.columns[1:]) # We don't apply preparation on the Species (=target) columns.

    name2int = preparator.Name2IntPreparator()
    name2int.prepare(dataset, ['species']) # Replace the species name by an integer.

    # Split the dataset in 70%/30%.
    training_dataset = dataset[:int(dataset.shape[0] * 0.7)]
    test_dataset = dataset[int(dataset.shape[0] * 0.7):]

    # Separate the inputs from the targets.
    training_inputs = training_dataset.drop('species', axis=1)
    training_targets = training_dataset['species']

    test_inputs = test_dataset.drop('species', axis=1)
    test_targets = test_dataset['species']

    # Create the classifier.
    classifiers = []

    for method in methods_authorized:
        if method == 'SVM':
            classifiers.append(models.svm_classifier.SVMClassifier())
        elif method == 'KNN':
            classifiers.append(models.knn_classifier.KNNClassifier())
        elif method == 'ridge':
            classifiers.append(models.ridge_classifier.RidgeClassifier())
        elif method == 'logistic_regression':
            classifiers.append(models.logistic_regression_classifier.LogisticRegressionClassifier())
        elif method == 'random_forest':
            classifiers.append(models.random_forest_classifier.RandomForestClassifier())
        elif method == 'multi_layer_neural_network':
            classifiers.append(models.mlperceptron_classifier.MLPerceptronClassifier())
        elif method == 'bagging':
            classifiers.append(models.voting_classifier.VotingClassifier())
        else:
            raise RuntimeError("Invalid training method name.")

    print("Models training...\n")
    for clf, method in zip(classifiers, methods_authorized):
        # Train.
        clf.train(training_inputs, training_targets)

        # Output the score.
        print(method)
        print("\t\t" + str(clf.score(test_inputs, test_targets)))

if __name__ == "__main__":
    main()

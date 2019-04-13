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
import pipeline.kaggle_pipeline

def main_kaggle(training_method):
    classifier = None
    if training_method == 'SVM':
        classifier = models.svm_classifier.SVMClassifier()
    elif training_method == 'KNN':
        classifier = models.knn_classifier.KNNClassifier()
    elif training_method == 'ridge':
        classifier = models.ridge_classifier.RidgeClassifier()
    elif training_method == 'logistic_regression':
        classifier = models.logistic_regression_classifier.LogisticRegressionClassifier()
    elif training_method == 'random_forest':
        classifier = models.random_forest_classifier.RandomForestClassifier()
    elif training_method == 'multi_layer_neural_network':
        classifier = models.mlperceptron_classifier.MLPerceptronClassifier()
    elif training_method == 'bagging':
        classifier = models.voting_classifier.VotingClassifier()
    else:
        raise RuntimeError("Invalid training method name.")

    # Load the database.
    print("Loading database...")
    db = data.database.Database('data')
    db.load()
    print("\tDatabase loaded.")

    # Create the pipeline.
    kp = pipeline.kaggle_pipeline.KagglePipeline(
                    classifier,
                    db.get_train_dataset().drop('id', axis=1),
                    db.get_test_dataset()
                )
    print("Training, predicting and writing file...")
    kp.predict("submission.csv")
    print("\tDone.")

def main():

    """ Main function.
    """

    if len(sys.argv) < 2:
        usage = "\n----------------\n" \
                "\nUsage: python src training_method\
                \n\ttraining_method: SVM, KNN, logistic_regression, random_forest, multi_layer_neural_network, bagging" \
                "\n\n----------------\n\n"
        print(usage)
        return

    training_method = sys.argv[1]

    if training_method == "kaggle":
            main_kaggle(sys.argv[2])
            return 0

    print("Training method: " + training_method)

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
    classifier = None
    if training_method == 'SVM':
        classifier = models.svm_classifier.SVMClassifier()
    elif training_method == 'KNN':
        classifier = models.knn_classifier.KNNClassifier()
    elif training_method == 'ridge':
        classifier = models.ridge_classifier.RidgeClassifier()
    elif training_method == 'logistic_regression':
        classifier = models.logistic_regression_classifier.LogisticRegressionClassifier()
    elif training_method == 'random_forest':
        classifier = models.random_forest_classifier.RandomForestClassifier()
    elif training_method == 'multi_layer_neural_network':
        classifier = models.mlperceptron_classifier.MLPerceptronClassifier()
    elif training_method == 'bagging':
        classifier = models.voting_classifier.VotingClassifier()
    else:
        raise RuntimeError("Invalid training method name.")

    # Train.
    classifier.train(training_inputs, training_targets)

    # Output the score.
    print(classifier.score(test_inputs, test_targets))

if __name__ == "__main__":
    main()

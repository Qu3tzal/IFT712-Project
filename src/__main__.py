# -*- coding: utf-8 -*-
import data.bootstrap, data.database
import features.preparator as preparator
import models.svm_classifier
import models.logistic_regression_classifier

def main():
    """ Main function.
    """
    # Load the database.
    db = data.database.Database('data')
    db.load()

    # Get the datasets.
    dataset = db.get_train_dataset().drop('id', axis=1)

    # Prepare the datasets.
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
    logistic_classifier = models.logistic_regression_classifier.LogisticRegressionClassifier()

    # Train.
    logistic_classifier.train(training_inputs, training_targets)

    # Output the score.
    print(logistic_classifier.score(training_inputs, training_targets))

if __name__ == "__main__":
    main()

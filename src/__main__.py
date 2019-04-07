# -*- coding: utf-8 -*-
import data.bootstrap, data.database
import features.preparator as preparator

def main():
    """Main function.
    """
    # Load the database.
    db = data.database.Database('data')
    db.load()

    # Get the datasets.
    train_ds = db.get_train_dataset()
    test_ds = db.get_test_dataset()

    # Prepare the datasets.
    standardizer = preparator.StandardizerPreparator()
    standardizer.prepare(train_ds, train_ds.columns[2:]) # We don't apply preparation on the ID and Species (=target) columns.
    standardizer.prepare(test_ds, test_ds.columns[1:]) # We don't apply preparation on the ID column.

    # Test the bootstrap.
    bootstraps = data.bootstrap.bootstrap_dataset(train_ds, 5)

    # Train.

    # Output the score.

if __name__ == "__main__":
    main()

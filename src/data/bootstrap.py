# -*- coding: utf-8 -*-


def bootstrap_dataset(dataset, m):
    """ Creates m new datasets from the given dataset.

        Creates m new datasets by bootstrapping the given dataset.

        Arg:
            dataset a DataFrame object
            m the number of datasets to generate
    """
    new_datasets = []

    for i in range(m):
        # Resample.
        resampled_data = dataset.sample(n=dataset.shape[0], replace=True)

        # Push the new dataset.
        new_datasets.append(resampled_data)

    return new_datasets

from .corruption import CORRUPTION_DATASETS, get_corruption_dataset_class
from .domainbed import DOMAINBED_DATASETS, get_domainbed_dataset_class


def get_dataset_class(dataset_name):
    if dataset_name in CORRUPTION_DATASETS:
        return get_corruption_dataset_class(dataset_name)

    elif dataset_name in DOMAINBED_DATASETS:
        return get_domainbed_dataset_class(dataset_name)

    else:
        raise Exception("Unknown dataset name")

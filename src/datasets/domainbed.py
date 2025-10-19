import os
from torchvision.datasets import ImageFolder

DOMAINBED_DATASETS = [
    'VLCS',
    'TerraIncognita',
]


class MultiImageFolder:

    def __init__(self, root, environments, transform=None):
        self.environments = environments

        self.datasets = []

        for i, environment in enumerate(self.environments):
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=transform)
            self.datasets.append(env_dataset)

        # assume they have same number of classes?
        self.classes = self.datasets[-1].classes
        self.num_classes = len(self.datasets[-1].classes)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class VLCS(MultiImageFolder):
    def __init__(self, root, transform=None):
        root = os.path.join(root, "domainbed", "VLCS")
        environments = [
            'Caltech101',
            'LabelMe',
            'SUN09',
            'VOC2007'
        ]
        MultiImageFolder.__init__(self, root, environments, transform=transform)


class TerraIncognita(MultiImageFolder):
    def __init__(self, root, transform=None):
        root = os.path.join(root, "domainbed", "terra_incognita")
        environments = [
            'location_100',
            'location_38',
            'location_43',
            'location_46'
        ]
        MultiImageFolder.__init__(self, root, environments, transform=transform)


def get_domainbed_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

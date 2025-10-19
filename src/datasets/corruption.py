import os
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

CORRUPTION_DATASETS = [
    # 32 x 32 images
    "CIFAR10CFull",
    "CIFAR100CFull",
]

# Use the order in most papers
main_corruptions = [
    # Noise
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',

    # Blur
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',

    # Weather
    'snow',
    'frost',
    'fog',
    'brightness',

    # Digital
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
]

extra_corruptions = [
    'speckle_noise',
    'gaussian_blur',
    'spatter',
    'saturate',
]

all_corruptions = [
    # Noise
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'speckle_noise',

    # Blur
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'gaussian_blur',

    # Weather
    'snow',
    'frost',
    'fog',
    'brightness',
    'spatter',

    # Digital
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    'saturate',
]


class NumpyImageDataset:

    def __init__(self, X, Y, transform=None):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        if transform:
            self.transform = transform

        else:
            self.transform = lambda x: x  # identity mapping

    def __getitem__(self, item):
        return self.transform(Image.fromarray(self.X[item], mode='RGB')), int(self.Y[item])

    def __len__(self):
        return len(self.X)


class MultipleCorruptionNumpyImageDataset:
    def __init__(self, root, extra=True, severity=5, transform=None):

        if extra:
            self.environments = all_corruptions
        else:
            self.environments = main_corruptions

        self.datasets = []

        label_path = os.path.join(root, f"labels.npy")
        label = np.load(label_path)

        num_data = len(label) // 5  # since there are severity 1 - 5
        label = label[num_data * (severity - 1): num_data * severity]

        for i, environment in enumerate(self.environments):
            feat_path = os.path.join(root, f"{environment}.npy")
            feat = np.load(feat_path)

            feat = feat[num_data * (severity - 1): num_data * severity]  # use the corresponding severity

            env_dataset = NumpyImageDataset(feat, label, transform)
            self.datasets.append(env_dataset)

        self.set_classes()

    def set_classes(self):
        # implemented in Subclass
        raise NotImplementedError

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class CIFAR10CFull(MultipleCorruptionNumpyImageDataset):

    def __init__(self, root, extra=True, severity=5, transform=None):
        root = os.path.join(root, "corruption", "CIFAR-10-C-Full")
        MultipleCorruptionNumpyImageDataset.__init__(self, root, extra, severity, transform)

    def set_classes(self):
        self.classes = ['airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck']
        self.num_classes = len(self.classes)


class CIFAR100CFull(MultipleCorruptionNumpyImageDataset):

    def __init__(self, root, extra=True, severity=5, transform=None):
        root = os.path.join(root, "corruption", "CIFAR-100-C-Full")
        MultipleCorruptionNumpyImageDataset.__init__(self, root, extra, severity, transform)

    def set_classes(self):
        self.classes = ['apple',
                        'aquarium_fish',
                        'baby',
                        'bear',
                        'beaver',
                        'bed',
                        'bee',
                        'beetle',
                        'bicycle',
                        'bottle',
                        'bowl',
                        'boy',
                        'bridge',
                        'bus',
                        'butterfly',
                        'camel',
                        'can',
                        'castle',
                        'caterpillar',
                        'cattle',
                        'chair',
                        'chimpanzee',
                        'clock',
                        'cloud',
                        'cockroach',
                        'couch',
                        'crab',
                        'crocodile',
                        'cup',
                        'dinosaur',
                        'dolphin',
                        'elephant',
                        'flatfish',
                        'forest',
                        'fox',
                        'girl',
                        'hamster',
                        'house',
                        'kangaroo',
                        'keyboard',
                        'lamp',
                        'lawn_mower',
                        'leopard',
                        'lion',
                        'lizard',
                        'lobster',
                        'man',
                        'maple_tree',
                        'motorcycle',
                        'mountain',
                        'mouse',
                        'mushroom',
                        'oak_tree',
                        'orange',
                        'orchid',
                        'otter',
                        'palm_tree',
                        'pear',
                        'pickup_truck',
                        'pine_tree',
                        'plain',
                        'plate',
                        'poppy',
                        'porcupine',
                        'possum',
                        'rabbit',
                        'raccoon',
                        'ray',
                        'road',
                        'rocket',
                        'rose',
                        'sea',
                        'seal',
                        'shark',
                        'shrew',
                        'skunk',
                        'skyscraper',
                        'snail',
                        'snake',
                        'spider',
                        'squirrel',
                        'streetcar',
                        'sunflower',
                        'sweet_pepper',
                        'table',
                        'tank',
                        'telephone',
                        'television',
                        'tiger',
                        'tractor',
                        'train',
                        'trout',
                        'tulip',
                        'turtle',
                        'wardrobe',
                        'whale',
                        'willow_tree',
                        'wolf',
                        'woman',
                        'worm']

        self.num_classes = len(self.classes)


def get_corruption_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

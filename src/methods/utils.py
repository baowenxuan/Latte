import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_entropy(x):
    return -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1)


def get_clip_logits(image_features, clip_weights, normalize=False, p=0.1, return_features=False):
    """
    Get clip logits and auxiliary outputs
    :param image_features: batch_size * feat_dim
    :param clip_weights: num_class * feat_dim
    :param normalize: whether to normalize
    :return:
    """

    # TODO: batch size > 1? item() only applies to batch_size = 1

    if normalize:
        image_features = F.normalize(image_features, dim=1)
        clip_weights = F.normalize(clip_weights, dim=1)

    batch_size = image_features.shape[0]

    if batch_size > 1:  # use augmix to generate multiple augmentations
        clip_logits = 100.0 * image_features @ clip_weights.T
        batch_entropy = softmax_entropy(clip_logits)
        selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_size * p)]
        image_features = F.normalize(image_features[selected_idx].mean(dim=0, keepdim=True), dim=1)

    clip_logits = 100.0 * image_features @ clip_weights.T  # HalfTensor if cuda else Float Tensor
    pred = clip_logits.argmax(dim=1).item()  # int
    proba = clip_logits.softmax(dim=1)  # HalfTensor if cuda else FloatTensor
    entropy = softmax_entropy(clip_logits).item()  # float

    if return_features:
        return clip_logits, pred, proba, entropy, image_features
    else:
        return clip_logits, pred, proba, entropy


def get_entropy_batch(image_features, clip_weights, normalize=False):
    if normalize:
        image_features = F.normalize(image_features, dim=1)
        clip_weights = F.normalize(clip_weights, dim=1)

    clip_logits = 100.0 * image_features @ clip_weights.T
    entropy = softmax_entropy(clip_logits)  # HalfTensor if cuda else FloatTensor

    return entropy

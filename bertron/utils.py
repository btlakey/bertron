import torch
from sklearn.metrics import f1_score
import numpy as np


def be_there_gpus():
    """check for presence of gpus
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f1_score_func(preds, labels):
    """calculate f1 score given nested numpy arrays

    :param preds: numpy.ndarray
        predictions, dim=()
    :param labels: numpy.ndarray
        truth values, dim()

    :return: float
        f1 score
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels, label_dict):
    """ class (author) accuracy (proportion true predictions: tp+tn / tp+tn+fp+fn)

    :param preds: numpy.ndarray
        predictions, dim=()
    :param labels: numpy.ndarray
        truth values, dim()
    :param label_dict: dict
        mapping between authors (key) and int labels (value)

    :return: None
        print class accuracy metrics
    """
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        label_mask = labels_flat == label
        y_preds = preds_flat[label_mask]
        y_true = labels_flat[label_mask]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])} / {len(y_true)}\n')

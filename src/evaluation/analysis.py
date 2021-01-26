# Different analysis functions to evaluate the perfomance of a model

import torch
import numpy as np
import sys

from sklearn.metrics import confusion_matrix
from torch import Parameter


def binary_accuracy(preds, y):
    """ 
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.zeros(preds.size())
    for i in range(len(preds)):
        idx_max = torch.where(preds[i] == preds[i].max())
        rounded_preds[i][idx_max] = 1
    rounded_preds = torch.FloatTensor(rounded_preds).int()
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = (correct.sum() / len(preds[0])) / len(correct)
    return acc


def evaluate_test(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    print("Evaluating...")
    predictions_all = []

    for i, batch in enumerate(iterator):
        if not i % 10:
            sys.stdout.write(f"\rIteration {i}        ")
            sys.stdout.flush()

        inputs, labels_onehot = batch

        inputs = Parameter(inputs.float(), requires_grad=False).cuda()

        predictions = model(inputs)
        labels_idx = torch.LongTensor(
            [np.where(label == 1)[0][0] for label in labels_onehot]
        ).cuda()

        loss = criterion(predictions, labels_idx)
        acc = binary_accuracy(predictions, labels_onehot)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        predictions_all.append(predictions)
    return (
        epoch_loss / len(iterator),
        epoch_acc / len(iterator),
        predictions_all,
    )


def to_confussion_matrix(y, preds):
    """Compute confussion matrix with ground truth `y` of `preds`."""
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

"""
Functions to build datasets to evaluate the performance of some trained model.

Different datasets should be implemented:
    1. Take random chunks of some size from the genome and label them.
    2. Take overlapping windows from the input genome starting at the first
       position.
"""

import numpy as np
import pandas as pd
import pyfastx
import torch
import sys


sys.path.append("../post-processing/")

# from preproc_pipeline import window_pipeline
from filters import lowpass_filter, bin_acc_filter, cut_filter
from analysis import evaluate_test
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset


class oversampdata(Dataset):
    def __init__(self, data):
        # first column is list of index sentence
        self.data = LongTensor(
            np.ndarray.astype(
                np.array([np.array(r) for r in data.iloc[:, 0].to_numpy()]),
                "int64",
            )
        )
        # second column is the label
        self.targets = LongTensor(
            np.ndarray.astype(
                np.array(
                    [np.array(eval(r)) for r in data.iloc[:, 1].to_numpy()]
                ),
                "int64",
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_val = self.data[index]
        target = self.targets[index]
        return data_val, target


def retrieve_annotations(fseq_idx, feature_table):
    annotations = pd.read_csv(feature_table, sep="\t")
    true_annotations = []

    for index, row in annotations.iterrows():
        if row["start"] in fseq_idx:
            true_annotations.append(row["start"])
        if (row["end"] in fseq_idx) and len(true_annotations) == 0:
            true_annotations.append(fseq_idx[0])
            true_annotations.append(row["end"])
        elif (row["end"] in fseq_idx) and len(true_annotations) >= 1:
            true_annotations.append(row["end"])
    return true_annotations


def annotate(idx_window, true_annot):
    for idx, annot in enumerate(true_annot):
        if annot in idx_window:
            # Start gene
            if idx % 2 == 0:
                if idx_window[-1] - annot >= 0.9 * len(idx_window):
                    return 1
            # End gene
            else:
                if annot - idx_window[0] >= 0.9 * len(idx_window):
                    return 1

        # Within genes
        elif idx_window and idx < (len(true_annot) - 1):
            if idx_window[0] > annot and idx_window[-1] < true_annot[idx + 1]:
                return 1
    return 0


def cut_annotate_seq(fseq, fseq_idx, true_annot, window_size, slide):
    seq = []
    seq_annot = []
    N = len(fseq)
    for i in range(0, N - window_size + 1, slide):
        seq.append(fseq[i : i + window_size])

        idx_window = fseq_idx[i : i + window_size]
        seq_annotation = annotate(idx_window, true_annot)
        seq_annot.append(seq_annotation)

    return (seq, seq_annot)


def code_one_hot(seq, vocab):
    fit_seq = list(map(lambda x: "UNK" if x not in vocab else x, seq))
    encoding = np.array([vocab[ch] for ch in fit_seq], dtype="int64")

    encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    return encoding


def class2onehot(seq, vocab):
    return vocab[seq]


def slide_genome(
    genomes,
    labels,
    feat_files,
    indeces,
    window_size,
    slide=5,
    sep="\t",
    **kwargs,
):
    r"""Take overlapping windows of size `window_size` of the genomes.

    Parameters
    ----------
    genomes: list
        list of paths (str) to genome files
    feat_files: list
        list of paths (str) to feature files
    indeces: list
        list of lists (len 2), indication the positions that correspond to each
        evaluation chunk in the genome
    window_size: int
        size of chunks
    slide: int
        every step of the windows
    sep: "\t"
        separator character of the feat_file table
    **kwargs:
        Ignored. Just for consistency with other post-processing methods.

    Result
    ------
    testing_loader torch.utils.data.DataLoader
        evaluation object

    """
    nuc_to_class = {"A": [0], "G": [1], "T": [2], "C": [3], "UNK": [4]}
    nuc_to_class = {k: np.array(v, "int64") for k, v in nuc_to_class.items()}
    # put all the genome eval info on one big dataframe
    # the information of the evaluation comes from half of the fasta file of
    # genomes (some of them are the first half, some of them, the second)
    # the dataset used for training is the opposite part
    dft = []
    for i in range(len(feat_files)):
        print(f"Processing {genomes[i]}...")
        fna = pyfastx.Fasta(genomes[i])
        genome = str(fna[0])

        test_seq_idx = list(range(indeces[i][0], indeces[i][1]))

        # Get the true annotations and annotate the windows accordingly
        true_annotations = retrieve_annotations(test_seq_idx, feat_files[i])
        (processed_test_seq, test_annotation) = cut_annotate_seq(
            genome, test_seq_idx, true_annotations, window_size, slide
        )
        df_testing = pd.DataFrame(dict(sequence=processed_test_seq))
        df_testing["c_class"] = df_testing["sequence"].apply(
            lambda x: code_one_hot(x, nuc_to_class)
        )
        df_testing["Annotation"] = test_annotation

        num_to_onehot = {0: [0, 1], 1: [1, 0]}
        df_testing["onehot_label"] = df_testing["Annotation"].apply(
            lambda x: class2onehot(x, num_to_onehot)
        )
        dft.append(df_testing)
    all_tests = pd.concat(dft)
    all_tests.to_csv("all_testing.tsv", sep="\t")  # \ŧ
    return all_tests


def evaluate_all(rnn, all_tests, criterion, order=1, fs=100, cutoff=5):
    """Evaluate on dataset `testingloader` before and after filter.

    Parameters
    ----------
    rnn: torch.model
        Deep Learning model for evaluation DNA sequential data.
    all_tests: pandas.DataFrame
        from `slide_genome`
    criterion: torch.optim.criterion
    order: int
    fs: float
        sample rate
    cutoff
        desired cutoff frequency of the filter

    Returns
    -------
    valid_acc: float
        raw accuraccy of model
    rounded_acc: float
        rounded accuracy of model (should be the same as `valid_acc`)
    filt_acc: float
        accuracy after applying a lowpass filter
    preds:
        raw predictions of model
    filt_preds:
        predicitions after lowpass filter
    test_annotation: pandas.Series
        ground truth to be compared

    """
    test_annotation = all_tests["Annotation"]
    testing_dataset = oversampdata(
        all_tests.loc[:, ["c_class", "onehot_label"]]
    )
    testingloader = DataLoader(
        testing_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    valid_loss, valid_acc, preds = evaluate_test(rnn, testingloader, criterion)
    rounded_preds = torch.zeros(len(preds))
    for i in range(len(preds)):
        if preds[i][0][0] > preds[i][0][1]:
            rounded_preds[i] = 1
        else:
            rounded_preds[i] = 0

    rounded_acc = bin_acc_filter(rounded_preds, test_annotation)
    filt_preds = cut_filter(lowpass_filter(rounded_preds, cutoff, fs, order))
    filt_acc = bin_acc_filter(filt_preds)
    return valid_acc, rounded_acc, filt_acc, preds, filt_preds, test_annotation

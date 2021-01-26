# -*- coding: utf-8 -*-
"""cbow_model.py

    Created on Mon Nov  22 2019

    @author: arita
"""
import pandas as pd
import torch
import torch.nn as nn


def window(fseq, window_size, slide=1):
    # create a window of size k
    N = len(fseq)
    for i in range(0, N - window_size + 1, slide):
        if i + window_size + slide < N:
            yield fseq[i : i + window_size]


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i
    return index


def get_max_prob_result(input, ix_to_word):
    return ix_to_word[get_index_of_max(input)]


def vocabulary(self, fseq):
    all_seqs = "".join([seq for seq in df_train.loc[:, "sequence"]])
    kmers = [kmer for kmer in window(all_seqs, k, SLIDE)]
    print("kmers ", len(set(kmers)))
    vocab_size = len(kmers)
    print("vocabulary size:", vocab_size)

    data = []

    # print("Filling context data...")
    # pbar = tqdm(total=vocab_size - 4-1) # just to output something in screen
    for i in range(
        2, vocab_size - 2
    ):  # first word to have 2 words before is the "third" one (0,1,2)
        context = (kmers[i - 2], kmers[i - 1], kmers[i + 1], kmers[i + 2])
        target = kmers[i]
        data.append((context, target))
    # pbar.update(1)

    print("data entries:", len(data), type(data))

    word_to_ix, ix_to_word = {}, {}
    for i, word in enumerate(set(kmers)):
        word_to_ix[word] = i
        ix_to_word[i] = word

    # print (len(word_to_ix))

    ix = len(ix_to_word)
    ix_to_word[ix] = "X"  # our padding character
    word_to_ix["X"] = ix

    return (word_to_ix, ix_to_word)


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super(CBOW, self).__init__()
        """
        Parameters
        ----------
        vocab_size:
        embedding_dim:
        padding_idx:

        Returns
        -------
        """

        # out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )  # used predefined nn.Embedding
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        # out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        print(type(word))
        word = torch.LongTensor([word_to_ix[word]])
        return self.embeddings(word).view(1, -1)

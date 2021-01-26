"""This document implements the LSTM architecture with embeddings."""

import torch.nn as nn


class embedLSTM(nn.Module):
    """We still need the DropConnect."""

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        lstm_layers,
        hidden_out,
        output_dim,
        padding_idx,
        t,
    ):

        super().__init__()
        self.nb_tags = output_dim

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=padding_idx)

        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,
            num_layers=lstm_layers,
            dropout=0.4,
        )

        self.fnn2 = nn.Sequential(
            nn.Linear(t * hidden_dim * 2, hidden_out),
            nn.BatchNorm1d(hidden_out),
            nn.SELU(),
            nn.AlphaDropout(),
            nn.Linear(hidden_out, output_dim),
        )

    def forward(self, text):

        # text = [sent len, batch size]
        # 1. embedding
        embedded = self.embedding(text)

        # 2. LSTM
        output, (hidden, lt) = self.rnn(embedded)

        # 3. get that so it's correctly packed for the hidden layer
        output = output.contiguous()
        output = output.reshape(output.shape[0], -1)

        # 4. classification through the 2 FF layers
        output = self.fnn2(output)
        return output


def criterion():
    """Just return the criterion we used for training."""
    return nn.CrossEntropyLoss()


if __name__ == "__main__":
    # just see if it can be instantiated
    rnn = embedLSTM(4, 4, 4, 4, 4, 4, 0, 1)

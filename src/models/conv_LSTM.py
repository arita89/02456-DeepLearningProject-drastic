"""Convolutional embeddings with LSTM implementation."""

import torch.nn as nn


class convLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_out, output_dim, t):

        super().__init__()
        self.nb_tags = output_dim

        # , in_channels, out_channels, kernel_size, stride
        in_channels = 1
        out_channels = 1
        kernel_size = 8
        stride = 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=0, bias=False
        )
        # self.maxpool = nn.maxpool()

        # out_seq_len = seq_len - kernel_size + 1

        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=False, batch_first=True)

        self.lhid = nn.Linear(22 * hidden_dim, hidden_out)

        self.fc = nn.Linear(hidden_out, output_dim)

        self.drop = nn.Dropout(0.25)

    def forward(self, text):

        text = text.permute(0, 2, 1)
        output = self.conv(text)
        # print(output.data.size())

        output = output.permute(0, 2, 1)
        # print(output.data.size())
        # text = [sent len, batch size]
        # 1. LSTM
        output, hidden = self.rnn(output)
        # print('rnn')
        # print(output.data.size())

        # 2. get that so it's correctly packed for the hidden layer
        output = output.contiguous()
        output = output.reshape(output.shape[0], -1)
        # print('reshape')
        # print(output.size())

        output = self.drop(nn.functional.relu(self.lhid(output)))
        # print('ff')
        # print(output.size())
        # print('after linear layer')
        # print(output.size())

        # 3. classification
        output = self.fc(output)
        # print('fff')
        # print(output.size())
        # print(output.size())
        # output = nn.functional.log_softmax(output, dim=1)
        # print(output.size())
        # output = output.view(text.size()[0], -1)
        return output


def criterion():
    """Just return the criterion we used for training."""
    return nn.CrossEntropyLoss()


if __name__ == "__main__":
    # just see if it can be instantiated
    rnn = convLSTM(4, 4, 4, 4, 4)

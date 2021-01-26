"""Convolutional layer + GRU used for one-hot encoded."""
import torch.nn as nn


def compute_conv_dim(dim_size, kernel, padding, stride):
    return int((dim_size + 2 * padding - (kernel-1) - 1) / stride + 1)


class convLSTM(nn.Module):
    def __init__(self, t, input_dim=1, hidden_dim=34, hidden_out=90, hiddenh=70, 
        output_dim=2, out_channels=1, kernel_size=8, stride=2, padding=0,
        bidirectional=True):

        super().__init__()
        self.nb_tags = output_dim

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, out_channels, kernel_size, 
                              stride, padding=padding, bias = False),
            nn.BatchNorm1d(out_channels),
            # nn.ReLU()
        )

        
        # compute output dimension of Convnet as input t of LSTM
        len_to_lstm = (
            out_channels *
            compute_conv_dim(t, kernel_size, padding, stride) 
        )
        directions = 2 if bidirectional else 1

        self.rnn = nn.GRU(out_channels, hidden_dim, bidirectional=bidirectional, 
                           batch_first=True, dropout=0.2)
        
        self.fnn2 = nn.Sequential(
            nn.Linear(int(len_to_lstm/out_channels)*hidden_dim*directions, hidden_out),
            nn.BatchNorm1d(hidden_out), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_out, hiddenh),
            nn.BatchNorm1d(hiddenh), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hiddenh, output_dim)
        )


    def forward(self, text):
        
        text = text.permute(0,2,1)
        output = self.conv(text)

        output = output.permute(0,2,1)
        
        # 1. LSTM
        output, hidden = self.rnn(output)

        # 2. get that so it's correctly packed for the hidden layer
        output = output.contiguous()
        output = output.reshape(output.shape[0], -1)

        # 3. classification
        output = self.fnn2(output)
        return output
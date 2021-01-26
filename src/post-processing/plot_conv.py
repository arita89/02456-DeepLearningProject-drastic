"""Original source: https://github.com/utkuozbulak/pytorch-cnn-visualizations.

Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak

"""

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from torch.nn import Parameter
from torch.optim import Adam


def recreate_logo(im_as_var):
    """Create a logo from a optimized Variable."""
    df = pd.DataFrame(
        im_as_var[0].detach().numpy(), columns=["A", "G", "T", "C"]
    )
    # apply softmax to retrieve stochastic vectors
    df = df.apply(lambda x: np.exp(x) / np.sum(np.exp(x)), axis=1)

    fig, ax = plt.subplots(1, 1)
    nn_logo = logomaker.Logo(df, ax=ax)
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=["left"], visible=True, bounds=[0, 1])
    nn_logo.draw()
    fig.tight_layout()
    return fig


class CNNLayerVisualization:
    """Produce an image that minimizes the loss of a filter on a conv layer."""

    def __init__(self, model, selected_layer, selected_filter):
        """Init with a torch neural network `model`."""
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists("../generated"):
            os.makedirs("../generated")

    def visualise_layer1D(self, save=True):
        """Plot activations but just for one dimension (four pixels/lettes)."""
        # Generate a random image
        random_seq = np.uint8(np.random.uniform(0, 4, (100, 4)))
        # TODO: the Parameter here must be initialized with one dummy dimesion
        #       emulating th batch size dimension
        var_seq = np.ndarray.astype(
            np.array([np.array(r) for r in random_seq]), "int64"
        )
        var_seq = Parameter(
            torch.LongTensor(var_seq[None, :, :]).float(), requires_grad=True
        )
        # size of resulting images
        plt.rcParams["figure.figsize"] = 16, 4
        # Define optimizer for the image
        optimizer = Adam([var_seq], lr=0.1, weight_decay=1e-6)
        for i in range(1, 100):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = var_seq
            for index, layer in enumerate(self.model.modules()):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            self.conv_output = x[0, self.selected_filter]
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print(
                "Iteration:",
                str(i),
                "Loss:",
                "{0:.2f}".format(loss.data.numpy()),
            )
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # Save image
            if i % 5 == 0 and save:
                self.created_image = recreate_logo(var_seq)
                im_path = (
                    "../generated/layer_vis_l"
                    + str(self.selected_layer)
                    + "_f"
                    + str(self.selected_filter)
                    + "_iter"
                    + str(i)
                    + ".jpg"
                )
                self.created_image.savefig(im_path)
                plt.close()  # save memory
        return var_seq


if __name__ == "__main__":
    import sys

    sys.path.append("../models/")

    from conv_LSTM_onehot import convLSTM

    cnn_layer = 0
    filter_pos = 1

    nuc_to_ix = {
        "A": [1, 0, 0, 0],
        "G": [0, 1, 0, 0],
        "T": [0, 0, 1, 0],
        "C": [0, 0, 0, 1],
    }
    class_to_nuc = {v[0]: k for k, v in nuc_to_ix.items()}
    t = 100
    path_model = "../../data/models/convlstm_big.pt"
    pretrained_model = convLSTM(
        input_dim=4,
        out_channels=4,
        stride=5,
        hidden_dim=60,
        hidden_out=140,
        output_dim=2,
        t=t,
    ).cpu()
    pretrained_model.load_state_dict(
        torch.load(path_model, map_location=torch.device("cpu"))
    )
    pretrained_model = pretrained_model

    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    opt_seq = layer_vis.visualise_layer1D().data.numpy()

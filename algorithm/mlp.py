"""Plain MLP Q-network."""

from torch import nn


class Mlp(nn.Module):
    """Plain MLP Q-network: state (a small fixed-length feature vector, not
    a candidate sequence) -> one Q-value per discrete tilt index.

    input_size -> hidden_sizes[0] -> ... -> hidden_sizes[-1] -> output_size.
    Each hidden layer is Linear+ReLU; the final layer is a plain Linear (no
    activation -- these are Q-value heads).
    """

    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        sizes = [input_size, *hidden_sizes]
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers += [nn.Linear(in_size, out_size), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)

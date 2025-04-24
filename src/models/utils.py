import torch


def relu_seq(sizes):
    layers = []
    d_input = sizes[0]

    for d_hidden in sizes[1:-1]:
        layers.append(nn.Linear(d_input, d_hidden))
        layers.append(nn.ReLU())
        d_input = d_hidden

    d_output = sizes[-1]
    layers.append(nn.Linear(d_input, d_output))  # output layer
    return layers


# TODO add dropout

class ReluNet(nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        layers = relu_seq(sizes)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



def make_leave_one_out(array: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Args:
        array: Tensor of shape (*B, N, *H).
        axis: The axis where N appears.

    Returns:
        Tensor of shape (*B, N, N-1, *H).
    """
    axis = axis % array.ndim
    n = array.size(axis)
    output = []

    for i in range(n):
        array_before = array.narrow(axis, 0, i)
        array_after = array.narrow(axis, i + 1, n - i - 1)
        sliced = torch.cat([array_before, array_after], dim=axis)
        output.append(sliced)

    output = torch.stack(output, dim=axis)
    return output
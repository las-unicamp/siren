import torch
from torch import nn

from src.hyperparameters import args
from src.initialization import original_siren_initialization


class SineLayer(nn.Module):
    """Linear layer followed by the sine activation.

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    bias : bool
        If True, the bias is included.

    is_first : bool
        If True, then it represents the first layer of the network. Note that
        it influences the initialization scheme.

    omega : int
        Hyperparameter. Determines scaling.

    custom_init_function : None or callable
        If None, then we are going to use the `original_siren_initialization`
        defined above. Otherwise, any callable that modifies the `weight`
        parameter in place.

    Attributes
    ----------
    linear : nn.Linear
        Linear layer.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega=30,  # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5
        custom_init_function=None,
    ):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if custom_init_function is None:
            original_siren_initialization(
                self.linear.weight, is_first=is_first, omega=omega
            )
        else:
            custom_init_function(self.linear.weight)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, in_features)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, out_features).
        """
        return torch.sin(self.omega * self.linear(x))


class SIREN(nn.Module):
    """Network composed of SineLayers.

    Parameters
    ----------
    hidden_features : int
        Number of hidden features (each hidden layer the same).

    hidden_layers : int
        Number of hidden layers.

    first_omega, hidden_omega : float
        Hyperparameter influencing scaling.

    custom_init_function : None or callable
        If None, then we are going to use the `original_siren_initialization`
        defined above. Otherwise any callable that modifies the `weight`
        parameter in place.

    Attributes
    ----------
    net : nn.Sequential
        Sequential collection of `SineLayer` and `nn.Linear` at the end.
    """

    def __init__(
        self,
        hidden_features=args.num_nodes_per_layer,
        hidden_layers=args.num_hidden_layers,
        first_omega=args.first_layer_omega,
        hidden_omega=args.hidden_layer_omega,
        custom_init_function=None,
    ):
        super().__init__()
        in_features = 2
        out_features = 1

        net = []
        net.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                custom_init_function=custom_init_function,
                omega=first_omega,
            )
        )

        for _ in range(hidden_layers):
            net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    custom_init_function=custom_init_function,
                    omega=hidden_omega,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)

        if custom_init_function is None:
            original_siren_initialization(
                final_linear.weight, is_first=False, omega=hidden_omega
            )
        else:
            custom_init_function(final_linear.weight)

        net.append(final_linear)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, 2)` representing the 2D pixel coordinates.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, 1)` representing the predicted
            intensities.
        """
        return self.net(x)

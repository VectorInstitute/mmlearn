"""Multi-layer perceptron (MLP)."""

from typing import Callable, List, Optional, Union

import torch
from hydra_zen import store


@store(group="modules/layers", provider="mmlearn")
class MLP(torch.nn.Sequential):
    """Multi-layer perceptron (MLP).

    This module will create a block of Linear -> Normalization -> Activation -> Dropout
    layers.

    Parameters
    ----------
    in_dim : int
        The input dimension.
    out_dim : int, optional, default=None
        The output dimension. If not specified, it is set to `in_dim`.
    hidden_dims : list, optional, default=None
        The dimensions of the hidden layers. The length of the list determines the
        number of hidden layers. This parameter is mutually exclusive with
        `hidden_dims_multiplier`.
    hidden_dims_multiplier : list, optional, default=None
        The multipliers to apply to the input dimension to get the dimensions of
        the hidden layers. The length of the list determines the number of hidden
        layers. The multipliers will be used to get the dimensions of the hidden
        layers. This parameter is mutually exclusive with `hidden_dims`.
    apply_multiplier_to_in_dim : bool, optional, default=False
        Whether to apply the `hidden_dims_multiplier` to `in_dim` to get the
        dimensions of the hidden layers. If `False`, the multipliers will be applied
        to the dimensions of the previous hidden layer, starting from `in_dim`.
        This parameter is only relevant when `hidden_dims_multiplier` is specified.
    norm_layer : callable, optional, default=None
        The normalization layer to use. If not specified, no normalization is used.
        Partial functions can be used to specify the normalization layer with specific
        parameters.
    activation_layer : callable, optional, default=torch.nn.ReLU
        The activation layer to use. If not specified, ReLU is used. Partial functions
        can be used to specify the activation layer with specific parameters.
    bias : bool, optional, default=True
        Whether to use bias in the linear layers.
    dropout : float, optional, default=0.0
        The dropout probability to use.

    """

    def __init__(  # noqa: PLR0912
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        hidden_dims_multiplier: Optional[List[float]] = None,
        apply_multiplier_to_in_dim: bool = False,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        bias: Union[bool, List[bool]] = True,
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:
        """Initialize the MLP module."""
        if hidden_dims is None and hidden_dims_multiplier is None:
            hidden_dims = []
        if hidden_dims is not None and hidden_dims_multiplier is not None:
            raise ValueError(
                "Only one of `hidden_dims` or `hidden_dims_multiplier` must be specified."
            )

        if hidden_dims is None and hidden_dims_multiplier is not None:
            if apply_multiplier_to_in_dim:
                hidden_dims = [
                    int(in_dim * multiplier) for multiplier in hidden_dims_multiplier
                ]
            else:
                hidden_dims = [int(in_dim * hidden_dims_multiplier[0])]
                for multiplier in hidden_dims_multiplier[1:]:
                    hidden_dims.append(int(hidden_dims[-1] * multiplier))

        if isinstance(bias, bool):
            bias_list: List[bool] = [bias] * (len(hidden_dims) + 1)  # type: ignore[arg-type]
        else:
            bias_list = bias
        if len(bias_list) != len(hidden_dims) + 1:  # type: ignore[arg-type]
            raise ValueError(
                "Expected `bias` to be a boolean or a list of booleans with length "
                "equal to the number of linear layers in the MLP."
            )

        if isinstance(dropout, float):
            dropout_list: List[float] = [dropout] * len(hidden_dims)  # type: ignore[arg-type]
        else:
            dropout_list = dropout
        if len(dropout_list) != len(hidden_dims):  # type: ignore[arg-type]
            raise ValueError(
                "Expected `dropout` to be a float or a list of floats with length "
                "equal to the number of dropout layers in the MLP."
            )

        layers = []
        for layer_idx, hidden_dim in enumerate(hidden_dims[:-1]):  # type: ignore[index]
            layers.append(
                torch.nn.Linear(in_dim, hidden_dim, bias=bias_list[layer_idx])
            )
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            if activation_layer is not None:
                layers.append(activation_layer())
            layers.append(torch.nn.Dropout(dropout_list[layer_idx]))
            in_dim = hidden_dim

        if out_dim is None:
            out_dim = in_dim

        layers.append(torch.nn.Linear(in_dim, out_dim, bias=bias_list[-1]))

        super().__init__(*layers)

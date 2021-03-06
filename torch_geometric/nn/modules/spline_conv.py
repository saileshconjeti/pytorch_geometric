import torch
from torch.nn import Module, Parameter

from .utils.inits import uniform
from .utils.repr import repr
from ..functional.spline_conv import spline_conv


def repeat_to(input, dim):
    if not isinstance(input, list):
        input = [input]

    if len(input) > dim:
        raise ValueError()

    if len(input) < dim:
        rest = dim - len(input)
        fill_value = input[len(input) - 1]
        input += [fill_value for _ in range(rest)]

    return input


class SplineConv(Module):
    """Spline-based Convolutional Operator :math:`(f \star g)(i) =
    1/\mathcal{N}(i) \sum_{l=1}^{M_{in}} \sum_{j \in \mathcal{N}(j)}
    f_l(j) \cdot g_l(u(i, j))`, where :math:`g_l` is a kernel function defined
    over the weighted B-Spline tensor product basis for a single input feature
    map.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Size of the convolving kernel.
        is_open_spline (bool or [bool], optional): Whether to use open or
            closed B-spline bases. (default :obj:`True`)
        degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 bias=True):

        super(SplineConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = repeat_to(kernel_size, dim)
        self.register_buffer('kernel_size', torch.LongTensor(self.kernel_size))
        self.is_open_spline = repeat_to(is_open_spline, dim)
        is_open_spline = torch.LongTensor(self.is_open_spline)
        self.register_buffer('is_open_spline', is_open_spline)
        self.degree = degree
        self.K = self._buffers['kernel_size'].prod()

        weight = torch.Tensor(self.K + 1, in_features, out_features)
        self.weight = Parameter(weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_features * (self.K + 1)
        uniform(size, self.weight, self.bias)

    def forward(self, adj, input):
        return spline_conv(
            adj, input, self.weight, self._buffers['kernel_size'],
            self._buffers['is_open_spline'], self.K, self.degree, self.bias)

    def __repr__(self):
        return repr(self, ['kernel_size', 'is_open_spline', 'degree'])

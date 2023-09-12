import torch
import torch.nn as nn
from functools import partial

class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            padding: str = '',
            bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
            se_ratio: float = 0.25, # SE module reduction ratio
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)

        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]
        self.stride = 2

        self.conv1 = create_conv2d(in_chs, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)
        self.norm1 = norm_act_layer(out_chs[0])
        self.conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)
        
        # Squeeze-and-Excitation module
        self.se_reduce = nn.Conv2d(out_chs[1], int(out_chs[1] * se_ratio), kernel_size=1)
        self.se_expand = nn.Conv2d(int(out_chs[1] * se_ratio), out_chs[1], kernel_size=1)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)
        named_apply(partial(_init_conv, scheme=scheme), self.se_reduce, name='weight')
        named_apply(partial(_init_conv, scheme=scheme), self.se_expand, name='weight')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        
        # Squeeze-and-Excitation module
        se_tensor = torch.mean(x, dim=(2, 3), keepdim=True)
        se_tensor = self.se_reduce(se_tensor)
        se_tensor = torch.relu(se_tensor)
        se_tensor = self.se_expand(se_tensor)
        se_tensor = torch.sigmoid(se_tensor)
        x = x * se_tensor
        
        return x

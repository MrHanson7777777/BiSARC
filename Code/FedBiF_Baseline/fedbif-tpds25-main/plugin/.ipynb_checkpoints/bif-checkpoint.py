import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import Bottleneck, BasicBlock
#from model.densenet import DenseBottleneck, Transition, DenseBlock
from torch.nn.modules.container import Sequential
class DenseBottleneck: pass
class Transition: pass
class DenseBlock: pass

def stochastic_quantize(weight, bit, bit_pos):
    max_range = weight.abs().max()

    alpha = max_range / (2**(bit-1))
    int_weight = torch.floor((weight+max_range)/alpha+torch.rand_like(weight)).clamp(0, 2**bit - 1)
    int_weight = int_weight.to(torch.int32)
    
    mask = 1 << bit_pos
    bit_values = (int_weight & mask) >> bit_pos
    freeze_weight = int_weight - (bit_values * mask)

    remain_weight = (freeze_weight * alpha) - max_range
    virtual = torch.empty_like(weight).uniform_(-1e-3*mask, 1e-3*mask)
    virtual = virtual.abs() * (bit_values*2-1)
    virtual_alpha = alpha * mask
    
    return remain_weight, virtual, virtual_alpha 

def virtual_to_binary(vitual, alpha):
    binary = torch.sign(torch.relu(vitual)) * alpha
    return vitual - vitual.detach() + binary.detach()

class BiF_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BiF_Linear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.zeros((self.out_features, self.in_features)), requires_grad=False)
        nn.init.normal_(self.weight, mean=0, std=0.01)

        self.virtual = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
    def before_client_send(self):
        with torch.no_grad():
            update = virtual_to_binary(self.virtual, self.alpha)
            self.weight.data = self.weight.data + update
            self.virtual.data = torch.zeros_like(self.weight)
            self.alpha.data = torch.tensor(0.0)
    
    def before_server_send(self, bit, freeze_bit):
        with torch.no_grad():
            weight, virtual, alpha = stochastic_quantize(self.weight, bit, freeze_bit)
            self.weight.data = weight
            self.virtual.data = virtual
            self.alpha.data = alpha

    def forward(self, x):
        update = virtual_to_binary(self.virtual, self.alpha)
        return F.linear(x, self.weight+update, None)

class BiF_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BiF_Conv2d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)), requires_grad=False)
        nn.init.normal_(self.weight, mean=0, std=0.01)

        self.virtual = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
    def before_client_send(self):
        with torch.no_grad():
            update = virtual_to_binary(self.virtual, self.alpha)
            self.weight.data = self.weight.data + update
            self.virtual.data = torch.zeros_like(self.weight)
            self.alpha.data = torch.tensor(0.0)
    
    def before_server_send(self, bit, freeze_bit):
        with torch.no_grad():
            weight, virtual, alpha = stochastic_quantize(self.weight, bit, freeze_bit)
            self.weight.data = weight
            self.virtual.data = virtual
            self.alpha.data = alpha

    def forward(self, x):
        update = virtual_to_binary(self.virtual, self.alpha)
        return F.conv2d(x, self.weight + update, None, self.stride, self.padding)


def bif_replace_modules(model):
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, BiF_Conv2d(module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.padding))
        elif isinstance(module, nn.Linear):
            setattr(model, name, BiF_Linear(module.in_features, module.out_features))
        if isinstance(module, (Sequential, Bottleneck, BasicBlock, DenseBottleneck, Transition, DenseBlock)):
            bif_replace_modules(module)

def bif_before_client_send(model):
    for name, module in model._modules.items():
        if isinstance(module, (BiF_Conv2d, BiF_Linear)):
            module.before_client_send()
        if isinstance(module, (Sequential, Bottleneck, BasicBlock, DenseBottleneck, Transition, DenseBlock)):
            bif_before_client_send(module)

def bif_before_server_send(model, bit, freeze_bit):
    for name, module in model._modules.items():
        if isinstance(module, (BiF_Conv2d, BiF_Linear)):
            module.before_server_send(bit, freeze_bit)
        if isinstance(module, (Sequential, Bottleneck, BasicBlock, DenseBottleneck, Transition, DenseBlock)):
            bif_before_server_send(module, bit, freeze_bit)


if __name__ == "__main__":
    weight = 0.01 * torch.tensor([-15, -8, -4, 0, 1, 3, 7])
    weight_q = stochastic_quantize(weight, 4, 2)
    print(weight_q == weight)

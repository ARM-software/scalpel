from __future__ import print_function
import torch.nn as nn
import torch
from torch.autograd import Variable

class MaskLayer_F(torch.autograd.Function):
    def forward(self, input, alpha, beta):
        positive = beta.gt(0.8001)
        negative = beta.lt(0.800)
        alpha[positive] = 1.0
        alpha[negative] = 0.0
        # save alpha after we modify it
        self.save_for_backward(input, alpha)
        if not (alpha.eq(1.0).sum() + alpha.eq(0.0).sum() == alpha.nelement()):
            print('ERROR: Please set the weight decay and lr of alpha to 0.0')
        if len(input.shape) == 4:
            input = input.mul(alpha.unsqueeze(2).unsqueeze(3))
        else:
            input = input.mul(alpha)
        return input

    def backward(self, grad_output):
        input, alpha = self.saved_variables
        grad_input = grad_output.clone()
        if len(input.shape) == 4:
            grad_input = grad_input.mul(alpha.data.unsqueeze(2).unsqueeze(3))
        else:
            grad_input = grad_input.mul(alpha.data)

        grad_beta = grad_output.clone()
        grad_beta = grad_beta.mul(input.data).sum(0, keepdim=True)
        if len(grad_beta.shape) == 4:
            grad_beta = grad_beta.sum(3).sum(2)
        return grad_input, None, grad_beta

class MaskLayer(nn.Module):
    def __init__(self, size=-1, conv=False, beta_initial=0.8002, beta_limit=0.802):
        assert(size>0)
        super(MaskLayer, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1, size).zero_().add(1.0))
        self.beta = nn.Parameter(torch.FloatTensor(1, size).zero_().add(beta_initial))
        self.beta_limit = beta_limit
        self.conv = conv
        return

    def forward(self, x):
        self.beta.data.clamp_(0.0, self.beta_limit)
        x = MaskLayer_F()(x, self.alpha, self.beta)
        return x

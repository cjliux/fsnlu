#coding: utf-8
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        # ctx.saved_for_backward = [input, parameters]
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # input, parameters = ctx.saved_for_backward
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)


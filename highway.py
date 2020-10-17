#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):

    ### YOUR CODE HERE for part 1f

    def __init__(self, input_size):
        super(Highway, self).__init__()

        self.input_size = input_size
        self.w_proj = nn.Linear(input_size, input_size, bias=True)
        self.w_gate = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """

        @param x_conv_out: torch tensor with size (batch_size, input_size)
        @return: x_highway: torch tensor with size (batch_size, input_size) s.t highway module
        applied on it.
        @rtype: torch.Tensor
        """
        x_proj = nn.functional.relu(self.w_proj(x_conv_out))  # (b,input_size)
        x_gate = torch.sigmoid(self.w_gate(x_conv_out))  # (b,input_size)
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out  # (b,input_size)

        return x_highway

    ### END YOUR CODE

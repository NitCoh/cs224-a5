#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):

    ### YOUR CODE HERE for part 1g


    def __init__(self, char_embed_size,word_embed_size, kernel_size=5, padding=1):
        super(CNN, self).__init__()

        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.kernel_size = kernel_size
        self.padding = padding
        # input shape (b, char_embed_size, max_word_len)
        self.conv1 = nn.Conv1d(in_channels=self.char_embed_size, out_channels=self.word_embed_size, kernel_size=self.kernel_size, padding=padding)


    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """

        @param x_reshaped: tensor of size (batch_size, e_char, m_word)
                where e_char = char embed dim and m_word is max length of word in batch

        @return:x_conv_out : tensor of size (batch_size, word_embed_size)
        """
        max_word_len = x_reshaped.shape[-1]
        x_conv = self.conv1(x_reshaped)
        # input shape (b, embed_size, m + 2*padding - k + 1)
        max_pooling = nn.MaxPool1d(max_word_len + 2*self.padding - self.kernel_size + 1)  # stride taken to be 1
        x_conv_out = max_pooling(nn.functional.relu(x_conv)).squeeze()
        return x_conv_out


    ### END YOUR CODE


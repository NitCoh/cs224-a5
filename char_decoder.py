#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        embeds = self.decoderCharEmb(input)  # (length, batch_size, e_char)
        hiddens, dec_hidden = self.charDecoder(embeds, dec_hidden)  # hiddens: (l, b, (directions=1) * h), dec_hidden:  tuple(tensor(1 * 1, b, h))
        S_t = self.char_output_projection(hiddens)  # (length, batch_size, self.vocab_size)

        return S_t, dec_hidden

        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        S_t, dec_hidden = self.forward(input=char_sequence[:-1], dec_hidden=dec_hidden) # S_t: (length, batch_size, self.vocab_size)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char_pad, reduction='sum')
        # We need to stack all the words together in one 2d tensor instead of 3d tensor, divided to batches of words (b, l, vocab_size)
        S_t_batched = S_t.view(-1, len(self.target_vocab.char2id))  # (batch * length, char_vocab_size)
        targets = char_sequence[1:].contiguous().view(-1)  # (length-1) * batch_size
        loss = loss_fn(S_t_batched, targets)  # tensor(scalar)
        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        _, batch_size, hidden_size = initialStates[0].shape
        decodedWords = [""] * batch_size
        current_char_idx = self.target_vocab.start_of_word
        current_char_idx = torch.tensor([current_char_idx] * batch_size, device=device).unsqueeze(0)  #(1, batch_size)
        h_prev, c_prev = initialStates
        for t in range(max_length):

            _, (h_new, c_new) = self.forward(current_char_idx, (h_prev, c_prev))  # h_new shape: (1, batch, hidden_size)
            scores = self.char_output_projection(h_new.squeeze(0))  # (batch, vocab_size)
            probs = torch.softmax(scores, dim=1)
            indices = torch.argmax(probs, dim=1).tolist()  # size: batch
            current_char_idx = torch.tensor(indices, device=device).unsqueeze(0)  # (1, batch_size)
            h_prev, c_prev = h_new, c_new
            chars = [self.target_vocab.id2char[i] for i in indices]   # size: batch
            decodedWords = [cur_string+c for c, cur_string in zip(chars, decodedWords)]  # size: batch

        # truncate
        decodedWords = [x.partition("}")[0] for x in decodedWords]

        return decodedWords
        ### END YOUR CODE


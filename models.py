#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""
Hierarchical Attention model for NLP
"""



#=========================================================================================================
#================================ 0. MODULE



import torch
from torch import nn


device = 'cuda:0'



#=========================================================================================================
#================================ 1. ATTENTIONLAYER



class DeepAttention(nn.Module):
    """
    Attention Network, used to help the decoder to focus on the most informative
    part of the sentence or the document at each step.
    """
    def __init__(self, dimension, attention_dim):
        """
        Arguments:
        ----------
        dimension: feature size of input images
        attention_dim: size of the attention network
        """
        super(Attention, self).__init__()

        self.attention = nn.Linear(dimension, attention_dim)  # linear layer to transform the input
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights


    def forward(self, input_sequence):
        """
        Forward propagation

        Arguments:
        ----------
        input_sequence: output of the bi_gru, tensor of dimension (batch_size, length_sequence, dimension)
        
        Return: 
        -------
        attention weighted encoding: tensor of shape (batch, dimension)
        """
        attention = self.attention(input_sequence)                   # (batch_size, length_sequence, attention_dim)
        attention = self.full_att(self.relu(attention))              # (batch_size, length_sequence, 1)
        weights = self.softmax(attention.squeeze(2))                 # (batch_size, length_sequence)

        attention_weighted_encoding = (input_sequence * weights.unsqueeze(2)).sum(dim=1)  # (batch_size, dimension)

        return attention_weighted_encoding



class Attention(nn.Module):
    """
    Attention Network, used to help the decoder to focus on the most informative
    part of the sentence or the document at each step.
    """
    def __init__(self, dimension):
        """
        Arguments:
        ----------
        dimension: feature size of input images
        """
        super(Attention, self).__init__()

        self.attention = nn.Linear(dimension, 1)  # linear layer to transform the input
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)          # softmax layer to calculate weights


    def forward(self, input_sequence):
        """
        Forward propagation

        Arguments:
        ----------
        input_sequence: output of the bi_gru, tensor of dimension (batch_size, length_sequence, dimension)
        
        Return: 
        -------
        attention weighted encoding: tensor of shape (batch, dimension)
        """
        attention = self.relu(self.attention(input_sequence))        # (batch_size, length_sequence, 1)
        weights = self.softmax(attention.squeeze(2))                 # (batch_size, length_sequence)

        attention_weighted_encoding = (input_sequence * weights.unsqueeze(2)).sum(dim=1)  # (batch_size, dimension)

        return attention_weighted_encoding



#=========================================================================================================
#================================ 2. HAN



class HAN(nn.Module):
    """
    """
    def __init__(self, attention_dim=[128, 128], bi_gru_dim=[128, 128], embedding_dim=13, vocab_size=5000, dropout=0.3):
        super(HAN, self).__init__()

        # Parameters
        self.attention_dim1 = attention_dim[0]
        self.attention_dim2 = attention_dim[1]

        self.bi_gru_dim1 = bi_gru_dim[0]
        self.bi_gru_dim2 = bi_gru_dim[1]

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # Modules
        self.dropout = nn.Dropout(p=self.dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.bi_gru1 = nn.GRU(input_size=self.embedding_dim, hidden_size=self.bi_gru_dim1,
                              bias=True, batch_first=True, bidirectional=True)
        # self.attention1 = DeepAttention(2 * self.bi_gru_dim1, self.attention_dim1)
        self.attention1 = Attention(2 * self.bi_gru_dim1)

        self.bi_gru2 = nn.GRU(input_size=2 * self.bi_gru_dim1, hidden_size=self.bi_gru_dim2,
                              bias=True, batch_first=True, bidirectional=True)
        # self.attention2 = DeepAttention(2 * self.bi_gru_dim2, self.attention_dim2)
        self.attention2 = Attention(2 * self.bi_gru_dim2)

        self.regressor = nn.Linear(2 * self.bi_gru_dim2, 1)
        self.activation = nn.Tanh()


    def load_pretrained_embeddings(self, embeddings, fine_tune=True):
        """
        Loads embedding layer with pre-trained embeddings.
        
        Arguments:
        ----------
        embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

        if not fine_tune:
            for p in self.embedding.parameters():
                p.requires_grad = False


    def forward(self, documents):
        """
        Arguments:
        ----------
        documents: tensor of shape (batch, sentences, words)
        """
        n_documents = documents.size(0)
        n_sentences = documents.size(1)
        device = documents.device

        ## Embedding
        embeddings = self.embedding(documents)  # (batch, sentences, words, embedding_dim)

        ## Sentence level encoding
        attention_weighted_sentences = []

        for sentences in embeddings:
            output, _ = self.bi_gru1(sentences)
            weighted_output = self.attention1(output)
            attention_weighted_sentences.append(weighted_output.unsqueeze(0))

        attention_weighted_sentences = torch.cat(attention_weighted_sentences, 0)
        attention_weighted_sentences = self.dropout(attention_weighted_sentences)

        ## Document level encoding
        output, _ = self.bi_gru2(attention_weighted_sentences)
        attention_weighted_documents = self.attention2(output)
        attention_weighted_documents = self.dropout(attention_weighted_documents)

        ## Final prediction
        out = self.regressor(attention_weighted_documents)
        out = 2 * self.activation(out)
    
        return out
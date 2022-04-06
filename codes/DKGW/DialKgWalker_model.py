from __future__ import absolute_import, division, print_function

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from torch.distributions import Categorical

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = 2*dim
        self.da = dim
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Linear(self.dim, self.da, bias=True)
        self.b = nn.Linear(self.da, 1, bias=False)

    def forward(self, h, mask):
        f = F.tanh(self.a(h))
        s = self.b(f).squeeze(dim=-1)
        s[~mask] = -1e9
        attention = F.softmax(s, dim=-1)
        attention = attention.unsqueeze(dim=2)
        h_bar = h*attention
        h_bar = torch.sum(h_bar, dim=1)
        return h_bar

class ModalityAttentionLayer(nn.Module):
    def __init__(self, opt):
        super(ModalityAttentionLayer, self).__init__()
        self.in_dim = 2*opt["hidden_dim"]
        self.out_dim = 1
        self.linear = nn.Linear(in_features=self.in_dim, out_features=self.out_dim, bias=True)
        self.ent_proj = nn.Linear(in_features = self.in_dim//2, out_features = self.in_dim)

    def forward(self, x_e, x_s, x_d):
        # Equation 2
        x_e = self.ent_proj(x_e)
        inp = torch.stack([x_e, x_s, x_d], dim=1)
        x = self.linear(inp).squeeze(2)
        a = F.tanh(x)
        alpha_m = F.softmax(a, dim=1).unsqueeze(-1)
        x_bar = alpha_m * inp
        x_bar = torch.sum(x_bar, dim=1)

        return x_bar

class SentenceEncoder(nn.Module):
    def __init__(self, opt):
        super(SentenceEncoder, self).__init__()
        self.in_dim = opt["word_dim"]
        self.hidden_size = opt["hidden_dim"]
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.self_attention = SelfAttentionLayer(self.hidden_size)

    def forward(self, word_embeddings, mask):
        lstm_embeddings, _ = self.lstm(word_embeddings)
        lstm_embeddings, _ = pad_packed_sequence(lstm_embeddings)
        lstm_embeddings = lstm_embeddings.permute(1, 0, 2)
        sentence_embedding = self.self_attention(lstm_embeddings, mask)
        return sentence_embedding

class DialogueEncoder(nn.Module):
    def __init__(self, opt):
        super(DialogueEncoder, self).__init__()
        self.in_dim = opt["word_dim"]
        self.hidden_size = opt["hidden_dim"]
        self.s_encoder = SentenceEncoder(opt)
        self.lstm = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.heirarchical_attention = SelfAttentionLayer(self.hidden_size)

    def forward(self, dialogue_packed, mask):
        dialogue_embedding = []
        for i in range(len(dialogue_packed)):
            d_encoded = self.s_encoder(dialogue_packed[i], mask[i])
            dialogue_embedding.append(d_encoded)

        dialogue = torch.stack(dialogue_embedding, dim=1) #Bx3xD
        dialogue_embedding, cc = self.lstm(dialogue)
        r, c, _ = dialogue_embedding.size()
        mask = torch.ones(r, c)==1

        dialogue_embedding = self.heirarchical_attention(dialogue_embedding, mask)

        return dialogue_embedding

class KGPathWalker(nn.Module):
    def __init__(self, opt):
        super(KGPathWalker, self).__init__()
        self.dim = opt["hidden_dim"]
        # self.seq_len = opt["seq_len"]
        self.n_rels = opt["n_relation"]
        self.W_x = nn.Linear(in_features=2*self.dim, out_features=self.dim)
        self.W_hi = nn.Linear(in_features=self.dim, out_features=1)
        self.W_ci = nn.Linear(in_features=self.dim, out_features=1)
        self.W_zc = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.W_hc = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.W_zo = nn.Linear(in_features=self.dim, out_features=1)
        self.W_ho = nn.Linear(in_features=self.dim, out_features=1)
        self.W_co = nn.Linear(in_features=self.dim, out_features=1)
        self.W_ha = nn.Linear(in_features=self.dim, out_features=self.n_rels)
        self.W_xa = nn.Linear(in_features=self.dim, out_features=self.n_rels)
    
    def forward(self, x_bar, relations, h_t, c_t):
        # Equation 7
        x_bar = self.W_x(x_bar)
        alpha_t = F.sigmoid(self.W_ha(h_t) + self.W_xa(x_bar))
        r_t = torch.matmul(alpha_t, relations)
        z_t = h_t + r_t

        # Equation 6
        i_t = F.sigmoid(self.W_hi(h_t) + self.W_ci(c_t))
        c_t = (1 - i_t)*(c_t) + (i_t)*(F.tanh(self.W_zc(z_t) + self.W_hc(h_t)))
        o_t = F.sigmoid(self.W_zo(z_t) + self.W_ho(h_t) + self.W_co(c_t))
        h_t = o_t * F.tanh(c_t)

        return h_t, c_t, r_t
        

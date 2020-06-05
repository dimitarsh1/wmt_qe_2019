import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.optim as optim
import numpy as np

class SoftDotAttention(nn.Module):
    """ Soft Dot Attention.
        Ref: http://www.aclweb.org/anthology/D15-1166
        Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """ Initialize layer.

        :param dim: Dimmension
        """
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.mask = None

    def forward(self, y, h):
        """Propogates input through the network.

        :param y: batch of sentences, T x batch x dim
        :param h: the hiddent states, batch x dim
        """
        y = y.transpose(1, 0)

        t = self.linear_in(h)
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(y, target).squeeze(2)  # batch x T
        attn = F.softmax(attn, dim=1)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x T

        weighted_y = torch.bmm(attn3, y).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_y, h), 1)

        h_tilde = torch.tanh(self.linear_out(h_tilde))

        return h_tilde, attn

class RTEAttention(nn.Module):
    """ Word by Word attention.
        Ref: https://arxiv.org/pdf/1509.06664.pdf
    """

    def __init__(self, dim, device):
        """ Initialize network.

        :param dim: Dimmension
        :param device: the device to run the training/testing on
        """
        super(RTEAttention, self).__init__()

        self.dim = dim
        self.device = device

        # Attention Parameters
        self.W_y = nn.Parameter(torch.randn(self.dim, self.dim).to(self.device))
        self.W_h = nn.Parameter(torch.randn(self.dim, self.dim).to(self.device))
        self.W_r = nn.Parameter(torch.randn(self.dim, self.dim).to(self.device))
        self.W_alpha = nn.Parameter(torch.randn(self.dim, 1).to(self.device))

        # Final combination Parameters
        self.W_x = nn.Parameter(torch.randn(self.dim, self.dim).to(self.device))  # dim x dim
        self.W_p = nn.Parameter(torch.randn(self.dim, self.dim).to(self.device))  # dim x dim

        self.register_parameter('W_y', self.W_y)
        self.register_parameter('W_h', self.W_h)
        self.register_parameter('W_r', self.W_r)
        self.register_parameter('W_alpha', self.W_alpha)
        self.register_parameter('W_x', self.W_x)
        self.register_parameter('W_p', self.W_p)

    def forward(self, y, mask_y, h):
        """ Computes the attention weights over y using h
            Returns an attention weighted representation of y, and the alphas

        :param y: The input of sentences, T x batch x dim
        :param mask_y: Mask for the input, T x batch
        :param h: Hidden states, batch x dim
        :returns: r, batch x dim
                  alpha, batch x T
        """
        y = y.transpose(1, 0)  # batch x T x dim

        mask_y = mask_y.transpose(1, 0)  # batch x T
        Wy = torch.bmm(y, self.W_y.unsqueeze(0).expand(y.size(0), *self.W_y.size()))  # batch x T x dim
        Wh = torch.mm(h, self.W_h)  # batch x dim

        M = torch.tanh(Wy + Wh.unsqueeze(1).expand(Wh.size(0), y.size(1), Wh.size(1)))  # batch x T x dim
        alpha = torch.bmm(M, self.W_alpha.unsqueeze(0).expand(y.size(0), *self.W_alpha.size())).squeeze(-1)  # batch x T

        alpha = alpha + (-1000.0 * (1. - mask_y))  # To ensure probability mass doesn't fall on non tokens
        alpha = F.softmax(alpha, dim=1)
        r = torch.bmm(alpha.unsqueeze(1), y).squeeze(1)  # batch x dim

        h_star = self.combine_last(r, h)

        return h_star, alpha

    def combine_last(self, r, hidden):
        """ Combining two matrixes

        :param r: r, batch x dim
        :param hidden: hidden states, batch x dim
        :returns: the tanh transformation of the combined matrixes
        """
        W_p_r = torch.mm(r, self.W_p)
        W_x_h = torch.mm(hidden, self.W_x)
        h_star = torch.tanh(W_p_r + W_x_h)

        return h_star


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def where(self, cond, x_1, x_2):
        cond = cond.float()
        return (cond * x_1) + ((1-cond) * x_2)

    def forward(self, similarity, label):
        ''' Computes the contrastive loss based on a similarity measure

        :param similarity: the similarity score
        :param label: the label towards which it should compare (the similarity)
        :returns: the contrastive loss
        '''
        loss_contrastive = torch.mean((1.0-label) * self.where(similarity < self.margin, torch.pow(similarity, 2), 0) + 
                                      (label) * 0.25 * torch.pow((1.0 - similarity), 2))
        return loss_contrastive

#    def forward(self, similarity, label):
#        ''' Computes the contrastive loss based on a similarity measure

#        :param similarity: the similarity score
#        :param label: the label towards which it should compare (the similarity)
#        :returns: the contrastive loss
#        '''
#        loss_contrastive = torch.mean((1.0-label) * torch.pow(similarity, 2) +
#                                      (label) * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))
#        return loss_contrastive



class SiameseSimilarity(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dict_size, batch_size, metric='eucledian', attention_type='dot', device=torch.device("cpu")):
        ''' Init method for the NN - bidirectional LSTM

        :param embedding_dim: The embedding dimension
        :param hidden_dim: the dimmention for the hidden states
        :param dict_size: the dictionary size
        :param batch_size: the batch size
        :param metric: the metric used for the similarity; default is Eucledian (options are Eucledian, Manhattan, Cosine similarity)
        :param device: the device to run the training/test; default is -1 = CPU
        '''
        super(SiameseSimilarity, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device
        self.metric = metric

        self.dropout = nn.Dropout(0.2) # Add a variable

        self.attention_type = attention_type
        if self.attention_type == 'dot':
            self.attention_left = SoftDotAttention(hidden_dim)
            self.attention_right = SoftDotAttention(hidden_dim)
        elif self.attention_type == 'rte':
            self.attention_left = RTEAttention(hidden_dim, device)
            self.attention_right = RTEAttention(hidden_dim, device)
        else:
            self.attention_left = None
            self.attention_right = None

        self.word_embeddings = nn.Embedding(dict_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, num_layers=2, dropout=0.2)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
        self.fc1 = nn.Sequential(
                    nn.Linear(batch_size * hidden_dim // 2, 100),
                    nn.ReLU(inplace=True),
                    nn.Linear(100, 10))
        self.fc2 = nn.Sequential(
                    nn.Linear(batch_size * hidden_dim // 2, 100),
                    nn.ReLU(inplace=True),
                    nn.Linear(100, 10))
        self.hidden2label = nn.Linear(hidden_dim // 2, 2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        ''' initializes the hidden states

        :returns: first hidden and first cell
        '''
        # the first is the hidden h            sim = F.pairwise_distance(h1, h2)
        # the second is the cell  c
        layers = 2

        h0 = torch.zeros(1 * 2 * layers, self.batch_size, self.hidden_dim // 2, requires_grad=True).to(self.device)
        c0 = torch.zeros(1 * 2 * layers, self.batch_size, self.hidden_dim // 2, requires_grad=True).to(self.device)

        return (h0, c0)


    def forward_once(self, sentence_embs):
        first_dim = sentence_embs.shape[0]
        x = sentence_embs.view(first_dim, self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        return lstm_out, self.hidden, mask

    def forward(self, sentence_left, sentence_right):
        ''' Forward method for the Siamese architecture

        :param sentence_left: the first sentence to compare
        :param sentence_right: the second sentence to compare
        :returns: output of the network for the first and the second sentence, as well as the similarity
        '''
        lstm_out_left, hidden_left, mask_left = self.forward_once(sentence_left)
        lstm_out_right, hidden_right, mask_right = self.forward_once(sentence_right)

        #mask_left = torch.ne(sentence_left, 0).type(torch.float32)
        #mask_right = torch.ne(sentence_right, 0).type(torch.float32)

        h_left = hidden_left[0] #view(self.batch_size, -1)
        h_right = hidden_right[0] #view(self.batch_size, -1)

        #h1 = self.hidden2label(h1)
        #h2 = self.hidden2label(h2)
        if self.attention_type == 'rte':
            h_left_star, alpha_left_vec = self.attention_left.forward(lstm_out_left, mask_left, lstm_out_right[-1])
            h_right_star, alpha_right_vec = self.attention_right.forward(lstm_out_right, mask_right, lstm_out_left[-1])
        elif self.attention_type == 'dot':
            h_left_star, alpha_left_vec = self.attention_left.forward(lstm_out_left, lstm_out_right[-1])
            h_right_star, alpha_right_vec = self.attention_right.forward(lstm_out_right, lstm_out_left[-1])
        else:
            h_left_star = h_left.view(h_left.shape[1], -1)
            h_right_star = h_right.view(h_right.shape[1], -1)

        if self.metric == 'manhattan':
            n = -torch.norm((h_left_star - h_right_star), self.batch_size, 1)
            sim = torch.exp(n)
        elif self.metric == 'eucledian':
            sim = F.pairwise_distance(h_left_star, h_right_star)
        elif self.metric == 'cosine':
            sim = F.cosine_similarity(h_left_star, h_right_star)

        return lstm_out_left[-1], lstm_out_right[-1], sim


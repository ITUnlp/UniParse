from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniparse.types import Parser
from uniparse.decoders import eisner

from uniparse.backend.pytorch_backend import _PytorchLossFunctions


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Set initial cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        gpu = next(self.parameters()).is_cuda
        if gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()

        out, _ = self.lstm(x.cuda() if gpu else x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        return out


class Kiperwasser(nn.Module, Parser):
    def save_to_file(self, filename):
        torch.save(self.state_dict(), filename)

    def load_from_file(self, filename):
        self.load_state_dict(torch.load(filename))

    def __init__(self, vocab):
        super().__init__()

        upos_dim = 25
        word_dim = 100
        hidden_dim = 100
        bilstm_out = (word_dim+upos_dim) * 2

        self.word_count = vocab.vocab_size
        self.upos_count = vocab.upos_size
        self.i2c = defaultdict(int, vocab.wordid2freq)
        self.label_count = vocab.label_count
        self._vocab = vocab

        self.wlookup = nn.Embedding(self.word_count, word_dim)
        self.tlookup = nn.Embedding(self.word_count, upos_dim)

        self.deep_bilstm = BiRNN(word_dim+upos_dim, word_dim+upos_dim, 2)

        # edge encoding
        self.edge_head = nn.Linear(bilstm_out, hidden_dim)
        self.edge_modi = nn.Linear(bilstm_out, hidden_dim, bias=True)

        # edge scoring
        self.e_scorer = nn.Linear(hidden_dim, 1, bias=True)

        # rel encoding
        self.label_head = nn.Linear(bilstm_out, hidden_dim)
        self.label_modi = nn.Linear(bilstm_out, hidden_dim, bias=True)

        # label scoring
        self.l_scorer = nn.Linear(hidden_dim, vocab.label_count, bias=True)

        self._loss = _PytorchLossFunctions()
        

    @staticmethod
    def _propability_map(matrix, dictionary):
        return np.vectorize(dictionary.__getitem__)(matrix)

    def forward(self, x):
        (word_ids, upos_ids), (gold_arcs, gold_rels) = x
        
        mask = np.greater(word_ids, self._vocab.ROOT)        

        batch_size, n = word_ids.shape

        target_arcs = gold_arcs
        is_train = gold_arcs is not None
        gpu = next(self.parameters()).is_cuda

        if is_train:
            c = self._propability_map(word_ids, self.i2c)
            drop_mask = np.greater(0.25/(c+0.25), np.random.rand(*word_ids.shape))
            word_ids = np.where(drop_mask, self._vocab.OOV, word_ids)  # replace with UNK / OOV

        word_id_tensor = torch.LongTensor(word_ids)
        upos_id_tensor = torch.LongTensor(upos_ids)

        if gpu:
            word_id_tensor = word_id_tensor.cuda()
            upos_id_tensor = upos_id_tensor.cuda()

        word_embs = self.wlookup(word_id_tensor)
        upos_embs = self.tlookup(upos_id_tensor)

        words = torch.cat([word_embs, upos_embs], dim=-1)

        word_exprs = self.deep_bilstm(words)

        word_h = self.edge_head(word_exprs)
        word_m = self.edge_modi(word_exprs)

        arc_score_list = []
        for i in range(n):
            modifier_i = word_h[:, i, None, :] + word_m  # we would like have head major
            modifier_i = torch.tanh(modifier_i)
            modifier_i_scores = self.e_scorer(modifier_i)
            arc_score_list.append(modifier_i_scores)

        arc_scores = torch.stack(arc_score_list, dim=1)
        arc_scores = arc_scores.view(batch_size, n, n)

        # Loss augmented inference
        if is_train:
            target_arcs[:, 0] = 0  # this guy contains negatives.. watch out for that ....
            margin = np.ones((batch_size, n, n))
            for bi in range(batch_size):
                for m in range(n):
                    h = target_arcs[bi, m]
                    margin[bi, m, h] -= 1
            margin_tensor = torch.Tensor(margin).cuda() if gpu else torch.Tensor(margin)
            arc_scores = arc_scores + margin_tensor

        # since we are major
        decoding_scores = arc_scores.transpose(1, 2).cpu().data.numpy().astype(np.float64)
        parsed_trees = np.array([eisner(s) for s in decoding_scores])

        tree_for_rels = target_arcs if is_train else parsed_trees
        tree_for_rels[:, 0] = 0
        batch_indicies = np.repeat(np.arange(batch_size), n)  # 0, 0, 0, 0, 1, 1 ... etc
        pred_tree_tensor = tree_for_rels.reshape(-1)

        rel_heads = word_exprs[batch_indicies, pred_tree_tensor, :]
        rel_heads = self.label_head(rel_heads).view((batch_size, n, -1))
        rel_modifiers = self.label_modi(word_exprs)

        rel_arcs = torch.tanh(rel_modifiers + rel_heads)

        rel_scores = self.l_scorer(rel_arcs)
        predicted_rels = rel_scores.argmax(-1).cpu().data.numpy()

        loss = None
        if is_train:
            arc_loss = self._loss.hinge(arc_scores.cpu(), parsed_trees, gold_arcs, mask)
            rel_loss = self._loss.hinge(rel_scores.cpu(), predicted_rels, gold_rels, mask)

            loss = arc_loss + rel_loss
        
        return parsed_trees, predicted_rels, loss



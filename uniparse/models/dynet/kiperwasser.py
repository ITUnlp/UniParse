"""Reimplementation of Kiperwasser and Goldbergs (2016)."""

from collections import defaultdict

import dynet as dy
import numpy as np

from uniparse.backend.dynet_backend import kiperwasser_loss, hinge
from uniparse.decoders import eisner


def Dense(model_parameters, input_dim, hidden_dim, activation, use_bias):
    """High level dense layer implementation."""
    w = model_parameters.add_parameters((hidden_dim, input_dim))
    b = model_parameters.add_parameters((hidden_dim, )) if use_bias else None

    def call(xs):
        output = w * xs
        if use_bias:
            output = output + b
        if activation:
            return activation(output)

        return output

    def apply(xs):
        if isinstance(xs, list):
            return [call(x) for x in xs]

        else:
            return call(xs)

    return apply


class Kiperwasser:
    """Reimplementation of Kiperwasser and Goldbergs (2016)."""

    def __init__(self, vocab):
        """Initialize parser parameters based on settings in the vocabulary object."""
        params = dy.ParameterCollection()

        upos_dim = 25
        word_dim = 100
        hidden_dim = 100
        bilstm_out = (word_dim + upos_dim) * 2

        lookup = defaultdict(int, vocab._id2freq).__getitem__
        self.freq_map = np.vectorize(lookup)

        self.wlookup = params.add_lookup_parameters((vocab.vocab_size, word_dim))
        self.tlookup = params.add_lookup_parameters((vocab.upos_size, upos_dim))

        self.encode_rnn = dy.BiRNNBuilder(2, word_dim + upos_dim, bilstm_out, params, dy.VanillaLSTMBuilder)

        # edge encoding
        self.edge_head = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.edge_modi = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.edge_bias = params.add_parameters((hidden_dim, ))

        # edge scoring
        self.e_scorer = Dense(params, hidden_dim, 1, activation=None, use_bias=True)

        # rel encoding
        self.label_head = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.label_modi = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.label_bias = params.add_parameters((hidden_dim, ))

        # label scoring
        self.l_scorer = Dense(params, hidden_dim, vocab.label_count, activation=None, use_bias=True)
        self.params = params
        self._vocab = vocab

    def _encode(self, token_ids, tag_ids, train):
        if train:
            c = self.freq_map(token_ids)
            drop_mask = np.greater(0.25 / (c + 0.25), np.random.rand(*token_ids.shape))
            token_ids = np.where(drop_mask, self._vocab.OOV, token_ids)

        # encode and contextualize
        word_embs = [dy.lookup_batch(self.wlookup, t) for t in token_ids.T]
        upos_embs = [dy.lookup_batch(self.tlookup, t) for t in tag_ids.T]
        words = [dy.concatenate([w, p]) for w, p in zip(word_embs, upos_embs)]

        contextualized_words = self.encode_rnn.transduce(words)

        return contextualized_words

    def _augmented_scores(self, scores, y, batch_size, n):
        margin = np.ones((n, n, batch_size))
        for bi in range(batch_size):
            for m in range(n):
                h = y[bi, m]
                margin[h, m, bi] -= 1

        margin_tensor = dy.inputTensor(margin, batched=True)
        scores = scores + margin_tensor

        return scores

    def _score_edges(self, word_exprs, batch_size, n):
        word_h = self.edge_head(word_exprs)
        word_m = self.edge_modi(word_exprs)

        arc_edges = [dy.tanh(head + modifier + self.edge_bias) for modifier in word_m for head in word_h]

        # edges scoring
        arc_scores = self.e_scorer(arc_edges)
        arc_scores = dy.concatenate_cols(arc_scores)
        arc_scores = dy.reshape(arc_scores, d=(n, n), batch_size=batch_size)

        return arc_scores

    def _score_relations(self, word_exprs, target):
        rel_heads = self.label_head(word_exprs)
        rel_modifiers = self.label_modi(word_exprs)

        stacked = dy.concatenate_cols(rel_heads)

        golds = []
        target[:, 0] = 0  # root is currently negative. mask this
        for column in target.T:
            m_gold = dy.pick_batch(stacked, indices=column, dim=1)
            golds.append(m_gold)

        rel_arcs = []
        for modifier, gold in zip(rel_modifiers, golds):
            rel_arc = dy.tanh(modifier + gold + self.label_bias.expr())
            rel_arcs.append(rel_arc)

        rel_arcs = dy.concatenate_cols(rel_arcs)
        rel_scores = self.l_scorer(rel_arcs)

        return rel_scores

    def __call__(self, x):
        """Forward pass of the model. expects a tuple of x and y."""
        (word_ids, tag_ids), (gold_arcs, gold_rels) = x

        batch_size, n = word_ids.shape
        train = gold_arcs is not None

        # Encode tokens and POS-tags into a sequence of tokens
        word_exprs = self._encode(word_ids, tag_ids, train)

        # Score all n**2 edges
        arc_scores = self._score_edges(word_exprs, batch_size, n)

        # During training employ loss augmeted inference as described in the paper.
        # Originates from Taskar et al., 2005.
        if train:
            arc_scores = self._augmented_scores(arc_scores, gold_arcs, batch_size, n)

        # sorry for the density...
        # 1. ensure tensor is atleast 3d (dynet squeezes batch dimension if it is equal to one)
        # 2. swap the first and third dimension, since our algorithm implementation assumes row major
        numpy_scores = np.atleast_3d(arc_scores.npvalue())
        numpy_scores = np.moveaxis(numpy_scores, -1, 0)

        # in case the data is padded we wish to cut the fat.
        sentence_lengths = self._vocab.get_lengths(word_ids)
        parsed_tree = eisner(numpy_scores, clip=sentence_lengths)

        # Score the (unlabeled) parsed tree
        edges_for_labels = gold_arcs if train else parsed_tree
        rel_scores = self._score_relations(word_exprs, edges_for_labels)

        # 1. Transpose predictions to swap batches and scores to adhere to row major
        # 2. Ensure matrix is atleast 2D since dynet squeezes batch dimension when it equals one
        predicted_rels = np.transpose(rel_scores.npvalue().argmax(0))
        predicted_rels = np.atleast_2d(predicted_rels)

        if train:
            mask = self._vocab.get_mask(word_ids)
            arc_loss = kiperwasser_loss(arc_scores, parsed_tree, gold_arcs, mask)
            rel_loss = hinge(rel_scores, predicted_rels, gold_rels, mask)
            loss = arc_loss + rel_loss
        else:
            loss = None

        return parsed_tree, predicted_rels, loss

    def parameters(self):
        """Return parameters of model."""
        return self.params

    def save_to_file(self, filename):
        """Save parameters to file."""
        self.params.save(filename)

    def load_from_file(self, filename):
        """Load parameters from file."""
        self.params.populate(filename)

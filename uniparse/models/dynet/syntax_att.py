"""TODO."""

from collections import defaultdict

import dynet as dy
import numpy as np

from uniparse.backend.dynet_backend import crossentropy, hinge, kiperwasser_loss
from uniparse.decoders import eisner, cle


def Dense(model_parameters, input_dim, hidden_dim, activation=None, use_bias=False):
    """High level dense layer implementation."""
    w = model_parameters.add_parameters((hidden_dim, input_dim))
    b = model_parameters.add_parameters((hidden_dim,)) if use_bias else None

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


class Parser:
    """TODO."""

    def __init__(self, vocab, **kwargs):
        """Initialize parser parameters based on settings in the vocabulary object."""
        params = dy.ParameterCollection()

        upos_dim = 100
        word_dim = 25
        hidden_dim = 100
        bilstm_out = (word_dim + upos_dim) * 2
        rnn_input = word_dim + upos_dim

        self._dropout = None  # 0.3

        lookup = defaultdict(int, vocab._id2freq).__getitem__
        self.freq_map = np.vectorize(lookup)

        self.wlookup = params.add_lookup_parameters((vocab.vocab_size, word_dim))
        self.tlookup = params.add_lookup_parameters((vocab.upos_size, upos_dim))

        self.rnn = dy.BiRNNBuilder(
            2, rnn_input, bilstm_out, params, dy.VanillaLSTMBuilder
        )

        # edge encoding
        self.edge_head = Dense(params, bilstm_out, hidden_dim)
        self.edge_modi = Dense(params, bilstm_out, hidden_dim)
        self.edge_parent = Dense(params, bilstm_out, hidden_dim)
        self.edge_bias = params.add_parameters((hidden_dim,))

        # edge scoring
        self.e1_scorer = Dense(params, hidden_dim, 1, activation=None, use_bias=True)

        # rel encoding
        self.label_head = Dense(params, bilstm_out, hidden_dim)
        self.label_modi = Dense(params, bilstm_out, hidden_dim)
        self.label_bias = params.add_parameters((hidden_dim,))

        # c, d = 1, 100
        self.U = params.add_lookup_parameters((word_dim, word_dim, 1))
        self.W = params.add_lookup_parameters((1, word_dim * 2))
        self.b = params.add_lookup_parameters((1, 1))

        # label scoring
        self.l_scorer = Dense(params, 250, vocab.label_count)
        self.params = params
        self._vocab = vocab

        self.ii = 0

    def _encode(self, token_ids, tag_ids, train):
        if train:
            c = self.freq_map(token_ids)
            drop_mask = np.greater(0.25 / (c + 0.25), np.random.rand(*token_ids.shape))
            token_ids = np.where(drop_mask, self._vocab.OOV, token_ids)

        # encode and contextualize
        word_embs = [dy.lookup_batch(self.wlookup, t) for t in token_ids.T]
        tag_embs = [dy.lookup_batch(self.tlookup, t) for t in tag_ids.T]
        words = [dy.concatenate([w, p]) for w, p in zip(word_embs, tag_embs)]

        contextualized_words = self.rnn.transduce(words)

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

    def attend_score(self, tags, words):
        n = len(words)
        d, batch_size = words[0].dim()

        tags = dy.concatenate_cols(tags)
        S = dy.transpose(tags) * tags
        S = dy.softmax(S, d=0)
        diagonal_mask = dy.inputTensor(np.eye(n) < 1)
        S = dy.cmult(S, diagonal_mask)
        # S :: (n, n)
        weighted_words = []
        for i, w in enumerate(words):
            weighted_sum = sum([dy.cmult(att_ij, w) for att_ij in dy.pick(S, i, 0)])
            weighted_words.append(weighted_sum)

        word_h = self.edge_head(weighted_words)
        word_m = self.edge_modi(weighted_words)

        arc_edges = [dy.tanh(h + m + self.edge_bias) for m in word_m for h in word_h]

        arc_scores = self.e1_scorer(arc_edges)
        arc_scores = dy.concatenate_cols(arc_scores)
        arc_scores = dy.reshape(arc_scores, d=(n, n), batch_size=batch_size)
        return arc_scores

    def _score_edges(self, word_exprs):
        """Score edges in a pointer-network manner."""
        # Linear project of heads

        n = len(word_exprs)
        d, batch_size = word_exprs[0].dim()
        word_h = self.edge_head(word_exprs)
        word_m = self.edge_modi(word_exprs)

        # tradish
        arc_edges = [
            dy.tanh(head + modifier + self.edge_bias)
            for modifier in word_m
            for head in word_h
        ]

        if self._dropout:
            arc_edges = [dy.dropout(arc, self._dropout) for arc in arc_edges]

        arc_scores = self.e1_scorer(arc_edges)

        arc_scores = dy.concatenate_cols(arc_scores)
        arc_scores = dy.reshape(arc_scores, d=(n, n), batch_size=batch_size)

        return arc_scores

    def __call__(self, x, decoder):
        """Forward pass of the model. expects a tuple of x and y."""
        (word_ids, tag_ids) = x[0]
        (gold_arcs, gold_rels) = x[1]
        train = gold_arcs is not None
        batch_size, n = word_ids.shape
        contextualized_x = self._encode(word_ids, tag_ids, train)
        # arc_scores = self.attend_score(c_tags, c_words)
        arc_scores = self._score_edges(contextualized_x)
        if train:
            arc_scores = self._augmented_scores(arc_scores, gold_arcs, batch_size, n)

        rel_scores = self.l_scorer(contextualized_x)
        rel_scores = dy.concatenate_cols(rel_scores)

        return arc_scores, rel_scores

    def parameters(self):
        """Return parameters of model."""
        return self.params

    def save_to_file(self, filename):
        """Save parameters to file."""
        self.params.save(filename)

    def load_from_file(self, filename):
        """Load parameters from file."""
        self.params.populate(filename)

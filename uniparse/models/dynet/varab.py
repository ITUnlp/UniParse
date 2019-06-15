"""TODO."""

from collections import defaultdict

import dynet as dy
import numpy as np

from uniparse.backend.dynet_backend import crossentropy, hinge, kiperwasser_loss
from uniparse.decoders import eisner


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


class Varab:
    """TODO."""

    def __init__(self, vocab, **kwargs):
        """Initialize parser parameters based on settings in the vocabulary object."""
        params = dy.ParameterCollection()

        upos_dim = 100
        word_dim = 100
        hidden_dim = 256
        # bilstm_out = (word_dim + upos_dim) * 2
        bilstm_out = upos_dim

        self._dropout = 0.0

        lookup = defaultdict(int, vocab._id2freq).__getitem__
        self.freq_map = np.vectorize(lookup)

        self.wlookup = params.add_lookup_parameters((vocab.vocab_size, word_dim))
        self.tlookup = params.add_lookup_parameters((vocab.upos_size, upos_dim))

        self.encoder_rnn = dy.BiRNNBuilder(
            2, upos_dim, bilstm_out, params, dy.VanillaLSTMBuilder
        )

        self.word_encoder_rnn = dy.BiRNNBuilder(
            2, word_dim, bilstm_out, params, dy.VanillaLSTMBuilder
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
        self.l_scorer = Dense(params, hidden_dim, vocab.label_count)
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

        contextualized_words = self.encoder_rnn.transduce(words)

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

    def _affine(self, m, h, U):
        (_, c, d), _ = U.dim()
        a = dy.reshape(U, (d, d * c))
        r = dy.reshape(dy.transpose(m) * a, (d, c))
        return dy.transpose(h) * r

    # improve impl.
    # 1. decompose the concatenation like kiperwasser
    # 2. don't score i==j
    # 3. do biaffine with labels
    # 4. doesn't work with labels
    def _biaffine(self, m, h, U, W, b):
        return self._affine(m, h, U) + (
            dy.reshape(dy.concatenate([m, h]), (1, 200)) * W + b
        )

    def teacher_parent_force(self):
        # arc_edges = []
        # for modifier in word_m:
        #     for hid, head in enumerate(word_h):
        #         golds[:, 0] = 0
        #         parents = dy.pick_batch(word_p, golds[:, hid], 1)
        #         e = dy.tanh(head + modifier + self.edge_bias)
        #         e = dy.concatenate([e, parents])
        #         arc_edges.append(e)
        pass

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
            arc_edges = [dy.dropout(arc, 0.33) for arc in arc_edges]

        arc_scores = self.e1_scorer(arc_edges)

        # biaffine (dozat & manning)
        # arc_scores = [
        #     self._biaffine(m, h, self.U, self.W, self.b)
        #     for m in word_m
        #     for h in word_h
        # ]

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

        if self._dropout:
            rel_arcs = [dy.dropout(x, 0.33) for x in rel_arcs]

        rel_arcs = dy.concatenate_cols(rel_arcs)
        rel_scores = self.l_scorer(rel_arcs)

        return rel_scores

    def __call__(self, x):
        """Forward pass of the model. expects a tuple of x and y."""
        (word_ids, tag_ids), (gold_arcs, gold_rels) = x

        batch_size, n = word_ids.shape
        train = gold_arcs is not None

        # Encode tokens and POS-tags into a sequence of tokens
        # word_exprs = self._encode(word_ids, tag_ids, train)

        # if self._dropout:
        #     word_exprs = [dy.dropout(x, 0.33) for x in word_exprs]

        # Score all n**2 edges
        word_exprs = [dy.lookup_batch(self.wlookup, t) for t in word_ids.T]
        tags = [dy.lookup_batch(self.tlookup, t) for t in tag_ids.T]

        c_tags = self.encoder_rnn.transduce(tags)
        c_words = self.word_encoder_rnn.transduce(word_exprs)
        arc_scores = self.attend_score(c_tags, c_words)
        # arc_scores = self._score_edges()

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
        rel_scores = self._score_relations(c_words, edges_for_labels)

        # 1. Transpose predictions to swap batches and scores to adhere to row major
        # 2. Ensure matrix is atleast 2D since dynet squeezes batch dimension when it equals one
        # - Note that np.atleast_2d inserts the 2nd dimension on the left,
        #   as opposed to np.atleast_3d which does so on the right
        predicted_rels = np.transpose(rel_scores.npvalue().argmax(0))
        predicted_rels = np.atleast_2d(predicted_rels)

        if train:
            mask = self._vocab.get_mask(word_ids)
            arc_loss = kiperwasser_loss(arc_scores, parsed_tree, gold_arcs, mask)
            rel_loss = hinge(rel_scores, predicted_rels, gold_rels, mask)

            # arc_loss = crossentropy(arc_scores, parsed_tree, gold_arcs, mask)
            # rel_loss = crossentropy(rel_scores, predicted_rels, gold_rels, mask)
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

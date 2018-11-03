from collections import defaultdict

import dynet as dy
import numpy as np

from uniparse.types import Parser


def Dense(model_parameters, input_dim, hidden_dim, activation, use_bias):
    """ Typical dense layer as required without dropout by Kiperwasser and Goldberg (2016) """
    w = model_parameters.add_parameters((hidden_dim, input_dim))
    b = model_parameters.add_parameters((hidden_dim,)) if use_bias else None

    def call(xs):
        """ todo """
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


class DependencyParser(Parser):
    """  Implementation of Kiperwasser and Goldbergs (2016) bilstm parser paper  """
    def __init__(self, vocab):
        params = dy.ParameterCollection()

        upos_dim = 25
        word_dim = 100
        hidden_dim = 100
        bilstm_out = (word_dim+upos_dim) * 2  # 250

        self.word_count = vocab.vocab_size
        self.upos_count = vocab.upos_size
        self.i2c = defaultdict(int, vocab._id2freq)
        self.label_count = vocab.label_count
        self._vocab = vocab

        self.wlookup = params.add_lookup_parameters((self.word_count, word_dim))
        self.tlookup = params.add_lookup_parameters((self.upos_count, upos_dim))

        self.deep_bilstm = dy.BiRNNBuilder(2, word_dim+upos_dim, bilstm_out, params, dy.VanillaLSTMBuilder)

        # edge encoding
        self.edge_head = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.edge_modi = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.edge_bias = params.add_parameters((hidden_dim,))

        # edge scoring
        self.e_scorer = Dense(params, hidden_dim, 1, activation=None, use_bias=True)        

        # rel encoding
        self.label_head = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.label_modi = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.label_bias = params.add_parameters((hidden_dim,))

        # label scoring
        self.l_scorer = Dense(params, hidden_dim, self.label_count, activation=None, use_bias=True)
        self.params = params

    def parameters(self):
        return self.params

    def save_to_file(self, filename: str) -> None:
        self.params.save(filename)

    def load_from_file(self, filename: str) -> None:
        self.params.populate(filename)

    @staticmethod
    def _propability_map(matrix, dictionary):
        return np.vectorize(dictionary.__getitem__)(matrix)

    def __call__(self, in_tuple):
        (word_ids, upos_ids), (gold_arcs, gold_rels) = in_tuple
        return self.run(word_ids, upos_ids, gold_arcs, gold_rels)

    def run(self, word_ids, upos_ids, target_arcs, rel_targets):
        batch_size, n = word_ids.shape

        train = target_arcs is not None

        mask = np.greater(word_ids, self._vocab.ROOT)

        n = word_ids.shape[-1]
        if train:
            c = self._propability_map(word_ids, self.i2c)
            drop_mask = np.greater(0.25/(c+0.25), np.random.rand(*word_ids.shape))
            word_ids = np.where(drop_mask, self._vocab.OOV, word_ids)  

        # encode and contextualize
        word_embs = [dy.lookup_batch(self.wlookup, word_ids[:, i]) for i in range(n)]
        upos_embs = [dy.lookup_batch(self.tlookup, upos_ids[:, i]) for i in range(n)]
        words = [dy.concatenate([w, p]) for w, p in zip(word_embs, upos_embs)]
        word_exprs = self.deep_bilstm.transduce(words)

        word_h = self.edge_head(word_exprs)
        word_m = self.edge_modi(word_exprs)

        # word_m = dy.concatenate_cols(word_m)
        # print(word_m.dim(), word_h[0].dim(), self.edge_bias.dim())
        # arc_edges = []
        # for head_i in range(n):
        #     arc_rep = dy.tanh(word_h[head_i] + word_m + self.edge_bias.expr())
        #     arc_edges.append(arc_rep)

        # arc_scores = self.e_scorer(arc_edges)
        #arc_scores = dy.concatenate_cols(arc_scores)

        arc_edges = [
            dy.tanh(word_h[head] + word_m[modifier] + self.edge_bias.expr())
            for modifier in range(n)
            for head in range(n)
        ]

        # edges scoring
        arc_scores = self.e_scorer(arc_edges)
        arc_scores = dy.concatenate_cols(arc_scores)
        arc_scores = dy.reshape(arc_scores, d=(n, n), batch_size=batch_size)

        # Loss augmented inference
        if train:
            margin = np.ones((n, n, batch_size))
            for bi in range(batch_size):
                for m in range(n):
                    h = target_arcs[bi, m]
                    margin[h, m, bi] -= 1

            margin_tensor = dy.inputTensor(margin, batched=True)
            arc_scores = arc_scores + margin_tensor


        rel_heads = self.label_head(word_exprs)
        rel_modifiers = self.label_modi(word_exprs)
        stacked = dy.concatenate_cols(rel_heads)
        # (d, n) x batch_size

        sentence_lengths = n - np.argmax(word_ids[:, ::-1] > self._vocab.PAD, axis=1)
        parsed_tree = self.decode(arc_scores, sentence_lengths) # if target_arcs is None else target_arcs

        golds = []
        parsed_tree[:, 0] = 0  # root is currently negative. mask this
        for column in parsed_tree.T:
            m_gold = dy.pick_batch(stacked, indices=column, dim=1)
            golds.append(m_gold)


        rel_arcs = []
        for modifier, gold in zip(rel_modifiers, golds):
            rel_arc = modifier + gold + self.label_bias.expr()
            rel_arcs.append(rel_arc)

        rel_arcs = dy.concatenate_cols(rel_arcs)
        # ((d, n), batch_size)
        rel_scores = self.l_scorer(rel_arcs)

        loss = self.compute_loss(arc_scores, rel_scores, target_arcs, rel_targets, mask) if train else None

        predicted_rels = rel_scores.npvalue().argmax(0)
        predicted_rels = predicted_rels[:, np.newaxis] if predicted_rels.ndim < 2 else predicted_rels
        predicted_rels = predicted_rels.T

        #parsed_tree = parsed_tree if not train else self.decode(arc_scores)

        return parsed_tree, predicted_rels, loss

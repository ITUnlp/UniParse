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
        hidden_dim = 125
        bilstm_out = (word_dim+upos_dim) * 2  # 250

        self.word_count = vocab.vocab_size
        self.upos_count = vocab.upos_size
        self.i2c = defaultdict(int, vocab._id2freq)
        self.label_count = vocab.label_count
        self._vocab = vocab

        self.wlookup = params.add_lookup_parameters((self.word_count, word_dim))
        self.tlookup = params.add_lookup_parameters((self.upos_count, upos_dim))

        #self.deep_bilstm = dy.BiRNNBuilder(2, word_dim+upos_dim, bilstm_out, params, dy.VanillaLSTMBuilder)

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

    def save_to_file(self, filename):
        self.params.save(filename)

    def load_from_file(self, filename):
        self.params.populate(filename)

    def __call__(self, in_tuple):
        (word_ids, upos_ids), (gold_arcs, gold_rels) = in_tuple
        return self.run(word_ids, upos_ids, gold_arcs, gold_rels)

    def run(self, word_ids, upos_ids, target_arcs, rel_targets):
        batch_size, n = word_ids.shape

        train = target_arcs is not None

        mask = np.greater(word_ids, self._vocab.ROOT)

        n = word_ids.shape[-1]

        # encode and contextualize
        word_embs = [dy.lookup_batch(self.wlookup, word_ids[:, i]) for i in range(n)]
        upos_embs = [dy.lookup_batch(self.tlookup, upos_ids[:, i]) for i in range(n)]
        word_exprs = [dy.concatenate([w, p]) for w, p in zip(word_embs, upos_embs)]
        #word_exprs = self.deep_bilstm.transduce(word_exprs)

        arcs = [
            dy.concatenate([m, h, p]) 
            for m in word_exprs
            for h in word_exprs
            for p in word_exprs
        ]

        arc_scores = self.e_scorer(arcs)
        dy.concatenate(arc_scores, )

        arc_scores = np.array([s.value() for s in arc_scores])
        arc_scores = arc_scores.reshape((n,n,n,batch_size))
        arc_scores = np.transpose(arc_scores)

        max_parents_in = arc_scores.argmax(3)
        arc_scores 

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

        return parsed_tree, predicted_rels, loss


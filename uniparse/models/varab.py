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

def MLP_Align(params, input_dim, hidden_dim, out_dim, activation, use_bias):
    w1 = params.add_lookup_parameters((hidden_dim, input_dim))
    b1 = params.add_lookup_parameters((hidden_dim,))

    w2 = params.add_lookup_parameters((hidden_dim, input_dim))
    b2 = params.add_lookup_parameters((hidden_dim,))

    w_out = params.add_lookup_parameters((hidden_dim, input_dim))
    b_out = params.add_lookup_parameters((hidden_dim,))


    def call(xs):
        encode_1 = w1 * xs
        encode_1 = encode_1 + b1

        encode_2 = w2 * xs
        encode_2 = encode_2 + b2
        
        output = w_out * (encode_1 + encode_2) + b_out

        return activation(output)

    return call



class DependencyParser(Parser):
    """  Implementation of Kiperwasser and Goldbergs (2016) bilstm parser paper  """
    def __init__(self, vocab):
        params = dy.ParameterCollection()

        upos_dim = 25
        word_dim = 100
        hidden_dim = 100
        bilstm_out = (word_dim+upos_dim) * 2 

        self.word_count = vocab.vocab_size
        self.upos_count = vocab.upos_size
        self.i2c = defaultdict(int, vocab._id2freq)
        self.label_count = vocab.label_count
        self._vocab = vocab
        self.freq_map = np.vectorize(self.i2c.__getitem__)

        self.wlookup = params.add_lookup_parameters((self.word_count, word_dim))
        self.tlookup = params.add_lookup_parameters((self.upos_count, upos_dim))

        self.deep_bilstm = dy.BiRNNBuilder(2, word_dim+upos_dim, bilstm_out, params, dy.VanillaLSTMBuilder)

        # edge encoding
        self.edge_head = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.edge_modi = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
        self.edge_sist = Dense(params, bilstm_out, hidden_dim, activation=None, use_bias=False)
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
            #c = self._propability_map(word_ids, self.i2c)
            c = self.freq_map(word_ids)
            drop_mask = np.greater(0.25/(c+0.25), np.random.rand(*word_ids.shape))
            word_ids = np.where(drop_mask, self._vocab.OOV, word_ids)  

        # encode and contextualize
        word_embs = [dy.lookup_batch(self.wlookup, wid) for wid in word_ids.T]
        upos_embs = [dy.lookup_batch(self.tlookup, pid) for pid in upos_ids.T]
        words = [dy.concatenate([w, p]) for w, p in zip(word_embs, upos_embs)]

        word_exprs = self.deep_bilstm.transduce(words)
        # word_exprs = words

        word_h = self.edge_head(word_exprs)
        word_m = self.edge_modi(word_exprs)
        word_s = self.edge_sist(word_exprs)


        arc_scores = []
        for mod in word_m:
            for head in word_h:
                edge_rep = dy.tanh(head + mod + self.edge_bias)
                edge_score = self.e_scorer(edge_rep)
                arc_scores.append(edge_score)
        
        # edges scoring
        # arc_scores = self.e_scorer(arc_edges)
        arc_scores = dy.concatenate_cols(arc_scores)
        arc_scores = dy.reshape(arc_scores, d=(n, n), batch_size=batch_size)

        # Loss augmented inference
        if train:
            margin = np.ones((n, n, batch_size))
            # bs = [0,0,0,0,1,1,1,1,2,2,2,2]
            # ns = [0,1,2,3,4,5,6,7,n]
            # hds = np.flatten(target_arcs)
            # margin[bs, ns, hds] -= 1
            for bi in range(batch_size):
                for m in range(n):
                    h = target_arcs[bi, m]
                    margin[h, m, bi] -= 1

            margin_tensor = dy.inputTensor(margin, batched=True)
            arc_scores = arc_scores + margin_tensor

        sentence_lengths = n - np.argmax(word_ids[:, ::-1] > self._vocab.PAD, axis=1)
        parsed_tree = self.decode(arc_scores, sentence_lengths)

        for m, h in enumerate(parsed_tree):
            # h < m :: right arc
            # h > m :: left arc
            _range = range(h, m) if h < m else range(m, h)
            for c in _range:
                new_arc_score = score(h, m, c)
                c_head = parsed_tree[c]
                if new_arc_score > arc_scores[:, m, c_head]:
                    parsed_tree[c] = m

        tree_for_labels = parsed_tree if target_arcs is None else target_arcs

        rel_heads = self.label_head(word_exprs)
        rel_modifiers = self.label_modi(word_exprs)

        stacked = dy.concatenate_cols(rel_heads)
        
        golds = []
        tree_for_labels[:, 0] = 0  # root is currently negative. mask this
        for column in tree_for_labels.T:
            m_gold = dy.pick_batch(stacked, indices=column, dim=1)
            golds.append(m_gold)


        rel_arcs = []
        for modifier, gold in zip(rel_modifiers, golds):
            rel_arc = dy.tanh(modifier + gold + self.label_bias.expr())
            rel_arcs.append(rel_arc)

        rel_arcs = dy.concatenate_cols(rel_arcs)
        
        rel_scores = self.l_scorer(rel_arcs)
        predicted_rels = rel_scores.npvalue().argmax(0)
        predicted_rels = predicted_rels[:, np.newaxis] if predicted_rels.ndim < 2 else predicted_rels
        predicted_rels = predicted_rels.T

        if train:
            arc_loss = self.loss_object.hinge(arc_scores, parsed_tree, target_arcs, mask, batch_size_norm=False)
            rel_loss = self.loss_object.hinge(rel_scores, predicted_rels, rel_targets, mask, batch_size_norm=False)
            
            #arc_loss = arc_loss / batch_size
            #rel_loss = rel_loss / batch_size
            #loss = arc_loss + rel_loss
            loss = (arc_loss + rel_loss) / batch_size
        else:
            loss = None

        loss = self.compute_loss(arc_scores, rel_scores, target_arcs, rel_targets, mask) if train else None

        return parsed_tree, predicted_rels, loss


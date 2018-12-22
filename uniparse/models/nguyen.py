from collections import defaultdict

import dynet as dy
import numpy as np

from uniparse.types import Parser

def Dense(model_parameters, input_dim, hidden_dim, activation, use_bias, dropout=0.0):
    """ Typical dense layer as required without dropout by Kiperwasser and Goldberg (2016) """
    w = model_parameters.add_parameters((hidden_dim, input_dim))
    b = model_parameters.add_parameters((hidden_dim,)) if use_bias else None

    def call(xs):
        """ todo """
        output = w.expr() * xs
        if use_bias:
            output = output + b.expr()
        if activation:
            return activation(output)

        return output

    def apply(xs):
        if isinstance(xs, list):
            return [call(dy.dropout(x, dropout)) for x in xs]
        else:
            return call(dy.dropout(xs, dropout))

    return apply


class MLP(object):
    def __init__(self, params, input_dim, hidden_dim, out_dim, activation, use_bias=True):
        self.activation = activation
        self.use_bias = use_bias

        # 1 layer
        self.W = params.add_parameters((hidden_dim, input_dim))
        self.bias = params.add_parameters((hidden_dim,)) if use_bias else None

        # 2 layer
        self.W2 = params.add_parameters((out_dim, hidden_dim))
        self.bias2 = params.add_parameters((out_dim,)) if use_bias else None


    def __call__(self, xs):
        if isinstance(xs, list):
            return [self.call(x) for x in xs]

        else:
            return self.call(xs)
    
    def call(self, xs):
        """ todo """
        # first pass
        output = self.W.expr() * xs
        if self.use_bias:
            output = output + self.bias.expr()
        if self.activation:
            output = self.activation(output)

        # second pass
        output = (self.W2.expr() * output) + self.bias2.expr()

        return output


class DependencyParser(Parser):
    """  Implementation of Kiperwasser and Goldbergs (2016) bilstm parser paper  """
    def __init__(self, vocab, embs):
        params = dy.ParameterCollection()

        upos_dim = 100
        word_dim = 100
        char_dim = 50
        hidden_dim = 100
        bilstm_out = 128 # 256

        self.word_count = vocab.vocab_size
        self.upos_count = vocab.upos_size
        self.char_count = vocab.char_size
        self.i2c = defaultdict(int, vocab._id2freq)
        self.label_count = vocab.label_count
        self._vocab = vocab

        self.char_dim = char_dim

        #self.wlookup = params.add_lookup_parameters((self.word_count, word_dim))
        self.wlookup = params.lookup_parameters_from_numpy(embs)
        self.clookup = params.add_lookup_parameters((self.char_count, char_dim))
        self.tlookup = params.add_lookup_parameters((self.upos_count, upos_dim))

        self.deep_bilstm = dy.BiRNNBuilder(2, word_dim+upos_dim, bilstm_out, params, dy.VanillaLSTMBuilder)
        #self.deep_bilstm = dy.BiRNNBuilder(2, word_dim+upos_dim+char_dim, bilstm_out, params, dy.VanillaLSTMBuilder)
        self.char_bilstm = dy.BiRNNBuilder(1, char_dim, char_dim, params, dy.VanillaLSTMBuilder)

        #self.pos_classifier = Dense(params, word_dim + char_dim, hidden_dim, activation=dy.tanh, use_bias=False)
        #self.pos_classifier2 = Dense(params, hidden_dim, self.upos_count, activation=dy.softmax, use_bias=False)

        self.pos_mlp = MLP(params, word_dim + char_dim, hidden_dim, self.upos_count, activation=dy.tanh)
        self.arc_mlp = MLP(params, bilstm_out*4, hidden_dim, 1, activation=dy.tanh)
        self.rel_mlp = MLP(params, bilstm_out*4, hidden_dim, self.label_count, activation=dy.tanh)

        self.params = params

    def parameters(self):
        return self.params

    def save_to_file(self, filename):
        self.params.save(filename)

    def load_from_file(self, filename):
        self.params.populate(filename)

    def __call__(self, in_tuple):
        (word_ids, upos_ids, chars), (gold_arcs, gold_rels) = in_tuple
        return self.run(word_ids, upos_ids, chars, gold_arcs, gold_rels)

    def run(self, word_ids, upos_ids, chars, target_arcs, rel_targets):
        batch_size, n, n_char_tokens = chars.shape

        train = target_arcs is not None

        mask = np.greater(word_ids, self._vocab.ROOT)

        n = word_ids.shape[-1]

        # encode and contextualize
        word_embs = [dy.lookup_batch(self.wlookup, words, update=True) for words in word_ids.T]

        # characters
        # transpose for column major and flatten with fortran mode
        # chars = np.reshape(chars.T, (n_char_tokens, batch_size*n), "F")
        # char_embs = [dy.lookup_batch(self.clookup, chars[i, :]) for i in range(n_char_tokens)]

        # # now char_embs are batches of character sequences
        # char_embs = self.char_bilstm.transduce(char_embs)

        # # pick the last one (final character state of the word)
        # cword_exprs = char_embs[-1]
        
        # cword_exprs = dy.reshape(cword_exprs, (self.char_dim, n), batch_size=batch_size)
        # cword_exprs = [dy.pick(cword_exprs, i, dim=1) for i in range(n)]

        # pos_words = [dy.concatenate([w, c]) for w, c in zip(word_embs, cword_exprs)]

        # pos_predictions = self.pos_mlp(pos_words)
        
        # if train:
        #     pos_loss = [dy.pickneglogsoftmax_batch(pos_preds, pos_gold) for pos_preds, pos_gold in zip(pos_predictions, upos_ids.T)]
        #     pos_loss = dy.esum(pos_loss[1:]) # don't compute loss for <root>
        #     pos_loss = dy.sum_batches(pos_loss) / batch_size

        # pos_preds = [pos_pred.npvalue().argmax(0) for pos_pred in pos_predictions]
        # pos_embs = [dy.lookup_batch(self.tlookup, np.atleast_1d(pos_pred)) for pos_pred in pos_preds]

        pos_embs = [dy.lookup_batch(self.tlookup, upos_ids[:, i]) for i in range(n)]

        
        # concat words, tags, and character embs
        word_exprs = [dy.concatenate([w, p]) for w, p in zip(word_embs, pos_embs)]
        #word_exprs = [dy.concatenate([w, p]) for w, p in zip(pos_words, pos_embs)]
        word_exprs = [dy.dropout(x, 0.33) for x in word_exprs]

        word_exprs = self.deep_bilstm.transduce(word_exprs)

        arc_edges = [
            dy.concatenate([
                word_exprs[head], word_exprs[modifier],
                dy.cmult(word_exprs[head], word_exprs[modifier]),
                dy.abs(word_exprs[head] - word_exprs[modifier])
            ])
            for modifier in range(n)
            for head in range(n)
        ]

        arc_edges = [dy.dropout(arc, 0.33) for arc in arc_edges]

        # edges scoring
        arc_scores = self.arc_mlp(arc_edges)
        arc_scores = dy.concatenate_cols(arc_scores)
        arc_scores = dy.reshape(arc_scores, d=(n, n), batch_size=batch_size)

        if train:
            # Loss augmented inference
            margin = np.ones((n, n, batch_size))
            for bi in range(batch_size):
                for m in range(n):
                    h = target_arcs[bi, m]
                    margin[h, m, bi] -= 1

            margin_tensor = dy.inputTensor(margin, batched=True)
            arc_scores = arc_scores + margin_tensor

        stacked = dy.concatenate_cols(word_exprs)
        # (d, n) x batch_size

        sentence_lengths = n - np.argmax(word_ids[:, ::-1] > self._vocab.PAD, axis=1)
        parsed_tree = self.decode(arc_scores, sentence_lengths) # if target_arcs is None else target_arcs

        golds = []
        parsed_tree[:, 0] = 0  # root is currently negative. mask this
        for column in parsed_tree.T:
            m_gold = dy.pick_batch(stacked, indices=column, dim=1)
            golds.append(m_gold)

        rel_arcs = [dy.concatenate([gh, m, dy.cmult(gh, m), dy.abs(gh - m)]) for gh, m in zip(golds, word_exprs)]
        rel_arcs = [dy.dropout(x, 0.33) for x in rel_arcs]
        rel_arcs = dy.concatenate_cols(rel_arcs)

        # ((rel_classes, n), batch_size)
        rel_scores = self.rel_mlp(rel_arcs)

        predicted_rels = rel_scores.npvalue().argmax(0)
        predicted_rels = predicted_rels[:, np.newaxis] if predicted_rels.ndim < 2 else predicted_rels
        predicted_rels = predicted_rels.T

        #num_tokens = int(np.sum(mask))
        #pos_correct = np.equal(np.transpose(pos_preds), upos_ids).astype(np.float32) * mask
        #pos_accuracy = np.sum(pos_correct) / num_tokens
        pos_accuracy = 1.0

        loss = None
        if train:
            arc_loss = self.loss_object.kiperwasser_loss(arc_scores, parsed_tree, target_arcs, mask, batch_size_norm=False)
            rel_loss = self.loss_object.hinge(rel_scores, predicted_rels, rel_targets, mask, batch_size_norm=False)
            
            #loss = (arc_loss + rel_loss + pos_loss) / batch_size
            loss = (arc_loss + rel_loss) / batch_size

        return parsed_tree, predicted_rels, pos_accuracy, loss
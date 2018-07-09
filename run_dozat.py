import argparse

import numpy as np
import dynet as dy

from uniparse import Vocabulary, Model
from uniparse.models.dozat import BaseParser
from uniparse.types import Callback
from uniparse.callbacks import TensorboardLoggerCallback, ModelSaveCallback


class UpdateParamsCallback(Callback):
    def on_batch_end(self, info):
        global_step = info["global_step"]
        if global_step % 2 == 0:
            optimizer.learning_rate = learning_rate*decay**(global_step / decay_steps)


argparser = argparse.ArgumentParser()

argparser.add_argument("--train", dest="train", help="Annotated CONLL train file", metavar="FILE", required=True)
argparser.add_argument("--dev", dest="dev", help="Annotated CONLL dev file", metavar="FILE", required=True)
argparser.add_argument("--test", dest="test", help="Annotated CONLL dev test", metavar="FILE", required=True)
argparser.add_argument("--emb", dest="embedding_file")
argparser.add_argument("--epochs", dest="epochs", type=int, default=283)
argparser.add_argument("--tb_dest", dest="tb_dest", required=True)
argparser.add_argument("--vocab_dest", dest="vocab_dest")
argparser.add_argument("--model_dest", dest="model_dest", required=True)

argparser.add_argument("--lstm_layers", dest="lstm_layers", type=int, default=3)
argparser.add_argument("--no_orth_init", dest="no_orth_init", action='store_true')
argparser.add_argument("--dropout", type=int, default=0.33)

arguments, unknown = argparser.parse_known_args()

np.random.seed(666)

# [Data]
min_occur_count = 2
train_file = arguments.train
dev_file = arguments.dev
pretrained_embeddings_file = arguments.embedding_file
vocab_destination = arguments.vocab_dest
tensorboard_destination = arguments.tb_dest
model_destination = arguments.model_dest

# [Network]
word_dims = 100
tag_dims = 100
lstm_hiddens = 400
mlp_arc_size = 500
mlp_rel_size = 100
lstm_layers = arguments.lstm_layers
dropout_emb = arguments.dropout
dropout_lstm_input = arguments.dropout
dropout_lstm_hidden = arguments.dropout
dropout_mlp = arguments.dropout

# Worry not. Its the inverse. If no_orth_init is enabled, then the orthogonal feature is disabled.
orth_init = False if arguments.no_orth_init else True

""" Hyperparamters for optimizer """
learning_rate = 2e-3
decay = .75
decay_steps = 5000
beta_1 = .9
beta_2 = .9
epsilon = 1e-12

# [Run]
batch_scale = 5000
n_epochs = arguments.epochs

if arguments.epochs < 283:
    print(">> WARNING: your running less epochs than originally stated (%s)" % arguments.epochs)

if arguments.lstm_layers != 3:
    print(">> WARNING: running with more or less bilstm layers than origin (%d)" % arguments.lstm_layers)

if arguments.embedding_file is None:
    print(">> WARNING: Running without pretrained embeddings.")

if arguments.no_orth_init:
    print(">> Warning: running without orthogonal initilization on parameters")

if arguments.dropout < 0.33:
    print(">> Warning: running model with less dropout")

vocab = Vocabulary()
vocab = vocab.fit(train_file, pretrained_embeddings_file, min_occur_count)
embs = vocab.load_embedding(normalize=True) if arguments.embedding_file else None

# save vocab for reproducing later
if vocab_destination:
    vocab.save(vocab_destination)
    print("> saving vocab to", vocab_destination)

""" """
model = BaseParser(vocab, word_dims, tag_dims,
                   dropout_emb, lstm_layers,
                   lstm_hiddens, dropout_lstm_input, dropout_lstm_hidden,
                   mlp_arc_size, mlp_rel_size, dropout_mlp, embs, orth_init)

""" Instantiate custom optimizer """
optimizer = dy.AdamTrainer(model.parameter_collection, learning_rate, beta_1, beta_2, epsilon)

""" Callbacks """
tensorboard_logger = TensorboardLoggerCallback(tensorboard_destination)
custom_learning_update_callback = UpdateParamsCallback()
save_callback = ModelSaveCallback(model_destination)
callbacks = [tensorboard_logger, custom_learning_update_callback, save_callback]

parser = Model(
    model, decoder="cle", loss="crossentropy", optimizer=optimizer, strategy="scaled_batch", vocab=vocab)

""" Prep data """
training_data = vocab.tokenize_conll(arguments.train)
dev_data = vocab.tokenize_conll(arguments.dev)
test_data = vocab.tokenize_conll(arguments.test)

parser.train(training_data, dev_file, dev_data, epochs=n_epochs, batch_size=batch_scale, callbacks=callbacks)

parser.load_from_file(model_destination)

metrics = parser.evaluate(arguments.test, test_data, batch_size=batch_scale)
test_UAS = metrics["nopunct_uas"]
test_LAS = metrics["nopunct_las"]

tensorboard_logger.raw_write("test_UAS", test_UAS)
tensorboard_logger.raw_write("test_LAS", test_LAS)

print()
print(metrics)
print(">> Test score:", test_UAS, test_LAS)

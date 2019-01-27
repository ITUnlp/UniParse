import argparse

from uniparse import Vocabulary
from uniparse import Model
from uniparse.callbacks import TensorboardLoggerCallback
from uniparse.callbacks import ModelSaveCallback

from uniparse.dataprovider import batch_by_buckets
from uniparse.dataprovider import scale_batch

from uniparse.models.dynet_models.dozat import Dozat

argparser = argparse.ArgumentParser()

argparser.add_argument("--train", required=True)
argparser.add_argument("--dev", required=True)
argparser.add_argument("--test", required=True)
argparser.add_argument("--epochs", type=int, default=30)
argparser.add_argument("--vocab_dest")
argparser.add_argument("--model_dest", required=True)

argparser.add_argument("--lstm_layers", dest="lstm_layers", type=int, default=3)
argparser.add_argument("--dropout", type=int, default=0.33)

arguments, unknown = argparser.parse_known_args()

# [Data]
min_occur_count = 2
train_file = arguments.train
dev_file = arguments.dev
# pretrained_embeddings_file = arguments.embedding_file

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


n_epochs = arguments.epochs
vocab = Vocabulary()
vocab = vocab.fit(arguments.train)

# save vocab for reproducability later
if arguments.vocab_dest:
    print("> saving vocab to", arguments.vocab_dest)
    vocab.save(arguments.vocab_dest)

# prep data
print(">> Loading in data")
train_data = vocab.tokenize_conll(arguments.train)
dev_data = vocab.tokenize_conll(arguments.dev)
test_data = vocab.tokenize_conll(arguments.test)

train_batches = batch_by_buckets(train_data, batch_size=32, shuffle=True)
dev_batches = batch_by_buckets(dev_data, batch_size=32, shuffle=True)
test_batches = batch_by_buckets(test_data, batch_size=32, shuffle=False)


# instantiate model
""" """
model = Dozat(vocab, word_dims, tag_dims,
                   dropout_emb, lstm_layers,
                   lstm_hiddens, dropout_lstm_input, dropout_lstm_hidden,
                   mlp_arc_size, mlp_rel_size, dropout_mlp, None, orthogonal_init=True)

save_callback = ModelSaveCallback(arguments.model_dest)
callbacks = [save_callback]

# prep params
parser = Model(model, decoder="eisner", loss="crossentropy", optimizer="adam", vocab=vocab)

parser.train(train_batches, arguments.dev, dev_batches, epochs=n_epochs, callbacks=callbacks)

parser.load_from_file(arguments.model_dest)

metrics = parser.evaluate(arguments.test, test_data)
test_UAS = metrics["nopunct_uas"]
test_LAS = metrics["nopunct_las"]

print(metrics)

print()
print(">>> Model maxed on dev at epoch", save_callback.best_epoch)
print(">>> Test score:", test_UAS, test_LAS)

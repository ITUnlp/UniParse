import argparse

from uniparse import Vocabulary
from uniparse import Model
from uniparse.callbacks import TensorboardLoggerCallback
from uniparse.callbacks import ModelSaveCallback

from uniparse.dataprovider import batch_by_buckets
from uniparse.dataprovider import scale_batch

from uniparse.models.dozatv2 import BaseParser

argparser = argparse.ArgumentParser()

argparser.add_argument("--train", dest="train", help="Annotated CONLL train file", metavar="FILE", required=True)
argparser.add_argument("--dev", dest="dev", help="Annotated CONLL dev file", metavar="FILE", required=True)
argparser.add_argument("--test", dest="test", help="Annotated CONLL dev test", metavar="FILE", required=True)
argparser.add_argument("--epochs", dest="epochs", type=int, default=30)
argparser.add_argument("--vocab_dest", dest="vocab_dest")
argparser.add_argument("--model_dest", dest="model_dest", required=True)

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

#train_batches = scale_batch(train_data, scale=1000, cluster_count=40, padding_token=vocab.PAD, shuffle=True)

# idx, batches = train_batches
# for (words, tags), (gold_arc, gold_rel) in batches:
#      print(words.T.shape)

train_batches = batch_by_buckets(train_data, batch_size=32, shuffle=True)
dev_batches = batch_by_buckets(dev_data, batch_size=32, shuffle=True)
test_batches = batch_by_buckets(test_data, batch_size=32, shuffle=False)


# instantiate model
""" """
model = BaseParser(vocab, word_dims, tag_dims,
                   dropout_emb, lstm_layers,
                   lstm_hiddens, dropout_lstm_input, dropout_lstm_hidden,
                   mlp_arc_size, mlp_rel_size, dropout_mlp, None, False)

save_callback = ModelSaveCallback(arguments.model_dest)
callbacks = [save_callback]

# prep params
parser = Model(model, decoder="eisner", loss="crossentropy", optimizer="adam", vocab=vocab)

parser.train(train_batches, arguments.dev, dev_batches, epochs=n_epochs, callbacks=callbacks)
# parser.load_from_file(arguments.model_dest)

metrics = parser.evaluate(arguments.test, test_data)
test_UAS = metrics["nopunct_uas"]
test_LAS = metrics["nopunct_las"]

print(metrics)

print()
print(">>> Model maxed on dev at epoch", save_callback.best_epoch)
print(">>> Test score:", test_UAS, test_LAS)

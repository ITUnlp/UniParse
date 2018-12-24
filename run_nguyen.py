import argparse

from uniparse import Model
from uniparse import Vocabulary

from uniparse.callbacks import ModelSaveCallback
from uniparse.callbacks import TensorboardLoggerCallback

from uniparse.dataprovider import scale_batch
from uniparse.dataprovider import batch_by_buckets
from uniparse.dataprovider import batch_by_buckets_with_chars

from uniparse.models.dynet_models.nguyen import Nguyen

parser = argparse.ArgumentParser()

parser.add_argument("--train", required=True)
parser.add_argument("--dev", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--vocab_dest")
parser.add_argument("--model_dest", required=True)
parser.add_argument("--embs", required=True)

arguments, unknown = parser.parse_known_args()

n_epochs = arguments.epochs
vocab = Vocabulary()
vocab = vocab.fit(arguments.train, pretrained_embeddings=arguments.embs)
embeddings = vocab.load_embedding()

# save vocab for reproducability later
if arguments.vocab_dest:
    print("> saving vocab to", arguments.vocab_dest)
    vocab.save(arguments.vocab_dest)

# prep data
print(">> Loading in data")
train_data = vocab.tokenize_conll(arguments.train)
dev_data = vocab.tokenize_conll(arguments.dev)
test_data = vocab.tokenize_conll(arguments.test)

train_batches = batch_by_buckets_with_chars(train_data, batch_size=32, shuffle=True)
dev_batches = batch_by_buckets_with_chars(dev_data, batch_size=32, shuffle=True)
test_batches = batch_by_buckets_with_chars(test_data, batch_size=32, shuffle=False)

# instantiate model
model = Nguyen(vocab, embs=embeddings)

save_callback = ModelSaveCallback(arguments.model_dest)
callbacks = [save_callback]

# prep params
parser = Model(model, decoder="eisner", loss="hinge", optimizer="adam", vocab=vocab)

parser.train(train_batches, arguments.dev, dev_batches, epochs=n_epochs, callbacks=callbacks)
parser.load_from_file(arguments.model_dest)

metrics = parser.evaluate(arguments.test, test_batches)
test_UAS = metrics["nopunct_uas"]
test_LAS = metrics["nopunct_las"]

print(metrics)

print()
print(">>> Model maxed on dev at epoch", save_callback.best_epoch)
print(">>> Test score:", test_UAS, test_LAS)

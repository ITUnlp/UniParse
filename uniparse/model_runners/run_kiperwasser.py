import argparse

from uniparse import Vocabulary
from uniparse import Model

from uniparse.callbacks import ModelSaveCallback

from uniparse.dataprovider import batch_by_buckets

from uniparse.models.dynet.kiperwasser import Kiperwasser

parser = argparse.ArgumentParser()

parser.add_argument("--train", required=True)
parser.add_argument("--dev", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--model", required=True)
parser.add_argument("--vocab")

arguments, unknown = parser.parse_known_args()

n_epochs = arguments.epochs
vocab = Vocabulary()
vocab = vocab.fit(arguments.train)

# save vocab for reproducability later
if arguments.vocab:
    print("> saving vocab to", arguments.vocab)
    vocab.save(arguments.vocab)

# prep data
print(">> Loading in data")
train_data = vocab.tokenize_conll(arguments.train)
dev_data = vocab.tokenize_conll(arguments.dev)
test_data = vocab.tokenize_conll(arguments.test)


train_batches = batch_by_buckets(train_data, batch_size=32, shuffle=True)
dev_batches = batch_by_buckets(dev_data, batch_size=32, shuffle=True)
test_batches = batch_by_buckets(test_data, batch_size=32, shuffle=False)

# instantiate model
model = Kiperwasser(vocab)

save_callback = ModelSaveCallback(arguments.model)
callbacks = [save_callback]

# prep params
parser = Model(model, optimizer="adam", vocab=vocab)

# todo ala
# parser = Model(model, optimizer="adam", vocab=vocab)

parser.train(train_batches, arguments.dev, dev_batches, epochs=n_epochs, callbacks=callbacks, verbose=True)
parser.load_from_file(arguments.model)

metrics = parser.evaluate(arguments.test, test_batches)
test_UAS = metrics["nopunct_uas"]
test_LAS = metrics["nopunct_las"]

print(metrics)

print()
print(">>> Model maxed on dev at epoch", save_callback.best_epoch)
print(">>> Test score:", test_UAS, test_LAS)

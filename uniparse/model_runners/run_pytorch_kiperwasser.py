import argparse

from uniparse.callbacks import ModelSaveCallback

from uniparse.dataprovider import batch_by_buckets

from uniparse.vocabulary import Vocabulary
from uniparse.models.pytorch_models.kiperwasser import Kiperwasser

from uniparse import Model

parser = argparse.ArgumentParser()

parser.add_argument("--train", dest="train", help="Annotated CONLL train file", metavar="FILE", required=True)
parser.add_argument("--dev", dest="dev", help="Annotated CONLL dev file", metavar="FILE", required=True)
parser.add_argument("--test", dest="test", help="Annotated CONLL dev test", metavar="FILE", required=True)
parser.add_argument("--model", dest="model", required=True)
arguments, unknown = parser.parse_known_args()

vocab = Vocabulary()
vocab = vocab.fit(arguments.train)

# prep data
training_data = vocab.tokenize_conll(arguments.train)
dev_data = vocab.tokenize_conll(arguments.dev)
test_data = vocab.tokenize_conll(arguments.test)

# batch data
train_batches = batch_by_buckets(training_data, batch_size=32, shuffle=True)
dev_batches = batch_by_buckets(dev_data, batch_size=32, shuffle=True)
test_batches = batch_by_buckets(test_data, batch_size=32, shuffle=False)

model = Kiperwasser(vocab)

save_callback = ModelSaveCallback(arguments.model)

# prep params
parser = Model(model, optimizer="adam", vocab=vocab)

parser.train(train_batches, arguments.dev, dev_batches, epochs=30, callbacks=[save_callback], verbose=True)

# load best model
model.load_from_file(arguments.model)

metrics = parser.evaluate(arguments.test, test_batches)

test_UAS = metrics["nopunct_uas"]
test_LAS = metrics["nopunct_las"]

print(metrics)

import uniparse.evaluation.universal_eval as universal_eval

universal_eval.evaluate_files(file_a, file_b)
universal_eval.perl_eval(file_a, file_b)
universal_eval.conll17_eval(file_a, file_b)

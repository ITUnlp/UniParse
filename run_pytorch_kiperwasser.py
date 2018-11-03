import argparse


from uniparse.callbacks import ModelSaveCallback

from uniparse import Model
from uniparse.vocabulary import Vocabulary
from uniparse.models.pytorch_kiperwasser import DependencyParser

parser = argparse.ArgumentParser()

parser.add_argument("--train", dest="train", help="Annotated CONLL train file", metavar="FILE", required=True)
parser.add_argument("--dev", dest="dev", help="Annotated CONLL dev file", metavar="FILE", required=True)
parser.add_argument("--test", dest="test", help="Annotated CONLL dev test", metavar="FILE", required=True)
parser.add_argument("--decoder", dest="decoder", required=True)
parser.add_argument("--model", dest="model", required=True)

parser.add_argument("--batch_size", default=32)
parser.add_argument("--epochs", default=15)

arguments, unknown = parser.parse_known_args()

vocab = Vocabulary()
vocab = vocab.fit(arguments.train)

# prep data
training_data = vocab.tokenize_conll(arguments.train)
dev_data = vocab.tokenize_conll(arguments.dev)
test_data = vocab.tokenize_conll(arguments.test)


model = DependencyParser(vocab)

save_callback = ModelSaveCallback(arguments.model)

# prep params
parser = Model(
    model, decoder=arguments.decoder, loss="hinge", optimizer="adam", strategy="bucket", vocab=vocab)


parser.train(
    training_data, arguments.dev, dev_data,
    epochs=arguments.epochs, batch_size=arguments.batch_size, callbacks=[save_callback]
)

# load best model
model.load_from_file(arguments.model)

metrics = parser.evaluate(arguments.test, test_data, batch_size=arguments.batch_size)
test_UAS = metrics["nopunct_uas"]
test_LAS = metrics["nopunct_las"]

print(metrics)

"""Generic run implementation that loads in dependency parser model and trains it."""

from uniparse import Model, Vocabulary

from uniparse.callbacks import ModelSaveCallback
from uniparse.dataprovider import batch_by_buckets
from uniparse.backend.dynet_backend import crossentropy, kiperwasser_loss, hinge
from uniparse.decoders import eisner, cle


def train(
    train_file,
    dev_file,
    test_file,
    n_epochs,
    parameter_file,
    vocab_file,
    model_class,
    batch_size=32,
    extras={},
):
    """Training procedure."""
    vocab = Vocabulary()
    vocab = vocab.fit(train_file)

    # save vocab for reproducability later
    if vocab_file:
        print("> saving vocab to", vocab_file)
        vocab.save(vocab_file)

    # prep data
    print(">> Loading in data")
    train_data = vocab.tokenize_conll(train_file)
    dev_data = vocab.tokenize_conll(dev_file)
    test_data = vocab.tokenize_conll(test_file)

    train_batches = batch_by_buckets(train_data, batch_size=batch_size, shuffle=True)
    dev_batches = batch_by_buckets(dev_data, batch_size=batch_size, shuffle=False)
    test_batches = batch_by_buckets(test_data, batch_size=batch_size, shuffle=False)

    model = model_class(vocab, extras=extras)

    save_callback = ModelSaveCallback(parameter_file)
    callbacks = [save_callback]

    # prep params
    loss = [kiperwasser_loss, hinge]
    parser = Model(model, optimizer="adam", vocab=vocab, loss=loss, decoder=eisner)

    parser.train(
        train_batches,
        dev_file,
        dev_batches,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=True,
    )
    parser.load_from_file(parameter_file)

    metrics = parser.evaluate(test_file, test_batches, delete_output=False)
    test_UAS = metrics["nopunct_uas"]
    test_LAS = metrics["nopunct_las"]

    print(metrics)

    print()
    print(">>> Model maxed on dev at epoch", save_callback.best_epoch)
    print(">>> Test score:", test_UAS, test_LAS)


def run(sample_file, parameter_file, vocab_file, model_class, batch_size=32):
    """Run generic parser defined in implementation, parameters, and vocab."""
    vocab = Vocabulary().load(vocab_file)

    model = model_class(vocab)

    print(">> Loading in data")
    sample_data = vocab.tokenize_conll(sample_file)

    sample_batches = batch_by_buckets(sample_data, batch_size=batch_size, shuffle=False)

    # prep params
    parser = Model(model, optimizer="adam", vocab=vocab)

    parser.load_from_file(parameter_file)

    metrics = parser.evaluate(sample_file, sample_batches, delete_output=False)
    test_UAS = metrics["nopunct_uas"]
    test_LAS = metrics["nopunct_las"]

    print(metrics)

    print(">>> Test score:", test_UAS, test_LAS)

"""Model class. Wraps a parser model and facilities training, and evaluation."""

import os
import sys
import time
import tempfile

from tqdm import tqdm

# TODO: Remove this block
try:
    # attempt to import to ensure early failure.
    import uniparse.decoders as _decoders
except Exception as e:
    print(
        ">> ERROR: can't import decoders. please run 'python setup.py build_ext --inplace' or 'pip install .' from the root directory"
    )
    raise e

from uniparse.backend import get_backend_from_params
from uniparse.backend import get_backend_from_string
import uniparse.evaluation.universal_eval as uni_eval

import numpy as np
import sklearn.utils


class Model:
    """
    Training procedure class for training dependency parsers.

    Facilitates the training loop, evaluation as well as custom behavior through callbacks.
    """

    def __init__(
        self,
        model,
        decoder=None,
        loss=None,
        optimizer=None,
        strategy=None,
        vocab=None,
        backend=None,
    ):
        """Instantiate Model class."""
        self._parser = model
        self._optimizer = None
        self._vocab = vocab

        # deprecated functionality
        if strategy:
            print(">> DEPRECATED: automatic batching is deprecated. Re-evaluate??")

        self._loss = loss
        self._decoder = [_decode_arcs, _decode_rels]

        # configure backend either based on user input
        # or infer it from the models parameters
        if backend:
            backend = get_backend_from_string(backend)
        else:
            backend = get_backend_from_params(model.parameters())

        self.backend = backend

        # extract optimizer / decoder / loss from strings
        if isinstance(optimizer, str):
            optimizer = self._get_optimizer(optimizer)
            self._optimizer = optimizer(model.parameters())
        else:
            self._optimizer = optimizer

    def _get_optimizer(self, input_optimizer):
        """Get optimizer from string."""
        backend = self.backend
        if isinstance(input_optimizer, str):
            optimizer_options = {
                "adam": backend.optimizers.adam,
                "rmsprop": backend.optimizers.rmsprop,
                "adadelta": backend.optimizers.adadelta,
                "adagrad": backend.optimizers.adagrad,
                "sgd": backend.optimizers.sgd,
            }

            if input_optimizer not in optimizer_options:
                raise ValueError("optimizer doesn't exist")

            return optimizer_options[input_optimizer]
        else:
            return input_optimizer

    # Huge TODO: get dev_file dependency out, so that the training procedure doesn't require file names.
    # OBS. Requires implementing in-memory evaluatation.
    def train(self, train, dev_file, dev, epochs, callbacks=None, verbose=True):
        """Train model."""
        callbacks = callbacks if callbacks else []

        _, samples = train

        backend = self.backend
        global_step = 0
        for epoch in range(1, epochs + 1):

            samples = sklearn.utils.shuffle(samples)

            it_samples = tqdm(samples) if verbose else samples
            epoch_time = time.time()
            for x, y in it_samples:
                backend.renew_cg()

                # words, lemmas, tags, chars = x
                words = x[0]
                # PAD = 0; ROOT = 1; OOV = 2; UNK = 2
                # Tokens > 1 are valid tokens we want to compute loss for use for accuracy metrics
                mask = np.greater(words, self._vocab.ROOT)

                output = self._parser((x, y), decoder=self._decoder)

                # loss functions adhere to the type ::
                # score: tensor -> y_hat: 2d_ndarray, y: 2d_ndarray, mask  2d_ndarray
                losses = []
                predictions = []
                for s, _y, l, d in zip(output, y, self._loss, self._decoder):
                    y_hat = d(s)  # decode
                    loss_i = l(s, y_hat, _y, mask)  # compute loss
                    losses.append(loss_i)
                    predictions.append(y_hat)

                loss = sum(losses)

                loss_value = backend.get_scalar(loss)
                loss.backward()

                backend.step(self._optimizer)

                uas, las = [
                   _defaualt_metric(y_hat, _y, mask) for y_hat, _y in zip(predictions, y)
                ]

                if verbose:
                    metric_tuple = (epoch, epochs, float(uas), float(las), loss_value)
                    it_samples.set_description(
                        "[%d/%d] arc %.2f, rel %.2f, loss %.3f" % metric_tuple
                    )

                global_step += 1

            metrics = self.evaluate(dev_file, dev)
            no_punct_dev_uas = metrics["nopunct_uas"]
            no_punct_dev_las = metrics["nopunct_las"]
            # punct_dev_uas = metrics["uas"]
            # punct_dev_las = metrics["las"]

            epoch_time = int(time.time() - epoch_time)
            print(
                "[%d][%ds] %0.5f, %0.5f "
                % (epoch, epoch_time, no_punct_dev_uas, no_punct_dev_las)
            )
            sys.stdout.flush()  # for python 2.7 compatibility

            batch_end_info = {
                "dev_uas": no_punct_dev_uas,
                "dev_las": no_punct_dev_las,
                "global_step": global_step,
                "model": self._parser,
            }

            for callback in callbacks:
                callback.on_epoch_end(epoch, batch_end_info)

        print(">> Finished at epoch %d" % epoch)

    def evaluate(self, test_file, test_data, delete_output=True):
        """Evaluate model on data. Requires data file name as well."""
        # TODO: in memory evaluation

        predictions = self.run(test_data)

        _, output_file = tempfile.mkstemp()
        uni_eval.write_predictions_to_file(
            predictions,
            reference_file=test_file,
            output_file=output_file,
            vocab=self._vocab,
        )

        metrics = uni_eval.evaluate_files(output_file, test_file)

        if delete_output:
            os.system("rm %s" % output_file)
        else:
            print(">> outputed predictions to %s" % output_file)

        return metrics

    def run(self, samples):
        """Run model on samples. Returns flat list of predictions."""
        indices, batches = samples

        predictions = []
        for idx, (x, y) in zip(indices, batches):
            self.backend.renew_cg()

            output = self._parser((x, (None, None)))
            arc_preds, rel_preds = [d(o) for o, d in zip(output, self._decoder)]

            # slicing to avoid including the root token
            for i, arc, rel in zip(idx, arc_preds, rel_preds):
                predictions.append((i, arc[1:], rel[1:]))

        predictions.sort(key=lambda tup: tup[0])

        return predictions

    def save_to_file(self, filename):
        """Save model to file."""
        # TODO: Save parameters and vocabulary in a united blob.
        self._parser.save_to_file(filename)

    def load_from_file(self, filename):
        """Load model from file."""
        # TODO: Load parameters and vocabulary in a united blob.
        self._parser.load_from_file(filename)


def _defaualt_metric(y_h, y, mask):
    num_tokens = int(np.sum(mask))
    correct_rels = np.equal(y_h, y).astype(np.float32) * mask
    las = np.sum(correct_rels) / num_tokens
    return las


# TODO: DyNet specific! Move to backend
def _decode_arcs(arc_scores):
    numpy_scores = np.atleast_3d(arc_scores.npvalue())
    numpy_scores = np.moveaxis(numpy_scores, -1, 0)
    return _decoders.eisner(numpy_scores)


# TODO: DyNet specific! Move to backend
def _decode_rels(rel_scores):
    predicted_rels = np.transpose(rel_scores.npvalue().argmax(0))
    predicted_rels = np.atleast_2d(predicted_rels)
    return predicted_rels

import os
import sys
import time
import ntpath
from typing import *

from uniparse.types import Parser
from uniparse.dataprovider import ScaledBatcher, BucketBatcher

try:
    import uniparse.decoders as decoders
except Exception as e:
    print(">> ERROR: can't import decoders. please run 'python setup.py build_ext --inplace' from the root directory")
    raise e

import uniparse.backend as backend_wrapper
import uniparse.evaluation.universal_eval as uni_eval

import numpy as np
import sklearn.utils


class Model(object):
    def __init__(self, model: Parser, decoder, loss, optimizer, strategy, vocab):
        self._model_uid = time.strftime("%m%d%H%M%S")
        self._parser = model
        self._optimizer = None
        self._vocab = vocab
        self._batch_strategy = strategy

        # retrieve backend wrapper
        self.backend = backend_wrapper.init_backend(model.get_backend_name())
        model.set_backend(self.backend)

        # extract optimizer / decoder / loss from strings
        if isinstance(optimizer, str):
            optimizer = self._get_optimizer(optimizer)
            self._optimizer = optimizer(model.parameters())
        else:
            self._optimizer = optimizer

        # extract decoder
        runtime_decoder = self._get_decoder(decoder)
        self._parser.set_decoder(runtime_decoder)

        # extract loss functions
        self.arc_loss, self.rel_loss = self._get_loss_functions(loss)

    def _get_optimizer(self, input_optimizer):
        # get setup optimizer
        backend = self.backend
        if isinstance(input_optimizer, str):
            optimizer_options = {
                "adam": backend.optimizers.adam,
                "rmsprop": backend.optimizers.rmsprop,
                "adadelta": backend.optimizers.adadelta,
                "adagrad": backend.optimizers.adagrad
            }

            if input_optimizer not in optimizer_options:
                raise ValueError("optimizer doesn't exist")

            return optimizer_options[input_optimizer]
        else:
            return input_optimizer

    @staticmethod
    def _get_decoder(input_decoder):
        if isinstance(input_decoder, str):
            decoder_options = {
                "eisner": decoders.eisner,
                "cle": decoders.cle
            }

            if input_decoder not in decoder_options:
                raise ValueError("decoder (%s) not available" % input_decoder)

            return decoder_options[input_decoder]
        else:
            return input_decoder

    def _get_loss_functions(self, input_loss: Union[str, Tuple[Any, Any]]):
        if isinstance(input_loss, str):
            loss = self.backend.loss
            loss_options = {
                # included for completeness
                "crossentropy": (loss.crossentropy, loss.crossentropy),
                "kiperwasser": (loss.hinge, loss.kipperwasser_hinge),
                "hinge": (loss.hinge, loss.hinge)
            }
            if input_loss not in loss_options:
                raise ValueError("unknown loss function %s" % input_loss)

            return loss_options[input_loss]
        else:
            return input_loss

    def _batch_data(self, samples: List, strategy: str, scale: int, shuffle: bool):
        if strategy == "bucket":
            dataprovider = BucketBatcher(samples, padding_token=self._vocab.PAD)
            _idx, _sentences = dataprovider.get_data(scale, shuffle)
        elif strategy == "scaled_batch":
            dataprovider = ScaledBatcher(samples, cluster_count=40, padding_token=self._vocab.PAD)
            _idx, _sentences = dataprovider.get_data(scale, shuffle)
        else:
            raise ValueError("no such data strategy")

        return _idx, _sentences
    
    def run(self, samples: List, batch_size: int):
        indices, batches = self._batch_data(samples, strategy=self._batch_strategy, scale=batch_size, shuffle=False)
        backend = self.backend

        predictions = []
        for indicies, (x, y) in zip(indices, batches):
            backend.renew_cg()

            words, lemmas, tags, chars = x

            words = backend.input_tensor(words, dtype="int")
            tags = backend.input_tensor(tags, dtype="int")
            lemmas = backend.input_tensor(lemmas, dtype="int")

            arc_preds, rel_preds, _, _ = self._parser((words, lemmas, tags, None, None, chars))

            outs = [(ind, arc[1:], rel[1:]) for ind, arc, rel in zip(indicies, arc_preds, rel_preds)]

            predictions.extend(outs)

        predictions.sort(key=lambda tup: tup[0])

        return predictions

    def train(self, train: List, dev_file: str, dev: List, epochs: int, batch_size: int, callbacks: List = None):
        callbacks = callbacks if callbacks else []  # This is done to avoid using the same list.

        training_data = self._batch_data(train, strategy=self._batch_strategy, scale=batch_size, shuffle=True)

        backend = self.backend
        _, samples = training_data
        global_step = 0
        for epoch in range(1, epochs+1):
            start = time.time()

            samples = sklearn.utils.shuffle(samples)

            print(f"> Epoch {epoch}")
            print("=====================")
            for step, (x, y) in enumerate(samples):
                # renew graph
                backend.renew_cg()

                words, lemmas, tags, chars = x
                gold_arcs, gold_rels = y

                batch_size, n = words.shape

                # PAD = 0; ROOT = 1; OOV = 2; UNK = 2
                # Tokens > 1 are valid tokens we want to compute loss for use for accuracy metrics
                mask = np.greater(words, self._vocab.ROOT)
                num_tokens = int(np.sum(mask))

                """ this is necesary for satisfy compatibility with pytorch """
                words = backend.input_tensor(words, dtype="int")
                postags = backend.input_tensor(tags, dtype="int")
                lemmas = backend.input_tensor(lemmas, dtype="int")

                arc_preds, rel_preds, arc_scores, rel_scores = self._parser(
                    (words, lemmas, postags, gold_arcs, gold_rels, chars)
                )

                arc_loss = self.arc_loss(arc_scores, arc_preds, gold_arcs, mask)
                rel_loss = self.rel_loss(rel_scores, None, gold_rels, mask)


                loss = arc_loss + rel_loss
                loss_value = backend.get_scalar(loss)
                loss.backward()

                backend.step(self._optimizer)

                arc_correct = np.equal(arc_preds, gold_arcs).astype(np.float32) * mask
                arc_accuracy = np.sum(arc_correct) / num_tokens

                rel_correct = np.equal(rel_preds, gold_rels).astype(np.float32) * mask
                rel_accuracy = np.sum(rel_correct) / num_tokens

                training_info = {
                    "arc_accuracy": arc_accuracy,
                    "rel_accuracy": rel_accuracy,
                    "arc_loss": backend.get_scalar(arc_loss),
                    "rel_loss": backend.get_scalar(rel_loss),
                    "global_step": global_step
                }

                for callback in callbacks:
                    callback.on_batch_end(training_info)

                sys.stdout.write(
                    "\r\rStep #%d: Acc: arc %.2f, rel %.2f, loss %.3f"
                    % (global_step, float(arc_accuracy), float(rel_accuracy), loss_value)
                )
                sys.stdout.flush()

                global_step += 1

            print()
            print(f">> Completed epoch {epoch} in ", time.time()-start)
            metrics = self.evaluate(dev_file, dev, batch_size)
            no_punct_dev_uas = metrics["nopunct_uas"]
            no_punct_dev_las = metrics["nopunct_las"]

            punct_dev_uas = metrics["uas"]
            punct_dev_las = metrics["las"]
            print(f">> UAS (wo. punct) {no_punct_dev_uas:.{5}}\t LAS (wo. punct) {no_punct_dev_las:.{5}}")
            print(f">> UAS (w. punct) {punct_dev_uas:.{5}}\t LAS (w. punct) {punct_dev_las:.{5}}")

            batch_end_info = {
                "dev_uas": no_punct_dev_uas,
                "dev_las": no_punct_dev_las,
                "global_step": global_step,
                "model": self._parser
            }

            for callback in callbacks:
                callback.on_epoch_end(epoch, batch_end_info)

            print()

        print(f">> Finished at epoch {epoch}")

    def evaluate(self, test_file: str, test_data: List, batch_size: int):
        stripped_filename = ntpath.basename(test_file)
        output_file = f"{self._model_uid}_on_{stripped_filename}"

        # run parser on data
        predictions = self.run(test_data, batch_size)

        # write to file
        uni_eval.write_predictions_to_file(
            predictions, reference_file=test_file, output_file=output_file, vocab=self._vocab)

        metrics = uni_eval.evaluate_files(output_file, test_file)

        os.system("rm %s" % output_file)

        return metrics

    def save_to_file(self, filename: str) -> None:
        self._parser.save_to_file(filename)

    def load_from_file(self, filename: str) -> None:
        self._parser.load_from_file(filename)

import os
import sys
import time
import ntpath
import tempfile

from tqdm import tqdm

try:
    import uniparse.decoders as decoders
except Exception as e:
    print(">> ERROR: can't import decoders. please run 'python setup.py build_ext --inplace' from the root directory")
    raise e

import uniparse.backend as backend_wrapper
import uniparse.evaluation.universal_eval as uni_eval

import numpy as np
import sklearn.utils

import inspect


def infer_backend(parameters):
    try:
        import dynet as dy
        if isinstance(parameters, dy.ParameterCollection):
            return "dynet"
    except:
        pass

    import torch
    if inspect.isgenerator(parameters):
        return "pytorch"

    raise ValueError("couldn't infer backend")


class Model(object):
    def __init__(self, model, decoder=None, loss=None, optimizer=None, strategy=None, vocab=None, backend=None):
        self._model_uid = time.strftime("%m%d%H%M%S")
        self._parser = model
        self._optimizer = None
        self._vocab = vocab
        self._batch_strategy = strategy

        if strategy:
            print("DEPRECATED: model batching. In the future batch the data own your own and pass it to the model.")

        if decoder:
            print("Decoder argument deprecated")

        if loss:
            print("Loss function argument deprecated")

        if backend:
            # provided by user
            backend = backend_wrapper.init_backend(backend_name)
        else:
            # infer backend through parameters
            model_parameters = model.parameters()
            backend_name = infer_backend(model_parameters)
            backend = backend_wrapper.init_backend(backend_name)

        # i want to move this away form the model. the model knows which backend its using
        model.set_backend(backend)
        self.backend = backend

        # retrieve backend wrapper
        # self.backend = backend_wrapper.init_backend(model.get_backend_name())
        # model.set_backend(self.backend)

        # extract optimizer / decoder / loss from strings
        if isinstance(optimizer, str):
            optimizer = self._get_optimizer(optimizer)
            self._optimizer = optimizer(model.parameters())
        else:
            self._optimizer = optimizer

        if decoder:
            # extract decoder
            runtime_decoder = self._get_decoder(decoder)
            self._parser.set_decoder(runtime_decoder)
            self._runtime_decoder = runtime_decoder

        # extract loss functions
        if loss:
            self.arc_loss, self.rel_loss = self._get_loss_functions(loss)
            self._parser.set_loss_function(lambda a, b, c, d, e: self.arc_loss(a, None, c, e) + self.rel_loss(
                b, None, d, e))
            self._parser.set_loss_object(self.backend.loss)

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
            decoder_options = {"eisner": decoders.eisner, "cle": decoders.cle}

            if input_decoder not in decoder_options:
                raise ValueError("decoder (%s) not available" % input_decoder)

            return decoder_options[input_decoder]
        else:
            return input_decoder

    def _get_loss_functions(self, input_loss):
        if isinstance(input_loss, str):
            loss = self.backend.loss
            loss_options = {
                # included for completeness
                "crossentropy": (loss.crossentropy, loss.crossentropy),
                "kiperwasser": (loss.hinge, loss.kiperwasser_hinge),
                "hinge": (loss.hinge, loss.hinge)
            }
            if input_loss not in loss_options:
                raise ValueError("unknown loss function %s" % input_loss)

            return loss_options[input_loss]
        else:
            return input_loss

    def train(self, train, dev_file, dev, epochs, callbacks=None, verbose=True):
        callbacks = callbacks if callbacks else []  # This is done to avoid using the same list.

        _, samples = train

        backend = self.backend
        global_step = 0
        for epoch in range(1, epochs + 1):

            samples = sklearn.utils.shuffle(samples)

            it_samples = tqdm(samples) if verbose else samples
            epoch_time = time.time()
            for x, y in it_samples:

                # renew graph
                backend.renew_cg()

                # words, lemmas, tags, chars = x
                gold_arcs, gold_rels = y
                words = x[0]

                # wat
                if len(words) < 1:
                    print("n words are less than 1.. whats happening")
                    continue

                # PAD = 0; ROOT = 1; OOV = 2; UNK = 2
                # Tokens > 1 are valid tokens we want to compute loss for use for accuracy metrics
                mask = np.greater(words, self._vocab.ROOT)
                num_tokens = int(np.sum(mask))

                pred_arcs, pred_rels, loss = self._parser((x, y))

                loss_value = backend.get_scalar(loss)
                loss.backward()  # backward compute

                backend.step(self._optimizer)

                arc_correct = np.equal(pred_arcs, gold_arcs).astype(np.float32) * mask
                arc_accuracy = np.sum(arc_correct) / num_tokens

                rel_correct = np.equal(pred_rels, gold_rels).astype(np.float32) * mask
                rel_accuracy = np.sum(rel_correct) / num_tokens

                if verbose:
                    metric_tuple = (epoch, epochs, float(arc_accuracy), float(rel_accuracy), loss_value)
                    it_samples.set_description("[%d/%d] arc %.2f, rel %.2f, loss %.3f" % metric_tuple)

                global_step += 1

            metrics = self.evaluate(dev_file, dev)
            no_punct_dev_uas = metrics["nopunct_uas"]
            no_punct_dev_las = metrics["nopunct_las"]
            #punct_dev_uas = metrics["uas"]
            #punct_dev_las = metrics["las"]

            epoch_time = int(time.time() - epoch_time)
            print("[%d][%ds] %0.5f, %0.5f " % (epoch, epoch_time, no_punct_dev_uas, no_punct_dev_las))
            sys.stdout.flush()  # for python 2.7 compatibility

            # remove callbacks

            batch_end_info = {
                "dev_uas": no_punct_dev_uas,
                "dev_las": no_punct_dev_las,
                "global_step": global_step,
                "model": self._parser
            }

            for callback in callbacks:
                callback.on_epoch_end(epoch, batch_end_info)

        print(">> Finished at epoch %d" % epoch)

    def evaluate(self, test_file, test_data):
        #stripped_filename = ntpath.basename(test_file)
        _, stripped_filename = tempfile.mkstemp()
        output_file = stripped_filename
        #output_file = "%s_on_%s" % (self._model_uid, stripped_filename)

        # run parser on data
        predictions = self.run(test_data)

        # write to file
        uni_eval.write_predictions_to_file(
            predictions, reference_file=test_file, output_file=output_file, vocab=self._vocab)

        metrics = uni_eval.evaluate_files(output_file, test_file)

        os.system("rm %s" % output_file)

        return metrics

    def run(self, samples):
        indices, batches = samples

        backend = self.backend

        predictions = []
        for idx, (x, y) in zip(indices, batches):
            backend.renew_cg()

            # words = backend.input_tensor(words, dtype="int")
            # tags = backend.input_tensor(tags, dtype="int")

            # arc_preds, rel_preds, pos_preds, _ = self._parser((x, (None, None)))
            arc_preds, rel_preds, _ = self._parser((x, (None, None)))

            outs = [(ind, arc[1:], rel[1:]) for ind, arc, rel in zip(idx, arc_preds, rel_preds)]

            predictions.extend(outs)

        predictions.sort(key=lambda tup: tup[0])

        return predictions

    def save_to_file(self, filename):
        self._parser.save_to_file(filename)

    def load_from_file(self, filename):
        self._parser.load_from_file(filename)

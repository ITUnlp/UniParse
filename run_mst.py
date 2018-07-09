import os
import time
import argparse
from typing import *

import numpy as np
import sklearn.utils

from uniparse import Vocabulary
import uniparse.evaluation.universal_eval as uni_eval


from uniparse.models.mst import MST
from uniparse.models.mst_encode import BetaEncodeHandler


def pre_encode(h, xs, accumulate_vocab=False):
    if accumulate_vocab:
        encoder.unlock_feature_space()

    encoded_dataset = []
    for _step, (words, lemmas, tags, heads, rels, chars) in enumerate(xs):

        ws = np.array(words, dtype=np.uint64)
        tags = np.array(tags, dtype=np.uint64)
        target_arcs = np.array(heads, dtype=np.int64)
        target_rels = np.array(rels, dtype=np.int64)

        # we do this to initialize the encoders vocab.
        _, _ = h(ws, tags, target_arcs, target_rels)
        
        # this is needed 
        encoded_samples = (ws, tags, target_arcs, target_rels)
        encoded_dataset.append(encoded_samples)

    # gotta do this to ensure not adding more stuff
    if accumulate_vocab:
        encoder.lock_feature_space()

    return encoded_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True)
parser.add_argument("--dev", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--model", required=True)

arguments, unknown = parser.parse_known_args()

TRAIN_FILE = arguments.train
DEV_FILE = arguments.dev
TEST_FILE = arguments.test
MODEL_FILE = arguments.model
n_epochs = 5

vocab = Vocabulary()
vocab.fit(TRAIN_FILE)

print(">> Loading in data")
TRAIN = vocab.tokenize_conll(arguments.train)
DEV = vocab.tokenize_conll(arguments.dev)
TEST = vocab.tokenize_conll(arguments.test)

encoder = BetaEncodeHandler()
print("> pre-encoding edges")
s = time.time()
TRAIN = pre_encode(encoder, TRAIN, accumulate_vocab=True)
DEV = pre_encode(encoder, DEV)
TEST = pre_encode(encoder, TEST)
print(">> done pre-encoding", time.time()-s)

# 5m is completely arbitrary
# REQUEST: fix this to be inferred from the encoder
parser = MST(5_000_000)


def evaluate(gold_file: str, data: List, name: str):

    predictions = []
    for step, (forms, tags, target_arcs, target_rels) in enumerate(data):
        dev_arc_edges, dev_rel_edges = encoder(forms, tags)
        dev_pred_arcs, dev_pred_rels, _, _ = parser(dev_arc_edges, dev_rel_edges, None, None)
        ns = [step] * len(dev_pred_arcs[1:])
        prediction_tuple = (ns, dev_pred_arcs[1:], dev_pred_rels[1:])
        predictions.append(prediction_tuple)

    output_file = "dev_epoch_%s.txt" % name
    uni_eval.write_predictions_to_file(
        predictions, reference_file=gold_file, output_file=output_file, vocab=vocab)

    metrics = uni_eval.evaluate_files(output_file, gold_file)
    os.system("rm %s" % output_file)

    return predictions, metrics


def train(training_data: List, dev_data_file: str, dev_data: List, epochs: int, model_param_file: str):
    start = step_time = time.time()
    max_uas = 0.0
    uas, ras = [], []
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}")
        print("=======================")

        # shuffle the dataset on each epoch
        training_data = sklearn.utils.shuffle(training_data)

        for step, (forms, tags, target_arcs, target_rels) in enumerate(training_data):
            arc_edges, rel_edges = encoder(forms, tags)
            pred_arcs, pred_rels, sample_uas, sample_ras = parser(arc_edges, rel_edges, target_arcs, target_rels)

            uas.append(sample_uas)
            ras.append(sample_ras)
            if step % 500 == 0 and step > 0:
                mean_uas = np.mean(uas)
                mean_ras = np.mean(ras)
                print(f"> Step {step} UAS {mean_uas:.{3}} LAS: {mean_ras:.{3}} Time {time.time()-step_time:.{3}}")
                step_time = time.time()
                uas, ras = [], []

        # time to evaluate
        print(">> Done with epoch %s. Evaluating on dev..." % epoch)
        predictions, metrics = evaluate(dev_data_file, dev_data, str(epoch))
        print(">> dev epoch %d" % epoch)
        print(metrics)
        print()

        nopunct_uas = metrics["nopunct_uas"]
        if nopunct_uas > max_uas:
            np.save(model_param_file, parser.W)
            print(">> saved to", model_param_file)
            max_uas = nopunct_uas

    print(">> Finished. Time spent", time.time()-start)


# train model
train(TRAIN, DEV_FILE, DEV, n_epochs, MODEL_FILE)

# populate with best parameters
parser.W = np.load("%s.npy" % MODEL_FILE)

print(">> Time to evaluate on test set")
predictions, test_metrics = evaluate(TEST_FILE, TEST, "test")
print(test_metrics)



""" An alternative evaluation implementation/script to the conll17 which evaluates all features """

import io
import os
import re
import sys
import time
import functools
import unicodedata
from subprocess import check_output

from uniparse import Vocabulary
from uniparse.evaluation.conll17_ud_eval import main

import numpy as np

# TODO !
# def relaxed_rel_equality():
#     correct_main_rel_category = gold_rel.split(":")[0] == pred_rel.split(":")[0]


class Sentence(object):
    __slots__ = ["id", "words", "tags", "heads", "rels", "n"]

    def __init__(self, _id, words, tags, heads, rels):
        self.id = _id
        self.n = len(words)
        self.words = words
        self.tags = tags
        self.heads = heads
        self.rels = rels

    def __iter__(self):
        return zip(self.words, self.tags, self.heads, self.rels)

    def __str__(self):
        tokens = []
        print("Id", self.id)
        for entry in [self.words, self.tags, self.heads, self.rels]:
            line = "\t".join(str(x) for x in entry)
            tokens.append(line)

        return "\n".join(tokens)


def invalid_line(line):
    if line == "\n":
        return False
    elif re.match(r'\d+\t', line):
        return False
    else:
        return True


def read_conll(filename):
    sentences = []
    words, tags, heads, rels = [], [], [], []
    with io.open(filename, encoding="UTF-8") as f:
        for line in f.readlines():
            # filter out comments and subphrase words
            if invalid_line(line):
                continue

            if line == "\n":
                sentence_number = len(sentences)
                # construct sentence and add to collection
                sentence = Sentence(sentence_number, words, tags, heads, rels)
                sentences.append(sentence)

                # clean up buffers
                words, tags, heads, rels = [], [], [], []
            else:
                info = line.strip().split("\t")
                assert len(info) == 10, "invalid line: %s" % line

                word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
                words.append(word)
                tags.append(tag)
                heads.append(head)
                rels.append(rel)

    return sentences


def is_unicode_token(token):
    unicode_words = [unicodedata.category(c).startswith('P') for c in token]
    return all(unicode_words)


def score_pair(left, right):
    tp_arc = np.array([own_head == other_head for own_head, other_head in zip(left.heads, right.heads)])
    tp_rel = np.array([own_rel == other_rel for own_rel, other_rel in zip(left.rels, right.rels)])
    tp_larc = np.logical_and(tp_arc, tp_rel)

    # generate mask v2
    no_puncts = [not is_unicode_token(w) for w in left.words]
    no_puncts = np.array(no_puncts, dtype=bool)

    # filter out punctuation
    nopunct_tp_arc = tp_arc[no_puncts]
    nopunct_tp_larc = tp_larc[no_puncts]

    return tp_arc, tp_larc, nopunct_tp_arc, nopunct_tp_larc


def compute_attachment_scores(gold_sentences, predicted_sentences, reduce=True):
    gold_sentences, predicted_sentences = list(gold_sentences), list(predicted_sentences)
    assert len(gold_sentences) == len(predicted_sentences), "sentences are not equal length"

    arc_predictions, larc_predictions = [], []
    nopunct_arc_predictions, nopunct_larc_predictions = [], []

    for gold_sentence, pred_sentence in zip(gold_sentences, predicted_sentences):
        # get score from pairs
        arcs, larcs, np_arcs, np_larcs = score_pair(gold_sentence, pred_sentence)

        # add to lists to sum the results
        arc_predictions.extend(arcs)
        larc_predictions.extend(larcs)
        nopunct_arc_predictions.extend(np_arcs)
        nopunct_larc_predictions.extend(np_larcs)

    # calculate precision
    if reduce:
        uas = np.mean(arc_predictions) if arc_predictions else 0.0
        las = np.mean(larc_predictions) if larc_predictions else 0.0
        nopunct_uas = np.mean(nopunct_arc_predictions) if nopunct_arc_predictions else 0.0
        nopunct_las = np.mean(nopunct_larc_predictions) if nopunct_larc_predictions else 0.0
    else:
        uas = arc_predictions,
        las = larc_predictions
        nopunct_uas = nopunct_arc_predictions
        nopunct_las = nopunct_larc_predictions

    return {
        "uas": uas,
        "las": las,
        "nopunct_uas": nopunct_uas,
        "nopunct_las": nopunct_las
    }


def evaluate_files(pred_file, gold_file, filter_f=None):
    pred_sentences = read_conll(pred_file)
    gold_sentences = read_conll(gold_file)

    if filter:
        gold_sentences = filter(filter_f, gold_sentences)
        pred_sentences = filter(filter_f, pred_sentences)

    return compute_attachment_scores(gold_sentences, pred_sentences)


# All credit to https://stackoverflow.com/a/40054132 - beatiful work
class _PrintSuppressor(object):
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, value_type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            pass
            # Do normal exception handling

    def write(self, x):
        pass


def _flat_map(lst):
    return functools.reduce(lambda x, y: x + y, [list(result) for result in lst])


def write_predictions_to_file(predictions, reference_file, output_file, vocab):
    indices, arcs, rels = zip(*predictions)
    flat_arcs = _flat_map(arcs)
    flat_rels = _flat_map(rels)

    idx = 0
    with io.open(reference_file, encoding="UTF-8") as f:
        with io.open(output_file, 'w', encoding="UTF-8") as fo:
            for line in f.readlines():
                if re.match(r'\d+\t', line):
                    info = line.strip().split()
                    assert len(info) == 10, 'Illegal line: %s' % line
                    info[6] = str(flat_arcs[idx])
                    info[7] = vocab.id2rel(flat_rels[idx])
                    fo.write('\t'.join(info) + '\n')
                    idx += 1
                else:
                    fo.write(line)


def perl_eval(test_file, pred_file):
    result_file = "tmp_%s" % str(time.time())
    os.system('perl uniparse/evaluation/eval.pl -q -b -g %s -s %s -o %s' % (test_file, pred_file, result_file))
    output = check_output(["tail", "-n" "3", result_file]).decode("utf-8")
    output = output.split("\n")[:2]
    las, uas = [line.strip().split()[-2] for line in output]
    os.system("rm %s" % result_file)
    return float(uas), float(las)


def conll17_eval(test_file, pred_file):
    with _PrintSuppressor():
        conll17_metrics = main(verbose=True, weights=None, gold_file=test_file, system_file=pred_file)
        uas = conll17_metrics["UAS"].precision * 100
        las = conll17_metrics["LAS"].precision * 100

        return uas, las


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--p", dest="prediction", help="Annotated CONLL gold file", metavar="FILE", required=True)
    parser.add_argument("--g", dest="gold", help="Annotated CONLL gold file", metavar="FILE", required=True)

    arguments, unknown = parser.parse_known_args()

    metrics = evaluate_files(arguments.prediction, arguments.gold)
    for k, v in metrics.items():
        print(k, v)







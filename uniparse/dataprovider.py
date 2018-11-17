from typing import Iterable
from typing import Callable
from typing import Dict

import math

from collections import defaultdict
from collections import namedtuple

import numpy as np
import sklearn.utils

from sklearn.cluster import KMeans


def gen_pad_3d(x, padding_token):
    batch_size = len(x)
    max_sentence_length = max(len(s) for s in x)
    max_word_length = max(len(word) for sentence in x for word in sentence)

    buffer_shape = (batch_size, max_sentence_length, max_word_length)
    buffer = np.full(buffer_shape, padding_token, dtype=np.int32)
    for i, sentence in enumerate(x):
        for j, char_ids in enumerate(sentence):
            buffer[i, j, :len(char_ids)] = char_ids

    return buffer


def gen_pad_2d(x: Iterable, padding_token):
    if not isinstance(x, list):
        x = list(x)

    batch_size = len(x)
    max_sentence_length = max(len(sentence) for sentence in x)
    buff = np.empty((batch_size, max_sentence_length))
    buff.fill(padding_token)
    for i, sentence in enumerate(x):
            buff[i, :len(sentence)] = sentence

    return buff


def split(seq, batch_size):
    return [
        seq[i:i + batch_size] 
        for i in range(0, len(seq), batch_size)
    ]


# Batch = namedtuple('Batch', ["x", "y"])
#
#
# def build_from_clusters(
#         scale_f: Callable, index_clusters: Dict[int, list], feature_clusters: Dict[int, list],
#         padding_token: int, shuffle=True):

#     batches, indicies = [], []
#     for len_key in feature_clusters.keys():
#         values = feature_clusters[len_key]
#         indices = index_clusters[len_key]

#         # shuffle the entire cluster
#         if shuffle:
#             indices, values = sklearn.utils.shuffle(indices, values)

#         indices = split(indices, batch_size)
#         data_splits = split(values, batch_size)

#         for batch in data_splits:
#             # get the first four features
#             # (conventionally set to :: form, lemma, tag, head, rel and take advantage of them being well formed)
#             features = map(lambda t: t[:5], batch)
#             padded_features = gen_pad_3d(features, padding_token)

#             # extract the character token ids
#             chars = map(lambda t: t[5], batch)
#             padded_chars = gen_pad_3d(chars, padding_token)

#             # this purely assumes the order of how vocabulary provides the data
#             # which in the format of :: Tuple[List(b,n), List(b,n), List(b,n), List(b,n), List(b,n), List(b,n,d)]
#             words, lemmas, tags, heads, rels = [
#                 padded_features[:, i, :] for i in range(5)
#             ]

#             # chars :: (batch_size, seq_len, longet_word)
#             # words+ :: (batch_size, seq_len)
#             _batch = Batch(x=[words, lemmas, tags, padded_chars], y=[heads, rels])
#             batches.append(_batch)

#         indicies.extend(indices)

#     if shuffle:
#         indices, batches = sklearn.utils.shuffle(indicies, batches)

#     return indicies, batches

def batch_by_buckets(samples, batch_size, shuffle):
    index_buckets = defaultdict(list)
    sample_buckets = defaultdict(list)

    # scan over dataset and add them to buckets of same length
    for sample_id, sample in enumerate(samples):
        words, *_ = sample
        n = len(words)

        index_buckets[n].append(sample_id)
        sample_buckets[n].append(sample)

    batches, batch_idx = [], []
    for sen_len in sample_buckets.keys():
        idxs = index_buckets[sen_len]
        samples = sample_buckets[sen_len]

        n_samples = len(idxs)
        n_splits = math.ceil(n_samples / batch_size)
        assert n_splits >= 1, "something went terribly wrong"

        adjusted_batch_size = math.ceil(n_samples / n_splits)

        batch_splits = split(samples, adjusted_batch_size)
        idx_splits = split(idxs, adjusted_batch_size)
        idx_splits = [np.array(e, dtype=np.int32) for e in idx_splits]

        for s in batch_splits:
            words, lemma, tags, gold_arcs, gold_rels, chars = zip(*s)

            x = list(zip(words, tags, gold_arcs, gold_rels))
            batch = np.array(x, dtype=np.int32)
            batches.append(((batch[:,0,:], batch[:,1,:],), (batch[:,2,:], batch[:,3,:])))

        batch_idx.extend(idx_splits)


    if shuffle:
        batch_idx, batches = sklearn.utils.shuffle(batch_idx, batches)

    return batch_idx, batches


def batch_by_buckets_with_chars(samples, batch_size, shuffle):
    index_buckets = defaultdict(list)
    sample_buckets = defaultdict(list)

    # scan over dataset and add them to buckets of same length
    for sample_id, sample in enumerate(samples):
        words, *_ = sample
        n = len(words)

        index_buckets[n].append(sample_id)
        sample_buckets[n].append(sample)

    batches, batch_idx = [], []
    for sen_len in sample_buckets.keys():
        idxs = index_buckets[sen_len]
        samples = sample_buckets[sen_len]

        n_samples = len(idxs)
        n_splits = math.ceil(n_samples / batch_size)
        assert n_splits >= 1, "something went terribly wrong"

        adjusted_batch_size = math.ceil(n_samples / n_splits)

        batch_splits = split(samples, adjusted_batch_size)
        idx_splits = split(idxs, adjusted_batch_size)
        idx_splits = [np.array(e, dtype=np.int32) for e in idx_splits]

        for s in batch_splits:
            words, lemma, tags, gold_arcs, gold_rels, chars = zip(*s)

            words = np.array(words, dtype=np.int32)
            tags = np.array(tags, dtype=np.int32)

            chars = gen_pad_3d(chars, padding_token=0)
            
            gold_arcs = np.array(gold_arcs, dtype=np.int32)
            gold_rels = np.array(gold_rels, dtype=np.int32)

            x = (words, tags, chars)
            y = (gold_arcs, gold_rels)

            batches.append((x,y))

        batch_idx.extend(idx_splits)


    if shuffle:
        batch_idx, batches = sklearn.utils.shuffle(batch_idx, batches)

    return batch_idx, batches

def scale_batch(samples, scale, cluster_count, padding_token, shuffle):
    kmeans = KMeans(cluster_count)

    # map sentences to their lengths
    lengths = np.array([len(s[0]) for s in samples]).reshape(-1, 1)

    labels = kmeans.fit_predict(lengths)

    clusters = defaultdict(lambda: [])
    clusters_max_length = defaultdict(int)
    for sample_id, (sample, cluster) in enumerate(zip(samples, labels)):
        clusters[cluster].append((sample_id, sample))
        clusters_max_length[cluster] = max(clusters_max_length[cluster], len(sample[0]))

    buckets = clusters
    bucket_max_lengths = clusters_max_length



    # create two lists, containing batched samples and corresponding origin indices
    batches, indices = [], []
    for cluster in buckets.keys():
        # retrieve a cluster and partition it into batches with padding
        samples = buckets[cluster]
        num_of_samples = len(samples)
        cluster_max_seq_len = bucket_max_lengths[cluster]

        # TODO. more flexible approach to features. currently hard coded for 4 features.
        HARDCODED_FEATURE_COUNT = 4

        # allocate data buffer to insert entire cluster into
        batch_buffer = np.zeros((num_of_samples, cluster_max_seq_len, HARDCODED_FEATURE_COUNT), dtype=np.int)
        batch_buffer[...] = padding_token  # assign padding

        # auxilery matrix to keep track of batched datas origin
        sample_indexer = np.zeros(num_of_samples, dtype=np.int)

        # insert each sample into buffer
        for i, (sample_id, sample) in enumerate(buckets[cluster]):
            words, lemma, tags, gold_arcs, gold_rels, chars = sample
            sample_n = len(words)
            batch_buffer[i, :sample_n, :] = np.array([words, tags, gold_arcs, gold_rels]).T
            sample_indexer[i] = sample_id

        # shuffle clusters
        if shuffle:
            batch_buffer, sample_indexer = sklearn.utils.shuffle(batch_buffer, sample_indexer)

        # this is the scaled difference
        # basically bases number of splits off tokens as opposed to sentences
        n_tokens = num_of_samples * cluster_max_seq_len

        # num_of_splits = num_of_samples // batch_size # normal batching
        num_of_splits = max(n_tokens // scale, 1)  # weighted batching

        # partition cluster by diving the size of the cluster with the desired scale
        batch_splits = np.array_split(batch_buffer, num_of_splits)
        index_splits = np.array_split(sample_indexer, num_of_splits)
        
        for bsplit in batch_splits:
            value_tuple = ((bsplit[:,:,0], bsplit[:,:,1]), (bsplit[:,:,2], bsplit[:,:,3]))
            batches.append(value_tuple)

        indices.extend(index_splits)


    if shuffle:
        indices, batches = sklearn.utils.shuffle(indices, batches)

    return indices, batches


# class BucketBatcher(object):
#     def __init__(self, samples, padding_token):
#         self._dataset = samples
#         self._padding_token = padding_token

#     def get_data(self, max_bucket_size, shuffle: bool = True):
#         data = self._dataset

#         len_clusters = defaultdict(list)
#         index_clusters = defaultdict(list)

#         # scan over dataset and add them to buckets of same length
#         for sample_id, sample in enumerate(data):
#             words, *_ = sample
#             n = len(words)

#             len_clusters[n].append(sample)
#             index_clusters[n].append(sample_id)

#         def batch_size_f(_): 
#             return max_bucket_size

#         batch_indicies, batches = build_from_clusters(
#             batch_size_f, index_clusters, len_clusters, self._padding_token, shuffle)

#         return batch_indicies, batches


# class ScaledBatcher(object):
#     def __init__(self, samples, cluster_count, padding_token):
#         self._dataset = samples
#         self._padding_token = padding_token

#         # map sentences to their lengths
#         lengths = np.array([len(s[0]) for s in samples]).reshape(-1, 1)
#         kmeans = KMeans(cluster_count)
#         labels = kmeans.fit_predict(lengths)

#         len_clusters = defaultdict(list)
#         index_clusters = defaultdict(list)

#         for sample_id, (sample, n) in enumerate(zip(samples, labels)):
#             len_clusters[n].append(sample)
#             index_clusters[n].append(sample_id)

#         self._clusters = len_clusters
#         self._indicies = index_clusters

#     def get_data(self, scale: int, shuffle=True):
#         def compute_batch_size(x):
#             cluster_size = len(x)
#             n = max(len(t[0]) for t in x)
#             n_tokens = cluster_size * n

#             num_of_splits = max(n_tokens // scale, 1)  # weighted batching
#             return max(cluster_size // num_of_splits, 1)

#         indicies, batches = build_from_clusters(
#             compute_batch_size, self._indicies, self._clusters, self._padding_token, shuffle)

#         return indicies, batches


class VanillaBatcher(object):
    def __init__(self):
        # do we want this ? Runtime becomes somewhat unreliable because large sentences scale.
        # to use this one hasto be very conservative about batch size_
        # I think we do... but lets w8
        pass

    def get_data(self, batch_size, shuffle=True):
        raise NotImplementedError()

class ScaledBatcher_V0(object):
    def __init__(self, samples, cluster_count, padding_token):
        self._dataset = samples
        self._padding_token = padding_token

        kmeans = KMeans(cluster_count)

        # map sentences to their lengths
        lengths = np.array([len(s) for s in samples]).reshape(-1, 1)

        labels = kmeans.fit_predict(lengths)

        clusters = defaultdict(lambda: [])
        clusters_max_length = defaultdict(int)
        for sample_id, (sample, cluster) in enumerate(zip(samples, labels)):
            clusters[cluster].append((sample_id, sample))
            clusters_max_length[cluster] = max(clusters_max_length[cluster], len(sample))

        self._clusters = clusters
        self._clusters_max_length = clusters_max_length

    def get_data(self, scale, shuffle=True):
        buckets = self._clusters
        bucket_max_lengths = self._clusters_max_length

        # create two lists, containing batched samples and corresponding origin indices
        batches, indices = [], []
        for cluster in buckets.keys():
            # retrieve a cluster and partition it into batches with padding
            samples = buckets[cluster]
            num_of_samples = len(samples)
            cluster_max_seq_len = bucket_max_lengths[cluster]

            # TODO. more flexible approach to features. currently hard coded for 4 features.
            HARDCODED_FEATURE_COUNT = 4

            # allocate data buffer to insert entire cluster into
            batch_buffer = np.zeros((num_of_samples, cluster_max_seq_len, HARDCODED_FEATURE_COUNT), dtype=np.int)
            batch_buffer[...] = self._padding_token  # assign padding

            # auxilery matrix to keep track of batched datas origin
            sample_indexer = np.zeros(num_of_samples, dtype=np.int)

            # insert each sample into buffer
            for i, (sample_id, sample) in enumerate(buckets[cluster]):
                sample_n = len(sample)
                batch_buffer[i, :sample_n, :] = sample
                sample_indexer[i] = sample_id

            # shuffle clusters
            if shuffle:
                batch_buffer, sample_indexer = sklearn.utils.shuffle(batch_buffer, sample_indexer)

            # this is the scaled difference
            # basically bases number of splits off tokens as opposed to sentences
            n_tokens = num_of_samples * cluster_max_seq_len

            # num_of_splits = num_of_samples // batch_size # normal batching
            num_of_splits = max(n_tokens // scale, 1)  # weighted batching

            # partition cluster by diving the size of the cluster with the desired scale
            batch_splits = np.array_split(batch_buffer, num_of_splits)
            index_splits = np.array_split(sample_indexer, num_of_splits)

            batches.extend(batch_splits)
            indices.extend(index_splits)

        if shuffle:
            indices, batches = sklearn.utils.shuffle(indices, batches)

        return indices, batches
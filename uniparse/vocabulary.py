from collections import Counter

import re
import pickle

import numpy as np


def validate_line(line):
    if line.startswith("#"):
        return False
    elif line == "\n":
        return True
    # think this is the one we want
    elif not re.match(r'\d+\t', line):
        return False
    else:
        return True


class Vocabulary(object):
    # Reserved token mappings
    PAD = 0
    ROOT = 1
    OOV = UNK = 2

    def __init__(self):
        self._id2word = None
        self._word2id = None
        self._lemma2id = None
        self._id2lemma = None
        self._tag2id = None
        self._id2tag = None
        self._rel2id = None
        self._id2rel = None
        self._id2freq = None
        self._id2char = None
        self._char2id = None
        self._words_in_train_data = None
        self._pret_file = None

    @staticmethod
    def _normalize_word(word):
        match = re.match("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+", word)
        return 'NUM' if match else word.lower()

    def fit(self, input_file, pretrained_embeddings=None, min_occur_count=0):
        word_counter, lemma_set, tag_set, rel_set, char_set = self._collect_tokens(input_file)

        self._id2word = ['<pad>', '<root>', '<unk>']
        self._id2lemma = ['<pad>', '<root>', '<unk>']
        self._id2tag = ['<pad>', '<root>', '<unk>']
        self._id2rel = ['<pad>', 'root']
        self._id2char = ['<pad>', 'root', '<unk>']
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)

        # add dataset tokens
        self._id2char += list(char_set)
        self._id2tag += list(tag_set)
        self._id2rel += list(rel_set)

        # map the reversed index
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._lemma2id = reverse(self._id2lemma)
        self._tag2id = reverse(self._id2tag)
        self._rel2id = reverse(self._id2rel)
        self._char2id = reverse(self._id2char)
        self._id2freq = {wid: word_counter[word] for word, wid in self._word2id.items()}
        self._words_in_train_data = len(self._id2word)

        self._pret_file = None
        if pretrained_embeddings:
            self._pret_file = pretrained_embeddings
            self._add_pret_words(pretrained_embeddings)

        return self

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def _collect_tokens(self, input_file):
        word_counter = Counter()
        lemma_set = set()
        tag_set = set()
        rel_set = set()
        char_set = set()
        with open(input_file, encoding="UTF-8") as f:
            for line in f.readlines():
                if not validate_line(line):
                    continue

                info = line.strip().split()
                if len(info) == 10:
                    word, lemma, tag, _, rel, chars = self._parse_conll_line(info, tokenize=False)
                    word_counter[word] += 1
                    lemma_set.add(lemma)
                    tag_set.add(tag)
                    char_set.update(chars)
                    if rel != 'root':
                        rel_set.add(rel)

        return word_counter, lemma_set, tag_set, rel_set, char_set

    def _add_pret_words(self, embedding_file):
        words_in_train_data = set(self._word2id.keys())
        offset = max(self._word2id.values()) + 1
        counter = 0

        with open(embedding_file, encoding="UTF-8") as f:
            for line in f.readlines():
                line = line.strip().split()
                if not line:
                    continue

                word = line[0]
                if word in words_in_train_data:
                    continue
                self._word2id[word] = offset + counter
                counter += 1

    def load_embedding(self, variance_normalize=False):
        """ load embeddings """

        assert self._pret_file is not None, "no embedding to load...."

        embs = [[]] * len(self._word2id.keys())
        vector = None
        with open(self._pret_file, encoding="UTF-8") as f:
            print(">> Loading embedding vectors")
            for i, line in enumerate(f.readlines(), start=1):
                line = line.strip().split()
                if not line:
                    continue

                word, vector = line[0], line[1:]
                word_id = self._word2id[word]

            print(">> Done loading embeddings (%d)" % i)

        emb_size = len(vector)
        for idx, emb in enumerate(embs):
            if not emb:
                embs[idx] = np.zeros(emb_size)
        pret_embs = np.array(embs, dtype=np.float32)

        if variance_normalize: 
            pret_embs /= np.std(pret_embs)

        return pret_embs

    def tokenize_conll(self, file: str):
        sentences = self._read_conll(file, tokenize=True)
        return sentences

    def _parse_conll_line(self, info, tokenize):
        word, lemma, tag, head, rel, chars = \
            info[1].lower(), info[2].lower(), info[3], int(info[6]), info[7], list(info[1])

        word = self._normalize_word(word)
        if tokenize:
            word, lemma, tag, head, rel = \
                self._word2id.get(word, self.OOV), self._lemma2id.get(lemma, self.OOV), \
                self._tag2id[tag], head, self._rel2id[rel]

            chars = [self.char2id(c) for c in chars]

        return word, lemma, tag, head, rel, chars

    def _read_conll(self, input_file: str, tokenize: bool = True):
        word_root = self.ROOT
        lemma_root = self.ROOT
        tag_root = self.ROOT
        rel_root = self.ROOT
        char_root = [self.ROOT]
        root_head = -1

        sents = []
        words, lemmas, tags, heads, rels, chars = \
            [word_root], [lemma_root], [tag_root], [root_head], [rel_root], [char_root]

        with open(input_file, encoding="UTF-8") as f:
            for line in f.readlines():
                if not validate_line(line):
                    continue

                info = line.strip().split("\t")
                if len(info) == 10:
                    word, lemma, tag, head, rel, characters = self._parse_conll_line(info, tokenize=tokenize)
                    words.append(word)
                    lemmas.append(lemma)
                    tags.append(tag)
                    heads.append(head)
                    rels.append(rel)
                    chars.append(characters)
                    # sent.append([word, tag, head, rel])
                    # word_chars.append([characters])
                else:
                    # sent_chars.append(word_chars)
                    sent = (words, lemmas, tags, heads, rels, chars)
                    sents.append(sent)

                    words, lemmas, tags, heads, rels, chars = \
                        [word_root], [lemma_root], [tag_root], [root_head], [rel_root], [char_root]
                    # sent = [[word_root, tag_root, root_head, rel_root]]
                    # word_chars = [char_root]

        return sents

    def word2id(self, x):
        return self._word2id.get(x, self.OOV)

    def id2word(self, x):
        return self._id2word[x]

    def rel2id(self, x):
        return self._rel2id[x]

    def id2rel(self, x):
        return self._id2rel[x]

    def tag2id(self, x):
        return self._tag2id.get(x, self.OOV)

    def char2id(self, x):
        return self._char2id.get(x, self.OOV)

    @property
    def wordid2freq(self):
        return self._id2freq

    @property
    def PUNCT(self):
        return self._tag2id["PUNCT"]

    @property
    def words_in_train(self):
        return self._words_in_train_data

    @property
    def vocab_size(self):
        return len(self._word2id)

    @property
    def upos_size(self):
        return len(self._tag2id)

    @property
    def tag_size(self):
        return len(self._tag2id)

    @property
    def char_size(self):
        return len(self._char2id)

    @property
    def label_count(self):
        return len(self._rel2id)

    @property
    def rel_size(self):
        return len(self._rel2id)

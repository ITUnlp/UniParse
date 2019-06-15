# UniParse

UniParse: A universal graph-based modular parsing framework, for quick prototyping and comparison of parser components.

UniParse is a collection of helpful tools and implementations to assist the process of rapidly developing efficient graph based dependency parsers. Backends for UniParse are currently dynet or pytorch.

This document describes how to install and use UniParse and how to develop your own new parser using the UniParse framework.

If you use UniParse, please cite the following publication, where you can also find further details:

**Varab, Daniel and Natalie Schluter. (2018).** [_UniParse: A universal graph-based parsing framework._](https://arxiv.org/pdf/1807.04053.pdf)

## Installing

Requirements [python](https://anaconda.org/anaconda/python)>=3.6 numpy, scipy, sklearn, and one or both of [dynet](http://dynet.readthedocs.io/en/latest/python.html)>=2.0.1 and [torch](https://pytorch.org/)>=0.4 (pytorch).

```bash
pip install numpy, scipy, sklearn

# uncomment desired (if any) backend
# pip install dynet>=2.0.1
# pip install torch>=0.4
```

## Compiling Decoders

```bash
# run from root directory
# decoders are accessible from uniparse.decoders.[eisner/cle]
python setup.py build_ext --inplace
```

Installation of [tensorflow](https://www.tensorflow.org/install/) for using tensorboard is optional.

## Running included models

UniParse includes two recent neural dependency parsers, namely the models by [Kiperwasser and Goldberg (2016)](https://arxiv.org/pdf/1603.04351.pdf) with either dynet or pytorch backend, and [Dozat and Manning (2017)](https://arxiv.org/pdf/1611.01734.pdf) with dynet backend. For running these we suggesting inspecting the run_* implementation and let

```bash
python run_kiperwasser.py \
--train [TRAIN].conllu \
--dev [DEV].conllu \
--test [TEST].conllu  \
--model [MODEL].model \
--decoder eisner \
(--dynet-devices GPU:0) \ # dynet specific
(--tensorboard [TENSORBOARD_OUTPUT])
```

Arguments surrounded by parentheses are optional. Note that many more arguments exist, and these are default-configured accordingly to the published configuration.

## Included models performance

Model                         | Language      | UAS wp. | LAS wp. | UAS np. | LAS np.
----------------------------- | ------------- | ------- | ------- | ------- | -------
Kiperwasser & Goldberg (2016) |               |         |         |         |
                              | Danish (UD)   | 83.18%  | 79.57%  | 83.67%  | 79.47%
                              | English (UD)  | 87.06%  | 84.68%  | 88.08%  | 85.43%
                              | English (PTB) | 92.56%  | 91.17%  | 93.14%  | 91.57%
Dozat & Manning (2017)        |               |         |         |         |
                              | Danish (UD)   | 87.42%  | 84.98%  | 87.84%  | 84.99%
                              | English (UD)  | 90.74   | 89.01%  | 91.47   | 89.38
                              | English (PTB) | 94.91%  | 93.70%  | 95.43%  | 94.06%
MST                           |               |         |         |         |
                              | Danish (UD)   | 67.17   | 55.52   | 68.80   | 55.30
                              | English (UD)  | 73.47   | 65.20   | 75.55   | 66.25
                              | English (PTB) | 74.00   | 63.60   | 76.07   | 64.67

with `wp.` denoting 'with punctuation', and `np.` 'no punctuation'. No punctuation follows the rule of excluding modifier tokens consisting entirely of unicode punctuation characters; this option is standard in current research.

## Components

Below we describe a brief introduction to each of the core components of the uniparse package, followed by a guide to how to stick them together.

All components are contained in /UniParse/uniparse/

### Vocabulary

The vocabulary encapsulates tokenisation of strings, and keeps track of mappings from `string -> int` as well as the inverse. This corresponds 1-to-1 to what other parser implementations call 'Alphabet'. The complete separation of this component is meant to ensure that preprocessing effects remain explicit and independently observable.

```python
vocab = Vocabulary()

# Fit the vocabulary to a corpus. Only UD formatted 'corpus' is currently supported. 
# You may optionally add a pretrained embedding which is then aligned properly.
vocab = vocab.fit(TRAINING_FILE, EMBEDDING_FILE)

# Tokenise data
training_data = vocab.tokenize_conll(TRAINING_FILE)
dev_data = vocab.tokenize_conll(DEV_FILE)
test_data = vocab.tokenize_conll(TEST_FILE)

# Get embedding as numpy array
embeddings = vocab.load_embedding(normalize=...)
```

./uniparse/vocabulary.py specifies the vocabulary.

### Dataprovider

Uniparse includes two means of batching the output of the vocabulary's tokenisation. `BucketBatcher` and `ScaledBatcher`.

`BucketBatcher` groups sentences of the same length into batches. The size of batches are bounded by a desired batch_size. This inherently causes some batches to be smaller than the desired batch_size, all the way down to of size 1\. However, the batch dimension is retained in these cases. Note that this batching strategy only employs padding on character sequences.

`ScaledBatcher` groups sentences into clusters with 1D k-nearest neighbours, and pads everything to the longest sample in each batch. The batch size is unorthodox in that it is a scaled product of the number of tokens. The essence of this strategy is to maintain (an approximate) constant token count in each batch. Even though some batches may contain many or few actual sentences, the contained token information remains the same at each learning step. A batch size is, given a cluster size `c`, and longest sentence length `l` .

![](https://latex.codecogs.com/svg.latex?\Large&space;nsplits=\frac{c}{(cl)/scale})<br>
![](https://latex.codecogs.com/svg.latex?\Large&space;batch\_size=\frac{c}{nsplits})

This batching strategy employs padding on both word level, as well as character sequences.

```python
bucket_provider = BucketBatcher(data, padding_token=[vocab.PAD | SOME_INTEGER])
scaled_provider = ScaledBatcher(data, cluster_count=..., padding_token=[vocab.PAD | SOME_INTEGER])

"""
indexes :: List
  list of original indicies to sort them in the end.
batches :: List[(x=Tuple, y=Tuple)]
  where len(x) == number of features [form, lemma, tag, characters]
  and len(y) == 2 for (target arcs and target relations)
"""
indexes, batches = dataprovider.get_data([batch_size | scale], shuffle=[True | False])
```

./uniparse/dataprovider.py specifies the batching strategy.

### Model

./uniparse/model.py together with a specialised model file placed in ./uniparse/models/ specifies the parser model. For example, the distributed parser implementations are located in ./uniparse/models/

### Evaluation Suite

Uniparse includes a unified script that covers utility and semantics of all previous commonly used evaluation implementations. The implementation wraps the perl script from conll2006/2007, as well as includes calls conll2017 universal dependency script. We reimplement the semantics of the no-punctuation metrics specified within the perl script. The script is located in the module `uniparse.evaluation.universal_eval`(.py) and is implemented with no dependencies and may be used directly.

Call it from the command line

```bash
python uniparse.evaluation.unversial_eval.py --p PREDICTION_UD_FILE --g GOLD_UD_FILE
```

or use it from within your code

```python
metrics = evaluate_files(PREDICTION_FILE, GOLD_FILE)
conll_metrics = conll17_eval(PREDICTION_FILE, GOLD_FILE)
perl_metrics = perl_eval(PREDICTION_FILE, GOLD_FILE)
```

## PTB split

Since the splitting of Penn treebank files is not fully standerdised we indicate the split used in experiments from [our paper](archivepaperlink), as well as supporting literature. Note that published model performances for systems we re-implement and distribute with UniParse may use different splits, which have a observerable impact on performance. Specifically, we note that [Dozat and Manning](https://arxiv.org/pdf/1611.01734.pdf)'s parser performs differently even using under splits than reported in their paper.

  Train   |  Dev   |  Test  | Discard
:-------: | :----: | :----: | :-----:
`{02-21}` | `{22}` | `{23}` | `{00}`

## Backlog

- Allow specifying loss from model
    - this allows for maximal flexibility without the need for fiddling with the internals of UniParse
- In-memory evaluataion
- Rename dataprovider module to 'data'
- template_runner/cli: Add support for word embeddings
- Move import of decoders in model.py to **init** in the decoders module
- Move call-backs to utility
- Port print statements to logging statements
- Add support for arbitrary optimizers
- Add clean dozat & manning impl. (in both dynet and pytorch)
- Custom score function
- Add continious tests via drone or the likes
    - python 2 compatibility
    - python 3 compatibility
- Implement binary crossentropy
    - ```python
    def CrossEntropy(yHat, y):
      if y == 1:
        return -log(yHat)
    else:
        return -log(1 - yHat)
    ```

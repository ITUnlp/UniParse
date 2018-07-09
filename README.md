# UniParse

UniParse: A universal graph-based parsing framework, for quick prototyping and comparison of components.  [ADD INTRODUCTORY STUFF--NATALIE] If you use UniParse, please cite

Varab, Daniel and Natalie Schluter. (2018). UniParse: A universal graph-based parsing framework. Complete this reference!!

[ADD INTRO STUFF--NATALIE]

## Installing

Requirements [python](https://anaconda.org/anaconda/python)>=3.6 numpy, scipy, sklearn, and one or both of [dynet](http://dynet.readthedocs.io/en/latest/python.html)>=2.0.1 and [torch](https://pytorch.org/)>=0.4 (pytorch). 
```
pip install numpy, scipy, sklearn

# uncomment desired (if any) backend
# pip install dynet>=2.0.1
# pip install torch>=0.4
```

[ADD INSTALL OVERVIEW STUFF HERE--NATALIE]

## Compiling Decoders
```
# run from root directory
# decoders are accessible from uniparse.decoders.[eisner/cle]
python setup.py build_ext --inplace
```
Installation of [tensorflow](https://www.tensorflow.org/install/) for using tensorboard is optional.

## Running included models
UniParse includes two recent neural dependency parsers, namely the models by [Kiperwasser & Goldberg](https://arxiv.org/pdf/1603.04351.pdf), and [Dozat and Manning](https://arxiv.org/pdf/1611.01734.pdf). For running these we suggesting inspecting the run_* implementation and let 

```
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
| Model                          |   Language    |   UAS wp.   |   LAS wp.   |   UAS np.   |   LAS np.  |
| ------------------------------ | ------------- | ----------- | ----------- | ----------- | -----------|
| Kiperwasser & Goldberg (2017)  |               |             |             |             |            |
|                                |  Danish (UD)  | 83.18%      | 79.57%      | 83.67%      | 79.47%     |
|                                |  English (UD) | 87.06%      | 84.68%      | 88.08%      | 85.43%     |
|                                | English (PTB) | 92.56%      | 91.17%      | 93.14%      | 91.57%     |
| Dozat & Manning (2017)         |    -          |             |             |             |            |
|                                |  Danish (UD)  | 87.42%      | 84.98%      | 87.84%      | 84.99%     |
|                                |  English (UD) | 90.74       | 89.01%      | 91.47       | 89.38      |
|                                | English (PTB) | 94.91%      | 93.70%      | 95.43%      | 94.06%     |
| MST (non-neural)               | -             |             |             |             |            |
|                                |  Danish (UD)  | 67.17       | 55.52       | 68.80       | 55.30      |
|                                |  English (UD) | 73.47       | 65.20       | 75.55       | 66.25      |
|                                | English (PTB) | 74.00       | 63.60       | 76.07       | 64.67      |

with `wp.` denoting 'with punctuation', and `np.` 'no punctuation'. No punctuation follows the rule of excluding modifier tokens consisting entirely of unicode punctuation characters; this option is standard in current research.

## Description
[MOVE THIS--NATALIE]
UniParse is a collection of helpful tools and implementations to assist the process of rapidly developing efficient graph based dependency parsers.

## Components
Below we describe a brief introduction to each of the core components of the uniparse package, followed by a guide to how to stick them together.

#### Vocabulary
The vocabulary encapsulates tokenization of strings, and keeps track of mappings from `string -> int` as well as the inverse. This corresponds 1-to-1 to what other parser implementations call 'Alphabet'. The complete separation of this component is meant to ensure that preprocessing effects remain explicit and independently observable.

```
vocab = Vocabulary()

# Fit the vocabulary to a corpus. Only UD formatted 'corpus' is currently supported. 
# You may optionally add a pretrained embedding which is then aligned properly.
vocab = vocab.fit(TRAINING_FILE, EMBEDDING_FILE)

# Tokenize data
training_data = vocab.tokenize_conll(TRAINING_FILE)
dev_data = vocab.tokenize_conll(DEV_FILE)
test_data = vocab.tokenize_conll(TEST_FILE)

# Get embedding as numpy array
embeddings = vocab.load_embedding(normalize=...)

```


#### Dataprovider
Uniparse includes two means of batching the ouput of the vocabularys tokenization. ``BucketBatcher`` and ``ScaledBatcher``.

``BucketBatcher`` groups sentences of the same length into batches. The size of batches are bounded by a desired batch_size.
This inherently causes som batches to be smaller than the desired batch_size, all the way down to of size 1. However, the batch dimension
is retained in these cases. Note that this batching strategy only employes padding on character sequences. 

``ScaledBatcher`` groups sentences into clusters with 1D k-nearest nabors, and pads everyting to the longest sample in each batch. 
The batch size is unorthodox in that it is a scaled product of the number of tokens. The essence of this strategy is to maintain
(an approximate) constant token count in each batch. Even though some batches may contain many or few actual sentences, 
the contained token information remains the same at each learning step.
A batch size is, given a cluster size ``c``, and longest sentence length ``l`` .

<img src="https://latex.codecogs.com/svg.latex?\Large&space;nsplits=\frac{c}{(cl)/scale}" />
<br>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;batch\_size=\frac{c}{nsplits}" />


This batching strategy employes padding on both word level, as well as character sequences.


```
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
#### Model

#### Evaluation Suite
Uniparse includes a unified script that covers utility and semantics of all previous commonly used evaluation implementations.
The implementation wraps the perl script from conll2006/2007, as well as includes calls conll2017 universal dependency script.
We reimplement the semantics of the no-puncation metrics specified of the perl script. The script is located in the module 
``uniparse.evaluation.universal_eval``(.py) and is implemented with no dependencies and may be used directly.

call it from the command line
````
python uniparse.evaluation.unversial_eval.py --p PREDICTION_UD_FILE --g GOLD_UD_FILE
````

or use it from within your code
````
    metrics = evaluate_files(PREDICTION_FILE, GOLD_FILE)
    conll_metrics = conll17_eval(PREDICTION_FILE, GOLD_FILE)
    perl_metrics = perl_eval(PREDICTION_FILE, GOLD_FILE)
````


## PTB split
Since the splitting of Penn treebank files is non-standerdized we denote a split, as well as supporting literature.
Note that published model performances use different splits, which we have observed to have a observerable impact on performance. Importantly we draw attention [Dozat and Manning](https://arxiv.org/pdf/1611.01734.pdf) performans differently under other splits that reported in the published work.


|   Train   |  Dev   |  Test  | Discard |
|:---------:|:------:|:------:|:-------:|
| `{02-21}` | `{22}` | `{23}` | `{00}`  | 


Run the following bash command to produce the corresponding conll formated files

````
cat 02.trees.conllu 03.trees.conllu 04.trees.conllu 05.trees.conllu 06.trees.conllu 07.trees.conllu 08.trees.conllu 09.trees.conllu 10.trees.conllu 11.trees.conllu 12.trees.conllu 13.trees.conllu 14.trees.conllu 15.trees.conllu 16.trees.conllu 17.trees.conllu 18.trees.conllu 19.trees.conllu 20.trees.conllu 21.trees.conllu > train.conllu
cat 22.trees.conllu > dev.conllu
cat 23.trees.conllu > test.conllu

````


[PLACE REFERENCES--NATALIE]
 - https://github.com/clulab/processors/wiki/Training-the-Neural-Network-Parser
 - https://arxiv.org/pdf/1602.07776.pdf
 - https://www-cs.stanford.edu/~danqi/papers/emnlp2014.pdf

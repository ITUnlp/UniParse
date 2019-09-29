# UniParse

UniParse: A universal graph-based modular parsing framework, for quick prototyping and comparison of parser components.  With this framework we provide a collection of helpful tools and implementations to assist the process of developing graph based dependency parsers. Neural Backends for UniParse are currently DyNet or PyTorch.


## Installation
Since UniParse contains cython code that depends on numpy, installation must be done in two steps.
```bash
# (1)
pip install -r requirements.txt

# (2)
pip install [-e] .  # include -e if you'd like to be able to modify framework code
```

## Blazing-fast decoders

```python
from uniparse.decoders import eisner, cle
# Both return a numpy array of integers representing the dependency tree.
```

| Algorithm         |     en_ud     |    en_ptb    |  sentences/s | % faster |
| ----------------- | ------------- | ------------ | ------------ | -------- |
| CLE    (Generic)  |     19.12     |     93.8     | ~ 404        |   -      |
| Eisner (Generic)  |     96.35     |     479.1    | ~ 80         |   -      |
| CLE    (UniParse) |     1.764     |     6.98     | ~ 5436       |   1345%  |
| Eisner (UniParse) |     1.49      |     6.31     | ~ 6009       |   7500%  |

## Evaluation
UniParse includes an evaluation script that works from within the framework, as well as by itself. For the former:

```python
from uniparse.evaluate import conll17_eval  # Wrapped UD evaluation script
from uniparse.evaluate import perl_eval  # Wrapped CONLL2006/2007 perl script. Ignores unicode punctuations (used for SOTA reports)
from uniparse.evaluate import evaluate_files  # UniParse rewritten evaluation. Provides scores with and without punctuation.

metrics1 = conll17_eval(test_file, gold_reference)
metrics2 = perl_eval(test_file, gold_reference)
metrics3 = evaluate_files(test_file, gold_reference)
# All return a dictionary :: {
#   uas: ...,
#   las: ...,
#   nopunct_uas: ...,
#   nopunct_las: ...
# }
```

... and for the latter, please copy the following path to a desired location `uniparse/evaluate/uniparse_evaluate.py` and use by running 
```
python uniparse_evaluate.py --test [FILENAME.CONLLU] --gold [GOLD_REFERENCE.CONLLU]
```


## Included models

### [Neural Models]
UniParse includes a small collection of state-of-the-art neural models that are implemented using a high level model wrapper that should reduce development time significantly. This component currently supports two neural backends, namely: DyNet and PyTorch. One of these libraries are required to use the model wrapper. The remaining components contained in UniParse have no dependencies.

```bash
# uncomment desired (if any) backend
# pip install dynet>=2.1
# pip install torch>=1.0
```


| Model                          |   Language    |   UAS w.p.   |   LAS w.p.   |   UAS n.p.   |   LAS n.p.  |
| ------------------------------ | ------------- | ------------ | ------------ | ------------ | ----------- |
| Kiperwasser & Goldberg (2016)  |               |              |              |              |             |
|                                |  Danish (UD)  | 83.18%       | 79.57%       | 83.67%       | 79.47%      |
|                                |  English (UD) | 87.06%       | 84.68%       | 88.08%       | 85.43%      |
|                                | English (PTB) | 92.56%       | 91.17%       | 93.14%       | 91.57%      |
| Dozat & Manning (2017)         |    -          |              |              |              |             |
|                                |  Danish (UD)  | 87.42%       | 84.98%       | 87.84%       | 84.99%      |
|                                |  English (UD) | 90.74        | 89.01%       | 91.47        | 89.38       |
|                                | English (PTB) | 94.91%       | 93.70%       | 95.43%       | 94.06%      |
| Nguyen and Verspoor (2018)     | -             |              |              |              |             |
|                                |  Danish (UD)  | TBA          | TBA          | TBA          | TBA         |
|                                |  English (UD) | TBA          | TBA          | TBA          | TBA         |
|                                | English (PTB) | TBA          | TBA          | TBA          | TBA         |
| MST (non-neural)               | -             |              |              |              |             |
|                                |  Danish (UD)  | 67.17        | 55.52        | 68.80        | 55.30       |
|                                |  English (UD) | 73.47        | 65.20        | 75.55        | 66.25       |
|                                | English (PTB) | 74.00        | 63.60        | 76.07        | 64.67       |

with `w.p.` and `n.p.` denoting 'with punctuation', 'no punctuation'. No punctuation follows the rule of excluding modifier tokens consisting entirely of unicode punctuation characters; this definition is standard in current research.



## PTB split
Since the splitting of Penn treebank files is not fully standerdised we indicate the split used in experiments from [our paper](archivepaperlink), as well as supporting literature.
Note that published model performances for systems we re-implement and distribute with UniParse may use different splits, which have a observerable impact on performance. Specifically, we note that [Dozat and Manning](https://arxiv.org/pdf/1611.01734.pdf)'s parser performs differently even using under splits than reported in their paper.

|   Train   |  Dev   |  Test  | Discard |
|:---------:|:------:|:------:|:-------:|
| `{02-21}` | `{22}` | `{23}` | `{00}`  | 


# Citation
If you use UniParse, please cite the following publication, where you can also find further details:

NoDaLida proceedings - TBA

# UniParse
UniParse: A universal graph-based modular parsing framework, for quick prototyping and comparison of parser components.  

**This code base has been moved to [github.com/danielvarab/uniparse](https://github.com/danielvarab/uniparse) as of October 1. 2019. All future development will happend in this repository.**

**To continue to use this version of UniParse, checkout the branch labeled `archived` which contains a frozen/archived state of UniParse.**

## Citation
If you are using UniParse, please cite our [paper](https://www.aclweb.org/anthology/W19-6149/).

```
@inproceedings{varab-schluter-2019-uniparse,
    title = "{U}ni{P}arse: A universal graph-based parsing toolkit",
    author = "Varab, Daniel  and Schluter, Natalie",
    booktitle = "Proceedings of the 22nd Nordic Conference on Computational Linguistics",
    month = "30 " # sep # " {--} 2 " # oct,
    year = "2019",
    address = "Turku, Finland",
    publisher = {Link{\"o}ping University Electronic Press},
    url = "https://www.aclweb.org/anthology/W19-6149",
    pages = "406--410",
    abstract = "This paper describes the design and use of the graph-based parsing framework and toolkit UniParse, released as an open-source python software package. UniParse as a framework novelly streamlines research prototyping, development and evaluation of graph-based dependency parsing architectures. UniParse does this by enabling highly efficient, sufficiently independent, easily readable, and easily extensible implementations for all dependency parser components. We distribute the toolkit with ready-made configurations as re-implementations of all current state-of-the-art first-order graph-based parsers, including even more efficient Cython implementations of both encoders and decoders, as well as the required specialised loss functions.",
}
```
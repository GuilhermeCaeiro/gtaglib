# GTAGLIB

This is the repository for the library GTAGLIB, that attempts to implement the algorithms of [1] and [2] to generate tags and tag clouds.

Please notice that this library is in a very, very early version and, as of the current version 0.1, it has issues.

## Installation

In order to install this library, your environment must run Python 3.7 of higher.

To install this library, from the root directory of this repository, first run the following line to install the requirements (right now, they might be higher than really needed):

```
pip install -r requirements.txt
```

And then run this next line to install the library itself.

```
python setup.py install
```

## Usage

In order to use this library, you have two option. One is to use it just to generate abstract, differential and set summary tags:

```
abstract, set_summary, differential = TagGenerator(
    semantic_field_size=40, 
    stemmer = "porter", 
    generate_bigrams=True,
    use_tfidf=True
).generate(dataset, method=1, root="rights")
```

And the other is to use it to generate tag clouds:

```
set_summary, differential = TagGenerator(
    semantic_field_size=40, 
    stemmer="porter", 
    generate_bigrams=True,
    use_tfidf=True
).generate_tag_cloud(
    dataset, 
    1,
    root="step", 
    outputdir="myoutputdir"
)
```

Further information about the supported parameters can be found [here](http://example.com/).

## References

[1] G. Xexeo, F. Morgado, and P. Fiuza, “Differential tag clouds: Highlighting particular features in documents,” in 2009 IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology, vol. 3, 2009, pp. 129–132.

[2] F. F. Morgado, “Representação de documentos através de nuvens de termos,” Master’s thesis, Federal University of Rio de Janeiro, Brazil, 2010. [Online]. Available: https://www.cos.ufrj.br/index.php/pt-BR/publicacoes-pesquisa/details/15/2172

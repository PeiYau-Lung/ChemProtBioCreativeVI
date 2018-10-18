# ChemProtBioCreativeVI
This repository contains the source code of the three-stage approach for the chemical-protein interaction extraction task in the BioCreative challenge VI. Details of the three-stage approach are described in: *Natural language processing based feature engineering for extracting chemical-protein interactions from literature*, (2018), Lung P-Y, He Z, Zhao T & Zhang J.

## Prerequisites
* Python 3.4+
* [Sklearn](http://scikit-learn.org/stable/install.html)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/build.html)
* [NLTK](https://www.nltk.org/install.html)
* [Stanford Neural Network Dependency Parser](https://stanfordnlp.github.io/CoreNLP/)

## Data 
Partial dataset used in the model are located in the data folder for demonstration purpose. It contains abstracts of PubMed articles, tagged chemical/protein entities and labeled relations released by the task organizers. The complete dataset, as well as the gold standard for testing set, can be found at [BiocreativeVI](http://www.biocreative.org/resources/corpora/chemprot-corpus-biocreative-vi/), or by contacting the organizers: [Martin Krallinger](krallinger.martin@gmail.com) & [Jesús Santamaría](jesus.sant@telefonica.net). 

## Usage
In the last line of `RunParser.py`, specify the path to the Stanford Neural Network Dependency Parser. Next, run the command 
```
$ sh demo.sh
```
This will run the pipeline, and generate `ChemProtTest_sumbit.tsv`, where each row contains: PubMedID, relation type, chemical entity, protein entity.  

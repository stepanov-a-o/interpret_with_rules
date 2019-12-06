## RIPPER-k and FURIA Rule Induction Algorithms for Interpretable Multiclass Classification

The code in this repository corresponds to a master thesis project. The thesis topic is 'RIPPER-k and FURIA Rule Induction Algorithms for Interpretable Multi-class Classification'. The thesis goal is to compare the interpretability of Fuzzy Unordered Rule Induction Algorithm (FURIA) and Repeated Incremental Pruning to Produce Error Reduction (RIPPER-k) rule induction algorithm on subsets of scikit-learn the 20 newsgroups text dataset.  

The most relevant and recent work in this field is the paper ‘Rule induction for global explanation of trained models’ by Sushil, Šuster and Daelemans (2018). The code in this repository builds upon that work; therefore, I want to acknowledge authors contribution and credit their work.

To read the authors paper, please refer to Sushil, M., Šuster, S., & Daelemans, W. (2018). Rule induction for global explanation of trained models. arXiv preprint arXiv:1808.09744. The link to the code for the paper is https://github.com/clips/interpret_with_rules

## Useful commands 

The command to induce rules from a pre-trained model is: 
```
python3 main.py -r gradient -loadmodel True -m your_model_name.tar
```
where your_model_name.tar is the name of the pre-trained model.

The command to induce rules from the original training data is
```
python3 main.py -r trainset -loadmodel False -m
```

The command to train a neural network on scikit-learn 20 newsgroups text dataset and then induce rules to interpret classifier's prediction is:  
```
python3 main.py -r gradient -loadmodel False -m your_model_name.tar
```

To get a complete list of all commands and options, please use: 
```
python3 main.py --help
```

To use the code with other datasets, please note that Weka needs the data to be present in ARFF or XRFF format to perform any classification tasks.

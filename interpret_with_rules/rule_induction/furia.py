import weka.core.jvm as jvm # jvm = Java Virtual Machine
# A Java virtual machine is a virtual machine that enables a computer to run 
# Java programs as well as programs written in other languages that are also 
# compiled to Java bytecode. The JVM is detailed by a specification that 
# formally describes what is required in a JVM implementation.

from weka.core.converters import Loader # The weka.core.converters module has a 
# convenience method for loading datasets called load_any_file. This method 
# determines a loader based on the file extension and then loads the full dataset.

from weka.classifiers import Classifier # classifier interface

from weka.classifiers import Evaluation # class for evaluating machine learning 
# models

from weka.core.classes import Random # (seed) Bases: weka.core.classes.JavaObject
# Wrapper for the java.util.Random class

from weka.filters import Filter # An abstract class for instance filters: objects 
# that take instances as input, carry out some transformation on the instance 
# and then output the instance
# More about Class Filter here http://weka.sourceforge.net/doc.dev/weka/filters/Filter.html

# from weka.classifiers import FilteredClassifier - you can use it to cross-validate
# filtered classifier and print evaluation and display ROC

from itertools import combinations # given an array of size n, generate and 
# print all possible combinations of r elements in array

from datetime import datetime
import numpy as np
import math
from tqdm import tqdm

# For more information and additional functions, such as cross-validation or 
# parameter optimization, please check:
# https://fracpete.github.io/python-weka-wrapper3/examples.html

# Some useful information about cross validation: 
# https://groups.google.com/forum/#!searchin/python-weka-wrapper/FURIA|sort:date/python-weka-wrapper/z0oKOm0trwo/g3XBoaJ2BQAJ

start_time = datetime.now()

def start_jvm():
    jvm.start()

def stop_jvm():
    jvm.stop()

def load_data(fname, dir_in = "../data/", incremental = False):
    """
    Loads data in weka format
    :param fname: filename for data
    :param dir_in: input directory
    :param incremental: True to read data incrementally.
    :return: The data and the loader object
    """
    loader = Loader(classname = "weka.core.converters.ArffLoader")
    if incremental:
        '''
        It generally means only loading into the warehouse the records that have 
        changed (inserts, updates etc.) since the last load; as opposed to doing 
        a full load of all the data (all records, including those that haven't 
        changed since the last load) into the warehouse.
        
        The advantage is that it reduces the amount of data being transferred 
        from system to system, as a full load may take hours / days to complete 
        depending on volume of data.
        
        The main disadvantage is around maintainability. With a full load, if 
        there's an error you can re-run the entire load without having to do 
        much else in the way of cleanup / preparation. With an incremental load, 
        the files generally need to be loaded in order. So if you have a problem 
        with one batch, others queue up behind it till you correct it. 
        Alternately you may find an error in a batch from a few days ago, and 
        need to re-load that batch once corrected, followed by every subsequent 
        batch in order to ensure that the data in the warehouse is consistent.
        Source: https://stackoverflow.com/questions/4471161/what-does-incremental-load-mean
        '''
        # Loads and prepares data from ARFF file
        data = loader.load_file(dir_in + fname, incremental=incremental)
        
    else:
        data = loader.load_file(dir_in + fname)
    data.class_is_last() # Required to specify which attribute is class attribute. 
    # For us, it is the last attribute.
    
    # You can also try to use class_is_first() and check the difference. 

    return data, loader

def merge_classes(data, idx_to_merge):
    """
    :param data: The data file to filter
    :param idx_to_merge: String representation of class indices to merge 
    :return: filtered data
    """
    merge_filter = Filter(classname = "weka.filters.unsupervised.attribute.MergeManyValues",
                          options=["-C", "last", "-R", idx_to_merge, "-unset-class-temporarily"])
    merge_filter.inputformat(data)
    filtered_data = merge_filter.filter(data)
    return filtered_data
    
def get_classifier(min_no, seed):
    cls = Classifier(classname = "weka.classifiers.rules.FURIA", 
		     options = ["-F", "6", "-N","2.0","-O","2","-S","1234","-p","0","-s","2"])
    '''    
    Return the classifier object given the options
    :param min_no: Minimum number of instances correctly covered by JRIP
    :param seed: Seed for randomizing instance order
    :return: classifier object
    
    Valid options are: 
    -F <number of folds> set number of folds for REP 
     One fold is used as pruning set (default 3)
     
    -N <min. weights> set the minimal weights of instances within a split (default 2.0)

    -O <number of runs> set the number of runs of optimizations(Default: 2)

    -D	set whether turn on the debug mode (Default: false)

    -S <seed> the seed of randomization (Default: 1)

    -E	whether NOT check the error rate>=0.5 in stopping criteria (default: check)

    -s	the action performed for uncovered instances (default: use stretching)

    -p the T-norm used as fuzzy AND-operator (default: Product T-norm)

    -output-debug-info if set, classifier is run in debug mode and may output
	 additional info to the console

    -do-not-check-capabilities if set, classifier capabilities are not checked 
    before classifier is built (use with caution).

    -num-decimal-places the number of decimal places for the output of numbers 
    in the model (default 2).

    -batch-size	the desired batch size for batch prediction  (default 100).
    ''' 
    return cls

def build_classifier(data, cls, incremental = False, loader = None):

    """
    Build classifier from the corresponding data
    :param data: weka data object
    :param cls: classifier object
    :param incremental: True if data is loaded incrementally
    :param loader: if incremental, the loader to load data
    :return: classifier
    """
    if incremental and loader is None:
        raise ValueError("Please enter a dataloader if incremental model")

    cls.build_classifier(data)

    if incremental:
        for inst in loader:
            cls.update_classifier(inst)

    return cls

def evaluate_classifier(cls, data, crossvalidate = False, n_folds = 10):
    """
    Evaluation
    :param cls: trained classifier
    :param data: data to test the model on
    :param crossvalidate: True to use crossvalidation
    :param n_folds: number of folds to cross validate for
    :return: evaluation object
    """
    evl = Evaluation(data)
    if crossvalidate:
        evl.crossvalidate_model(cls, data, n_folds, Random(5))
    else:
        evl.test_model(cls, data)

    return evl    

def optimize_rule_params(data, incremental, dataloader, class_index = None):
    """
    Iterate over different parameter values and train a rule induction model. 
    The best parameters are retained.
    :param data: Data to use for training and evaluating
    :param incremental: True if data is loaded incremetally
    :param dataloader: Data loader object if incremental is True
    :param class_index: Index of the class to compute F-score. None gives a 
    macro-averaged F-score.
    """
    
    stats = data.attribute_stats(data.class_index)
    min_inst = min(stats.nominal_counts)
    print("Number of instances in the minority class: {}".format(min_inst))

    print("Optimising over FURIA parameters")

    best_n, best_seed, best_model, best_eval, best_score = None, None, None, None, None

    # start_n = math.floor(0.01*min_inst)
    start_n = 2
    seeds = np.arange(0, 1, 1)

    for seed in seeds:
        # seed = int(seed)
        for n in tqdm(range(start_n, min_inst, 1)): 

            cls = get_classifier(n, seed)
            cls = build_classifier(data, cls, incremental, dataloader)

            evl = evaluate_classifier(cls, data, crossvalidate=False)

            if class_index is None:
                cur_score = evl.unweighted_macro_f_measure
            else:
                cur_score = evl.f_measure(class_index)

            if math.isnan(cur_score):
                break  # don't iterate to higher N value if current value covers zero instances for any class.

            # print("Unweighted macro f-measure for N {} and seed {}: {} \n".format(n, seed, cur_score))

            if best_eval is None or cur_score >= best_score:
                best_model = cls
                best_eval = evl
                best_n = n
                best_seed = seed
                best_score = cur_score

    print("Final results: ")
    print("Best performance found for N {} and seed {}".format(best_n, best_seed))
    print("Corresponding model: ", best_model)
    print("Corresponding results: ", best_eval.summary())
    
    if class_index is not None:
        print("Corresponding precision, recall, F-score: ",
              best_eval.precision(class_index),
              best_eval.recall(class_index),
              best_score)
    else:
        print("Unweighted Macro precision, recall and F-score:",
              (best_eval.precision(0) + best_eval.precision(1)) / 2,
              (best_eval.recall(0) + best_eval.recall(1)) / 2,
              best_score)
    print("Corresponding confusion matrix: ", best_eval.confusion_matrix)

def induce_furia_rules(data_file, data_dir='../data/', out_file = 'furia_rules.out', out_dir='../out/furia/'):
    """
    Induce the rules using FURIA
    :param data_file: File contaning training data in arff format
    :param data_dir: directory path for input file
    :param out_file: Filename to write the output model to
    :param out_dir: Directory to write the output file in
    """
    start_jvm()

    try:
        incremental = False
        data, dataloader = load_data(data_file, data_dir, incremental=incremental)

        n_classes = data.get_instance(0).num_classes
        print("Found {} classes".format(n_classes))

        if n_classes > 8: # input 2 for one vs rest setup for more than 2 classes
            class_list = [str(i) for i in range(1, n_classes+1, 1)]
            for to_merge in combinations(class_list, n_classes-1):
                print("Merging classes ", to_merge)

                new_data = merge_classes(data, ','.join(to_merge))
                optimize_rule_params(new_data, incremental, dataloader, 0) 
                # merged attribute is always the last one, so 0 index for desired class
        else:
            optimize_rule_params(data, incremental, dataloader) 
            # normal learning for binary cases

    except Exception as e:
        print(e)
    finally:
        stop_jvm()
        print(datetime.now() - start_time)


    # f_model = out_prefix+'-model.dat'
    # f_out = out_prefix+'-results.txt'
    #
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

if __name__ == '__main__':

    induce_furia_rules('newsgroups-reweighed-sa-test-pred.arff',
                       data_dir = '/Users/antonstepanov/Dropbox/Tilburg University/Master/Block 4/Thesis/Code/interpret_with_rules_master_updated/')
    
# .arff is the file format used by WEKA

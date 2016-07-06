# Marvin

This is a package implementing semi-supervised learning algorithms, which learn a binary classifier from independently drawn sets of labeled and unlabeled data. This package contains an implementation of the `Marvin` and `Hedge-Mower` algorithms in the paper: 

> Muffled Semi-Supervised Learning. Akshay Balsubramani and Yoav Freund. [Link to arXiv](http://arxiv.org/abs/1605.08833).

The "muffled" formulation implemented here gives a rigorous theoretical justification for using the unlabeled data differently from typical approaches. Here, the guidance provided by the labeled data is muffled on the unlabeled data by hallucinating the opposite labels to the majority prediction. 

This package is under constant revision and expansion, and so there will be implementation changes from the code used to generate the paper. The performance of the latest pulled version may be better (but not worse) than the paper's reported results. 


## System Requirements
- Python v2.7.11, NumPy v1.10.4, SciPy v0.17.0 (All are standard latest stable releases. Earlier versions may work, as Marvin only uses basic matrix and linear algebra functionality.)
- scikit-learn v0.18.dev0 (The development release, not stable. This is only needed to quickly explore the specialist partitioning structure of a decision tree, and so only used for the `Hedge-Mower` algorithm and variants, not the basic version of `Marvin`. A much slower (>15x) backward-compatible implementation in pure Python, which works with the stable release scikit-learn v0.17.1, is provided in the source code comments of `composite_feature.py`.)



## Example Usage
Working examples are provided in scripts, each generating a basic CSV log file (details in header source comments):

1. `slack_minimizer.py`: 
This contains code for a generic classifier (feature) aggregator in the muffled formulation. Included is a class to minimize the slack function, and a working example of such aggregation.

2. `ssb-mower.py`: 
This runs the `Hedge-Mower` family of algorithms for aggregating decision trees, including the two extremes referred to in the [paper](http://arxiv.org/abs/1605.08833) as `Hedge-Mower` and `Hedge-Mower-1`.

3. `marvin.py`: 
This runs the `Marvin` family of algorithms for incrementally learning an ensemble of classifiers and simultaneously how to best aggregate them -- a similar concept to supervised boosting. The file contains a class to run such algorithms with, and a working example of its usage.



## Implementation Notes:
`ssb-benchmarks.py` provides code for running the benchmarks we compare against, using `scikit-learn` to fit and evaluate ensemble and non-ensemble classification methods.


- To do: extend to other loss functions as in [this paper](http://arxiv.org/abs/1510.00452). 


## Further Information:
- For more on the "muffled" approach to semi-supervised learning, please refer to the following papers:

> Optimal Binary Classifier Aggregation for General Losses. Akshay Balsubramani and Yoav Freund. [Link to arXiv](http://arxiv.org/abs/1510.00452).
> Scalable Semi-Supervised Aggregation of Classifiers. Akshay Balsubramani and Yoav Freund. [Link to arXiv](http://arxiv.org/abs/1506.05790).
> Optimally Combining Classifiers Using Unlabeled Data. Akshay Balsubramani and Yoav Freund. [Link to arXiv](http://arxiv.org/abs/1503.01811).


## Contact:
Akshay Balsubramani (email listed in the [paper](http://arxiv.org/abs/1605.08833) and on [github](https://github.com/aikanor)).
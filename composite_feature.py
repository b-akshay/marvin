import numpy as np
import scipy as sp
import SSutils
import scipy.special
from sklearn.tree import _tree


class composite_feature (object):
    """
    A class that keeps track of all possible partitions of a tree - a set of specialists. 
    All rho values are uniform over all data being predicted upon.
    The weights are n*rho \geq 1 , except when they are 0 for hypotheses which always sleep. 
    The features kept are either all intermediate tree nodes (mode='allrel'), or just those with the top k scores for some parameter k (mode='topk'). 
    k=0 corresponds to just wrapping the tree predictions in a featurized form. 
 
    self.b : b-vector for the specialists
    self.featurize(dataset) : returns feature matrix S for the dataset
    self.weights : vector showing n*rho for each specialist
    self.pred_signs: vector indicating if the specialist in question should predict its labeling or the inversion.

    How relevant_ndces work: they are set by a combination of k >= 0 and mode = {'allrel', 'topk'}. 
    We either consider all intermediate specialist nodes ('allrel') or just the top few ('topk'), in which case the k parameter is used.
    is_sklearn_rf == true iff features are in {0,1} instead of {-1,1}.
    """

    def __init__(self, base_classifier, holdout_set, holdout_labels, is_tree=False, failure_prob=0.01, k=0, numsamp_min=0, is_sklearn_rf=False):
        self.base_classifier = base_classifier
        self.is_tree = is_tree
        if self.is_tree:
            # The following instance variables are only used in the backward-compatible 
            self.tr = base_classifier.tree_
            self.numnodes = self.tr.capacity
        else:
            self.numnodes = 3     # for the classifier, and its +/- predictions (as specialists that only predict one class)
        self.is_sklearn_rf = is_sklearn_rf
        self.reinitialize (holdout_set, holdout_labels, failure_prob=failure_prob, k=k, numsamp_min=numsamp_min)


    """
    This function adjusts weights, preds, b, and lists of relevant indices; 
    so it is only called when we want to modify the correspondence of the underlying tree partition. 
    Does not involve any expansion of unlabeled data in memory. 
    """
    def reinitialize (self, holdout_set, holdout_labels, failure_prob=0.01, k=0, numsamp_min=0):
        mat = self.specialists_onoff (holdout_set)
        numsamps = np.array(mat.sum(axis=0)).flatten()

        spec_weights = 1.0/numsamps
        spec_weights[np.isinf(spec_weights)] = 0.0
        self.weights = np.max(numsamps)*spec_weights

        # Note the label transformation here; currently base predictions in {0,1}
        preds = self.base_predict (holdout_set)
        transmat = mat.multiply(sp.sparse.lil_matrix(preds).transpose())
        normfeats = transmat.multiply(sp.sparse.lil_matrix(spec_weights))
        plugin_est = normfeats.transpose().dot(holdout_labels)

        self.pred_signs = np.ones(len(plugin_est))
        self.pred_signs[plugin_est < 0.0] = -1.0

        # useful for debugging purposes
        self.plugin_est = np.multiply(plugin_est, self.pred_signs)
        self.numsamps = numsamps

        self.b = SSutils.calc_b_wilson(self.plugin_est, self.numsamps, failure_prob=failure_prob)
        #print self.b, self.plugin_est, self.numsamps
        
        self.numsamp_min = numsamp_min
        self.update_relevant_ndces(k)


    """
    Returns (F.T, b) corresponding to a subset of good features.
    Dimension of returned matrix: (# examples) x (# relevant nodes)
    """
    def featurize (self, dataset, mode='allrel'):
        if mode == 'allrel':
            relevant_ndces = self.relevant_ndces
        elif mode == 'topk':
            relevant_ndces = self.topk_ndces
        mat = self.specialists_onoff (dataset)
        # Now just pick out the right features.
        # print relevant_ndces
        mat = mat[:, relevant_ndces]
        numsamps = np.array(mat.sum(axis=0)).flatten()
        spec_weights = 1.0/numsamps
        spec_weights[np.isinf(spec_weights)] = 0.0
        weights = np.max(numsamps)*spec_weights

        preds = self.base_predict (dataset)
        transmat = mat.multiply(sp.sparse.lil_matrix(preds).transpose())

        # print weights.shape, relevant_ndces.shape, transmat.shape
        weights[np.isinf(weights)] = 0.0
        weighted_preds = np.multiply(weights, self.pred_signs[relevant_ndces])
        return (transmat.multiply(sp.sparse.lil_matrix(weighted_preds)) , self.b[relevant_ndces])


    """
    "datamat" is a (# examples) x (# features) matrix, one row per example. This assumes it fits in memory.
    Returns a 0-1 matrix, (# examples) x (# nodes), indicating which examples fell in which intermediate nodes (the decision path). 
    Currently, this is the only place in which any tree API is actually used. 
    ONLY COMPATIBLE WITH SCIKIT-LEARN VERSION 0.18.dev0.0. See function body. 

    is_tree is an alternate implementation of specialists_onoff(), which does not use any tree API. 
    It splits classifiers into 3 sub-features: original classifier and +/- uniclass specialist predictors.
    Works with any classifier following the sklearn API (fit(), predict())
    """
    def specialists_onoff (self, datamat):
        if self.is_tree:
            return self.base_classifier.decision_path(datamat)
        else:
            numpts = datamat.shape[0]
            toret = np.zeros((numpts, 3))
            toret[:, 0] = 1.0
            preds = self.base_predict (datamat)
            toret[np.where(preds == -1)[0], 1] = 1.0
            toret[np.where(preds == 1)[0], 2] = 1.0
            return sp.sparse.csr_matrix(toret)
        """
        The above implementation IS INCOMPATIBLE WITH THE CURRENT OFFICIAL RELEASE of scikit-learn (0.17.1)! 
        The following pure python code is backward-compatible (~30x slower), for the case is_tree == True:

        feature = self.tr.feature
        threshold = self.tr.threshold
        children_left = self.tr.children_left
        children_right = self.tr.children_right
        numpts = datamat.shape[0]
        mtx = np.zeros((numpts, self.numnodes))
        for i in range(numpts):
            datapt = datamat[i]
            new_node = 0 # because root has node_id 0
            ndx_in_list = 0
            while new_node != _tree.TREE_LEAF:
                mtx[i, new_node] = 1.0
                ndx_in_list += 1
                if datapt[feature[new_node]] <= threshold[new_node]:
                    new_node = children_left[new_node]
                else:
                    new_node = children_right[new_node]
        return mtx
        """


    """
    Procedure to update the relevant indices. 
    Relevant_ndces and topk_ndces should always contain the root.
    """
    def update_relevant_ndces (self, k):
        self.weighty_ndces = np.where(self.numsamps >= self.numsamp_min)[0]
        relevant_ndces = set(np.nonzero(self.b)[0]) & set(self.weighty_ndces)        # intersection of these two sets.
        relevant_ndces.add(0)
        self.relevant_ndces = list(relevant_ndces)
        if k < len(self.b):
            topk_ndces = set(np.argpartition(-self.b, k)[:k])        # select the indices of the k features with highest b-value. Only does a partial sort!
        else:
            topk_ndces = set(range(len(self.b)))
        topk_ndces.add(0)   # adding the root index if it doesn't exist already. This means there will generally be k+1 > k features.
        self.topk_ndces = list(topk_ndces & relevant_ndces)


    def base_predict (self, datamat):
        return self.y_trans(self.base_classifier.predict (datamat))


    def y_trans (self, y):
        if self.is_sklearn_rf:
            return 2.0*y - 1
        else:
            return y
import numpy as np
import scipy as sp
from sklearn.tree import _tree

class CompositeFeature(object):
    """
    A class that keeps track of a set of specialists derived from a base classifier. 
    Typically, the base classifier is a decision tree, in which case its nodes 
    (including root, leaves, and internal nodes) constitute the specialists. 

    Regardless of the base classifier, there is always at least one derived specialist: 
    the base classifier itself, taken as a 'generalist' that predicts on all data given. 
    If there are no other derived specialists, the class is just a wrapper 
    around the base classifier's predictions. 
    For details, see the derivation in Sec. 3 of the paper::
    "Scalable semi-supervised aggregation of classifiers," Balsubramani and Freund, NIPS 2015.

    Attributes:
        base_classifier: The base classifier, following the scikit-learn API.
        use_tree_partition: Boolean, true iff base_classifier is a decision tree 
            and the derived specialists are (a subset of) its nodes. 
        label_corrs: A vector of lower bounds on label correlations of the base classifier 
            and its derived specialists. This is the vector denoted 'b' in the derivation
            of the muffled framework. This vector >= 0, because derived specialists 
            with negative label correlations automatically have their predictions inverted.
        spec_weights: A vector of floats specifying each specialist's weight, which is 
            1/(fraction of data on which the specialist predicts), i.e. >= 1. 
            If the specialist does not predict on any data, its weight is set to 0. 
    """

    def __init__(self, base_classifier, init_data, init_labels, 
        use_tree_partition=False, failure_prob=0.01, from_sklearn_rf=False):
        self.base_classifier = base_classifier
        self.use_tree_partition = use_tree_partition
        # The following Boolean is true iff labels are {0,1}, not {-1,1}. See _label_trans(...) docs.
        self._from_sklearn_rf = from_sklearn_rf
        self._init_from_data(init_data, init_labels, failure_prob)

    def _init_from_data(self, init_data, init_labels, failure_prob, sample_counts=None):
        """
        Updates several quantities associated with each specialist, using an input labeled dataset. 
        These include specialist weights self.spec_weights, label correlation bounds self.label_corrs, 
        and the list of relevant specialist indices self._relevant_ndces. 

        Args:
            init_data: Matrix of input data, one row per example. 
                Dimensions: {# examples} rows, {# features} columns. 
            init_labels: Vector of labels of init_data, one label per example. 
            failure_prob: Float. The allowed failure (tail) probability used 
                to define the Wilson confidence interval for each classifier.
            k: Optional int, approximately specifying the number of derived specialists to select. 
                Precisely, the best k (by Wilson error bound) are selected, along 
                with the base classifier if it is not already one of the best k.
            sample_counts: Optional vector of ints, specifying the number of samples 
                on which each specialist predicts. Defaults to being calculated from init_data.

        Does not call self.featurize(...) because that function only computes on relevant specialists, 
        while this one must compute on all specialists to figure out which are relevant. 
        """
        abst_indicators = self._specialists_onoff(init_data)
        if sample_counts is None:
            self._numsamps = np.array(abst_indicators.sum(axis=0)).flatten()
        else:
            self._numsamps = sample_counts
        self.spec_weights = self.calc_specialist_weights(self._numsamps)
        base_preds = self.base_predict(init_data)
        preds_abst = abst_indicators.multiply(sp.sparse.lil_matrix(base_preds).transpose())
        normfeats = preds_abst.multiply(sp.sparse.lil_matrix(self.spec_weights))
        labelcorrs_plugin = (1.0/np.max(self._numsamps))*normfeats.transpose().dot(init_labels)
        # Build vector self._preds_flip of {-1,1}s indicating if each specialist is correlated (1) 
        # or anti-correlated (-1) with the true labels; allows inversion of the latter into the former.
        self._preds_flip = np.sign(labelcorrs_plugin)
        labelcorrs_plugin = np.abs(labelcorrs_plugin)
        self.label_corrs = self.calc_b_wilson(
            labelcorrs_plugin, self._numsamps, failure_prob=failure_prob)
        #self._numsamp_min = numsamp_min
        self._update_relevant_ndces()

    def featurize(self, dataset, k=-1):
        """
        Converts a matrix of raw data examples into a matrix of the predictions on the data
        of the base classifier and a selected subset of its derived specialists (the "best" ones).

        Args:
            dataset: Matrix of data on which to predict. One row per example; {# examples} rows, {# features} cols. 
            k: Optional int, approximately specifying the number of derived specialists to select. 
                Precisely, the best k (by Wilson error bound) are taken, along with the 
                base classifier if it is not already one of the best k. 
                So k==0 means this is just a wrapper around the base classifier's predictions. 
                Defaults to -1, which selects all possible derived specialists with 
                Wilson confidence intervals not containing error 1/2, 
                as measured by self._relevant_ndces. Any other negative value results in an error. 

        Returns:
            A pair (Ft, b). Ft is a matrix of the selected classifiers' predictions. 
            It has {# examples} rows and {# selected classifiers} columns.
            b is a vector of the selected classifiers' label correlations 
            (Wilson lower bounds), over the data on which they predict. Equivalently, 
            it is self.label_corrs over just the selected classifiers.
        """
        if k == -1:
            relevant_ndces = self._relevant_ndces
        elif k >= 0:
            relevant_ndces = self._calculate_topk_ndces(k)
        else:
            relevant_ndces = None    # Results in an error, by intention.
        abst_indicators = self._specialists_onoff(dataset)
        abst_indicators = abst_indicators[:, relevant_ndces]
        # numsamps = np.array(abst_indicators.sum(axis=0)).flatten()
        # spec_weights = self.calc_specialist_weights(numsamps)    # Use instead of self.spec_weights
        base_preds = self.base_predict(dataset)
        # The following large matrix multiplication only involves relevant specialist indices, 
        # and can be much faster than computing all indices when only very few are relevant.
        preds_abst = abst_indicators.multiply(sp.sparse.lil_matrix(base_preds).transpose())
        signed_spec_wts = np.multiply(
            self.spec_weights[relevant_ndces], self._preds_flip[relevant_ndces])
        return (preds_abst.multiply(
            sp.sparse.lil_matrix(signed_spec_wts)), self.label_corrs[relevant_ndces])

    def calc_specialist_weights(self, numsamps):
        """
        Calculates vector of specialist weights from a vector of 
        the number of samples on which each predicts. Helper function.

        Args:
            numsamps: A nonnegative vector of ints, specifying the number of samples 
                on which each specialist predicts.

        Returns:
            A vector of floats specifying each specialist's weight, which is 
            If numsamps[i] == 0 for some specialist i, the corresponding weight will be 0. 

        Note that the return value is invariant to the scaling of numsamps, 
        i.e. multiplying it by a positive constant. 
        Similarly, calculating numsamps using a uniform random subsample of a dataset 
        will result in approximately the same return value as using the full dataset.
        """
        weights = 1.0/numsamps
        weights[np.isinf(weights)] = 0.0
        return np.max(numsamps)*weights

    def base_predict(self, datamat):
        """
        Predictions of the base classifier on an input dataset.

        Args:
            datamat: Matrix of data on which to predict. Each row contains one example. 

        Returns:
            Vector of label predictions of the base classifier on each example.
        """
        return self._label_trans(self.base_classifier.predict(datamat))

    def calc_b_wilson(self, labelcorrs_plugin, numsamps, failure_prob=0.01):
        """
        Calculate Wilson interval lower bound on label correlation for each classifier.

        Args:
            labelcorrs_plugin: Array containing estimated label correlation for each classifier. 
                Assumed to be >= 0. 
            numsamps: Array containing the number of (labeled) samples used 
                to estimate the corresponding element of labelcorrs_plugin. 
            failure_prob: Optional float specifying the allowed failure (tail) probability used 
                to define the Wilson confidence interval for each classifier.
                Defaults to 0.01, a fairly aggressive value in practice.

        Returns: 
            Array containing the Wilson interval lower bound on label correlation for each classifier. 
            An entry of the array is zero iff either of the following conditions 
            are met by the corresponding classifier: 
            (a) It always abstains (i.e. numsamps[classifier index] == 0). 
            (b) Its error is estimated on too few samples to sufficiently narrow the error bar, 
                so that the Wilson lower bound is <= 0. In other words, the interval contains 0.
        """
        err_est = 0.5*(1.0 - labelcorrs_plugin)
        z = sp.stats.norm.ppf(0.5*(2-failure_prob))
        zsq_n = z*z/numsamps
        sleeping_specs = np.isinf(zsq_n)
        recentered_mean_errors = np.multiply(1.0/(1+zsq_n), (err_est + 0.5*zsq_n))
        recentered_plugins = 1.0 - 2*recentered_mean_errors
        stddevs = np.sqrt(np.multiply(zsq_n, np.multiply(
            err_est, 1.0 - err_est)) + np.square(0.5*zsq_n))
        stddevs = np.multiply(1.0/(1+zsq_n), stddevs)
        toret = np.maximum(recentered_plugins - 2*stddevs, 0.0)
        toret[sleeping_specs] = 0.0
        return toret

    def _specialists_onoff(self, dataset):
        """
        Indicates which examples of the given input dataset are abstained upon by 
        the base classifier and each of its derived specialists.

        Args:
            dataset: Matrix of data on which specialists predict/abstain. 
                One row per example, typically a numpy matrix. 
                Dimensions: {# examples} rows, {# features} columns. 

        Returns: 
            A binary sparse matrix (values in {0,1}) in scipy CSR format. 
            Dimensions: {# examples} rows, {# derived specialists} columns. 
            An entry '1' indicates that a specialist predicts an example's label, and '0' that it abstains. 
            If use_tree_partition == True, derived specialists correspond to nodes of the base decision tree. 
            Otherwise, there are 3 derived specialists: the base classifier, 
            and two one-class specialists corresponding to its '-1' and '1' predictions respectively.

        .. note::
        This is the only place in which the sklearn.tree API is actually used. 
        Currently ONLY COMPATIBLE WITH SCIKIT-LEARN VERSION 0.18.dev0.0. 
        """
        if self.use_tree_partition:
            return self.base_classifier.decision_path(dataset)
            # When use_tree_partition is true and a version of sklearn <= 0.17 is used, 
            # the above line will not work. Here is a backward-compatible implementation: 
            # return self._specialists_onoff_backwardcompatible_tree(dataset)
        else:
            numpts = dataset.shape[0]
            toret = np.zeros((numpts, 3))
            toret[:, 0] = 1.0
            base_preds = self.base_predict (dataset)
            toret[np.where(base_preds == -1)[0], 1] = 1.0
            toret[np.where(base_preds == 1)[0], 2] = 1.0
            return sp.sparse.csr_matrix(toret)

    def _update_relevant_ndces(self):
        """
        Updates the list of relevant indices (self._relevant_ndces). This contains:
        (1) The base classifier in all cases (so it always has length >= 1). 
        (2) Any derived specialist with nonzero Wilson bound (see self.calc_b_wilson(...) 
            docs for details on when this happens).
        """
        weighty_ndces = range(len(self._numsamps))
        relevant_ndces = set(np.nonzero(self.label_corrs)[0]) & set(weighty_ndces)
        relevant_ndces.add(0)
        self._relevant_ndces = list(relevant_ndces)

    def _calculate_topk_ndces(self, k):
        """
        Calculate the indices of the k specialists with highest b-value, 
        including the base classifier regardless of its b-value. 

        Args:
            k: int >= 0, approximately specifying the number of derived specialists to select. 
                Precisely, the best k (by Wilson error bound) are taken, along with the 
                base classifier if it is not already one of the best k. 

        Returns:
            A list containing the indices of the top k classifiers. 
            The list always at least contains the base classifier's index (i.e. 0).
            Therefore, the list is of length k if the base classifier is one of the top k, 
            and length k+1 otherwise. If k is greater than the total number of derived 
            specialists, returns all of them.
        """
        if k < len(self.label_corrs):
            topk_ndces = set(np.argpartition(-self.label_corrs, k)[:k])  #Only does a partial sort of b!
        else:
            topk_ndces = set(range(len(self.label_corrs)))
        topk_ndces.add(0)
        return list(topk_ndces & set(self._relevant_ndces))

    def _label_trans(self, labels):
        """
        A workaround, necessary because sklearn random forests predict in {0,1} 
        even when trained on labels in {-1,1} (strange but true!).
        """
        if self._from_sklearn_rf:
            return 2.0*labels - 1
        else:
            return labels

    def _specialists_onoff_backwardcompatible_tree(self, datamat):
        """
        <DEPRECATED>
        When use_tree_partition == True, _specialists_onoff IS INCOMPATIBLE 
        with the current official release of scikit-learn (0.17.1)! 
        This function provides a compatible implementation for completeness; 
        but it is ~30x slower (does not follow C pointers directly) and not recommended!
        """
        feature = self.base_classifier.tree_.feature
        threshold = self.base_classifier.tree_.threshold
        children_left = self.base_classifier.tree_.children_left
        children_right = self.base_classifier.tree_.children_right
        numpts = datamat.shape[0]
        mtx = np.zeros((numpts, self.base_classifier.tree_.capacity))
        for i in range(numpts):
            datapt = datamat[i]
            new_node = 0  #Because root has node_id 0.
            ndx_in_list = 0
            while new_node != _tree.TREE_LEAF:
                mtx[i, new_node] = 1.0
                ndx_in_list += 1
                if datapt[feature[new_node]] <= threshold[new_node]:
                    new_node = children_left[new_node]
                else:
                    new_node = children_right[new_node]
        return mtx
import numpy as np
import scipy as sp
import muffled_utils
from sklearn.tree import _tree


class CompositeFeature(object):
    """
    A class that keeps track of a set of specialists derived from a base classifier. 
    Often, the base classifier is a decision tree, in which case its nodes 
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

    def __init__(self, base_classifier, init_data, init_labels=None, sample_counts=None, 
        use_tree_partition=False, failure_prob=0.01, from_sklearn_rf=False):
        self.base_classifier = base_classifier
        self.use_tree_partition = use_tree_partition
        # The following Boolean is true iff labels are {0,1}, not {-1,1}. See self._label_trans(...) docs.
        self._from_sklearn_rf = from_sklearn_rf
        self._relevant_ndces = [0]   # By default, the base classifier is always selected.
        self.init_from_data(init_data, init_labels, failure_prob, sample_counts)

    def init_from_data(self, init_data, init_labels, failure_prob, sample_counts):
        """
        Updates several quantities associated with each specialist, using an input labeled dataset. 
        These include specialist weights self.spec_weights, label correlation bounds self.label_corrs, 
        and lists of relevant specialist indices. 

        Args:
            init_data: Matrix of input data, one row per example. Dimensions: {# examples} rows, {# features} columns. 
            init_labels: Vector of labels of init_data, one label per example. None if no labels given.
            failure_prob: Float; allowed failure (tail) probability used to define the Wilson confidence interval for each classifier.
            sample_counts: Vector of ints, specifying the number of samples on which each specialist predicts. 
        """
        # Does not call self.predictions(...) because that function only computes on relevant specialists, 
        # while this one must compute on all specialists to figure out which are relevant. 
        abst_indicators = self._specialists_onoff(init_data)
        if sample_counts is None:
            self._numsamps = np.array(abst_indicators.sum(axis=0)).flatten()
        else:
            self._numsamps = sample_counts
        self.spec_weights = muffled_utils.calc_specialist_weights(self._numsamps)
        self.label_corrs = None
        self._preds_flip = np.ones(len(self.spec_weights))
        if init_labels is not None:
            base_preds = self.base_predict(init_data)
            preds_abst = abst_indicators.multiply(sp.sparse.lil_matrix(base_preds).transpose())
            normfeats = preds_abst.multiply(sp.sparse.lil_matrix(self.spec_weights))
            labelcorrs_plugin = (1.0/np.max(self._numsamps))*normfeats.transpose().dot(init_labels)
            # Build vector self._preds_flip of {-1,1}s indicating if each specialist is correlated (1) 
            # or anti-correlated (-1) with the true labels; allows inversion of the latter into the former.
            self._preds_flip = np.sign(labelcorrs_plugin)
            labelcorrs_plugin = np.abs(labelcorrs_plugin)
            self.label_corrs = muffled_utils.calc_b_bound(
                labelcorrs_plugin, self._numsamps, failure_prob=failure_prob)
            self._update_relevant_ndces()

    def predictions(self, dataset, k=-1):
        """
        Converts a matrix of raw data examples into a matrix of the predictions on the data
        of the base classifier and a selected subset of its derived specialists (the "best" ones).

        Args:
            dataset: Matrix of data on which to predict. One row per example; {# examples} rows, {# features} cols. 
            k: Optional int, approximately specifying the number of derived specialists to select. 
                Precisely, the best k (by Wilson error bound) are taken, along with the 
                base classifier if it is not already one of the best k. So k==0 means this is just a 
                wrapper around the base classifier's predictions. Defaults to -1, which selects 
                all derived specialists with Wilson intervals not containing error 1/2. 

        Returns:
            A matrix of the selected classifiers' predictions, with {# examples} rows 
            and {# selected classifiers} columns.
        """
        if k == -1:
            relevant_ndces = self._relevant_ndces
        elif k >= 0:
            relevant_ndces = self._calculate_topk_ndces(k)
        abst_indicators = (self._specialists_onoff(dataset))[:, relevant_ndces]
        base_preds = self.base_predict(dataset)
        # The following large matrix multiplication only involves relevant specialist indices, 
        # and can be much faster than computing all indices when only very few are relevant.
        preds_abst = abst_indicators.multiply(sp.sparse.lil_matrix(base_preds).transpose())
        signed_spec_wts = np.multiply(
            self.spec_weights[relevant_ndces], self._preds_flip[relevant_ndces])
        return preds_abst.multiply(sp.sparse.lil_matrix(signed_spec_wts))

    def relevant_label_corrs(self, k=-1):
        """ 
        Label correlations of the base classifier and a selected subset of its derived specialists. 
        k is approximately the number of specialists to select; see self.predictions(...) doc. 
        If no labels have ever been provided, returns zeroes.
        """
        default_corrs = np.zeros(len(self.spec_weights))
        label_corrs = default_corrs if self.label_corrs is None else self.label_corrs
        if k == -1:
            relevant_ndces = self._relevant_ndces
        elif k >= 0:
            relevant_ndces = self._calculate_topk_ndces(k)
        return label_corrs[relevant_ndces]

    # def featurize(self, dataset, k=-1):
    #     if k == -1:
    #         relevant_ndces = self._relevant_ndces
    #     elif k >= 0:
    #         relevant_ndces = self._calculate_topk_ndces(k)
    #     abst_indicators = self._specialists_onoff(dataset)
    #     abst_indicators = abst_indicators[:, relevant_ndces]
    #     base_preds = self.base_predict(dataset)
    #     # The following large matrix multiplication only involves relevant specialist indices, 
    #     # and can be much faster than computing all indices when only very few are relevant.
    #     preds_abst = abst_indicators.multiply(sp.sparse.lil_matrix(base_preds).transpose())
    #     signed_spec_wts = np.multiply(
    #         self.spec_weights[relevant_ndces], self._preds_flip[relevant_ndces])
    #     return (preds_abst.multiply(
    #         sp.sparse.lil_matrix(signed_spec_wts)), self.label_corrs[relevant_ndces])

    def base_predict(self, datamat):
        """
        Predictions of the base classifier on an input dataset.

        Args:
            datamat: Matrix of data on which to predict. Each row contains one example. 

        Returns:
            Vector of label predictions of the base classifier on each example.
        """
        return self._label_trans(self.base_classifier.predict(datamat))

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
        (2) Any derived specialist with nonzero Wilson bound (see muffled_utils.calc_b_bound(...) 
            docs for details on when this happens).
        """
        all_ndces = range(len(self._numsamps))
        relevant_ndces = set(np.nonzero(self.label_corrs)[0]) & set(all_ndces)
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
        assert self.label_corrs is not None , "Label correlations must be calculated before top k indices."
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
        When use_tree_partition == True, _specialists_onoff is incompatible with scikit-learn <=0.17. 
        This function provides a backward-compatible implementation for completeness; 
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


# def predict_multiple(classifier_arr, data, k=0, from_sklearn_rf=False, 
#     use_tree_partition=False, logging_interval=1, sample_counts=None):
#     listpreds = []
#     counter = 0
#     for h in classifier_arr:
#         counter += 1
#         compfeat = CompositeFeature(h, data, sample_counts=sample_counts, 
#             from_sklearn_rf=from_sklearn_rf, use_tree_partition=use_tree_partition)
#         preds = compfeat.predictions(data, k=k)
#         listpreds.append(preds)
#         if counter%logging_interval == 0:
#             print ('Classifier ' + str(counter) + ' done.')     # \tTime = ' + str(time.time() - inittime))
#     return sp.sparse.csr_matrix(sp.sparse.hstack(listpreds))

def predict_multiple(classifier_arr, x_out, x_unl, x_validate, y_out=None, 
    k=0, failure_prob=0.01, from_sklearn_rf=False, use_tree_partition=True, logging_interval=1):
    """
    Generate predictions on a holdout set and other datasets from a list of classifiers 
    and their derived specialists. 
    Args:
        classifier_arr: List of classifiers from which to derive specialists and 
            generate predictions on data.
        x_out: Holdout dataset from which to calculate label correlations of ensemble. 
            Matrix with {# examples} rows and {# features} cols.
        x_unl: As x_out, but unlabeled dataset.
        x_validate: As x_out, but validation dataset.
        y_out: Optional vector of holdout set's labels; defaults to None.
    Returns:
        4-tuple (A,B,C,D). A is a vector of label correlations of the input classifiers 
        and any of their derived specialists, measured on the holdout set (defaults to zeroes if y_out not given).
        B is a matrix of ensemble predictions on the holdout set, with {# classifiers} columns, 
        and {# examples} rows. C and D are similar to B, on unlabeled and validation sets respectively.
    """
    listfeats_unl = []
    listfeats_out = []
    listfeats_val = []
    listofbs = []
    counter = 0
    for h in classifier_arr:
        counter += 1
        compfeat = CompositeFeature(h, x_out, init_labels=y_out, failure_prob=failure_prob, 
            from_sklearn_rf=from_sklearn_rf, use_tree_partition=use_tree_partition)
        newb = compfeat.relevant_label_corrs(k=k)
        listofbs.append(newb)
        feats_out = compfeat.predictions(x_out, k=k)
        feats_unl = compfeat.predictions(x_unl, k=k)
        feats_val = compfeat.predictions(x_validate, k=k)
        listfeats_unl.append(feats_unl)
        listfeats_out.append(feats_out)
        listfeats_val.append(feats_val)
        if counter%logging_interval == 0:
            print ('Classifier ' + str(counter) + ' done.')     # \tTime = ' + str(time.time() - inittime))
    allfeats_unl = sp.sparse.csr_matrix(sp.sparse.hstack(listfeats_unl))
    allfeats_out = sp.sparse.csr_matrix(sp.sparse.hstack(listfeats_out))
    allfeats_val = sp.sparse.csr_matrix(sp.sparse.hstack(listfeats_val))
    b_vector = np.hstack(tuple(listofbs))
    return (b_vector, allfeats_out, allfeats_unl, allfeats_val)
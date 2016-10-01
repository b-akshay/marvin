import numpy as np
import scipy as sp
import time
import sklearn.metrics
import composite_feature
import slack_minimizer
import muffled_utils


class Marvin(object):
    """
    A class to sequentially learn and aggregate classifiers with the algorithm "Marvin", 
    as in the paper:: 
    "Muffled Semi-Supervised Learning," Balsubramani and Freund. 

    As the paper specifies, Marvin iteratively reweights a labeled and an unlabeled set 
    to learn classifiers which are aggregated in a growing ensemble, analogous 
    to boosting algorithms like AdaBoost. 
    This implementation also uses a holdout set to accurately estimate 
    classifier label correlations, necessary for step size line search. 
    Attributes:
        base_learner: The base learner, a classification algorithm following the sklearn API. 
            Can be trained on a weighted dataset to return a classifier with low error on that set.
        b_vector: Vector of label correlation bounds for the ensemble classifiers. 
            Needed for step size line search and total correction; estimated with holdout set.
        weights: Weight vector for the ensemble classifiers.
        enslist: List specifying the classifiers in the ensemble in the order they are added. 
            Each entry is a pair, with the first element being a composite_feature object and 
            the second being an integer k >= 0 representing the number of derived specialists added 
            (see composite_feature.featurize(...) doc).
        labeled_set: Matrix of labeled data, {# examples} rows and {# features} cols.
        labeled_labels: Vector containing the labels of the labeled data.
        unlabeled_set: As labeled_set, but with the unlabeled dataset. 
        holdout_set: As labeled_set, but with the holdout dataset. 
        holdout_labels: Vector containing the labels of the holdout data. 
        unlabeled_labels: Vector containing the labels of the unlabeled data. 
            Used only for diagnostics, not for learning.
        validation_set: As labeled_set, but with the validation dataset. 
            This is held out from the entire learning procedure. 
        validation_labels: Vector containing the labels of the validation data. 

    This implementation assumes that all three datasets fit in memory. They are kept as 
    class variables because they are used repeatedly every time a classifier is learned and 
    specialists are derived from it, to add to the ensemble.
    """

    def __init__(self, base_learner, labeled_set, labeled_labels, 
                 unlabeled_set, holdout_set, holdout_labels, unlabeled_labels=None, 
                 validation_set=None, validation_labels=None, init_hyp=None, 
                 num_data_to_track=200, use_tree_partition=True):
        self.base_learner = base_learner
        self.enslist = []
        self.labeled_set = labeled_set
        self.unlabeled_set = unlabeled_set
        self.labeled_labels = labeled_labels
        self.unlabeled_labels = unlabeled_labels
        self.holdout_set = holdout_set
        self.holdout_labels = holdout_labels
        if validation_set is None:
            self.validation_set = holdout_set
            self.validation_labels = holdout_labels
        else:
            self.validation_set = validation_set
            self.validation_labels = validation_labels
        self.b_vector = np.array([])
        self.weights = np.array([])
        # Keep track of ensemble predictions so far, so that adding a new classifier 
        # with total correction does not involve generating predictions for the whole 
        # ensemble (performance bottleneck).
        self._labeled_set_feats = []
        self._unlabeled_set_feats = []
        self._holdout_set_feats = []
        self._validation_set_feats = []
        self._lab_MCsamples = np.random.choice(
            range(self.labeled_set.shape[0]), num_data_to_track)
        self._unl_MCsamples = np.random.choice(
            range(self.unlabeled_set.shape[0]), num_data_to_track)
        self._val_MCsamples = np.random.choice(
            range(self.validation_set.shape[0]), num_data_to_track)
        # Tracking unlabeled scores is useful for efficiently running Marvin; 
        # the rest are for diagnostics/plots.
        self._scoresunl = np.zeros(self.unlabeled_set.shape[0])
        self._scoreslab = np.zeros(self.labeled_set.shape[0])
        self._scoresout = np.zeros(self.holdout_set.shape[0])
        self._scoresval = np.zeros(self.validation_set.shape[0])
        # The following is true iff base_learner is an sklearn tree with its nodes as derived specialists.
        self._use_tree_partition = use_tree_partition
        # Initialize with the best single (generalist) classifier, on just the labeled data.
        if (init_hyp is None) and (base_learner is not None):
            init_hyp = self.base_learner.fit(self.labeled_set, self.labeled_labels)
        self.add_classifier(init_hyp, total_correction=False, correction_duration=10, 
            init_stepsize=3.0, failure_prob=0.01, k=0)

    def aggregate(self, duration, correction_interval=1, correction_duration=10, k=0, 
        failure_prob=0.001, unl_stride_size=0, logging_interval=1, tol=0.00005):
        """
        Run the Marvin classifier aggregation algorithm.
        Args:
            duration: int; Number of timesteps to run. In other words, number of calls 
                to self.base_learner.
            correction_interval: optional int; Number of iterations between totally corrective 
                updates. Defaults to 1 (every update is totally corrective). 0 indicates that 
                updates should not be totally corrective. 
            k: optional int; Approximate number of derived specialists to generate 
                from the new classifier. See composite_feature.featurize(...) docs.
            correction_duration: optional int; Number of iterations of totally corrective SGD 
                to run each iteration. Increase if total correction is not near reaching equilibrium.
            failure_prob: optional float; The allowed failure (tail) probability used to define 
                the Wilson error bound on the new classifier.
            logging_interval: optional int > 0; Interval between timesteps on which statistics are computed.
            unl_stride_size: optional int; The number of unlabeled data per minibatch. 
                Defaults to 0, which indicates that the whole unlabeled set should be used 
                each minibatch. 
            tol: optional float; Tolerance used to terminate line search.
        Returns:
            A pair giving performance statistics at each iteration. The first element 
            is a list of 6-tuples of floats, with the first two elements giving the error and AUC 
            on the unlabeled data, the next two on the holdout data, and the last two 
            on the validation data, as in slack_minimizer.sgd(...). 
            The second element of the pair is a 3-tuple, whose entries are each arrays 
            of the scores of the tracked data from (labeled, unlabeled, validation) sets 
            respectively, at each time on which they are logged.
        """
        inittime = time.time()
        numunl = self.unlabeled_set.shape[0]
        if unl_stride_size == 0:
            unl_stride_size = numunl
        statauc = []
        lab_particles = []
        unl_particles = []
        val_particles = []
        # TODO(Akshay): Raise ValueError if correction_interval < 0.
        # self.y_trainMC = np.array(self.labeled_labels)[self._lab_MCsamples]
        # self.y_testMC = np.array(self.unlabeled_labels)[self._unl_MCsamples]
        # self.y_outMC = np.array(self.validation_labels)[self._val_MCsamples]
        time_to_start = 5
        if time_to_start == -1:
            time_to_start = len(self.enslist)
        for iterct in range(duration):
            unl_indices_this_iteration = np.random.choice(
                numunl, unl_stride_size, replace=False)
            newhyp = self.learn_new_classifier(unl_indices_this_iteration=unl_indices_this_iteration)
            correct_this_iteration = bool(correction_interval and (iterct%correction_interval == 0))
            stepsize = 0.5*np.power(iterct + time_to_start, -0.5)
            self.add_classifier(newhyp, total_correction=correct_this_iteration, 
                correction_duration=10, init_stepsize=stepsize, failure_prob=failure_prob, 
                k=k, tol=tol)

            # TODO Write an ipynb running this, logging results to files, and plotting particle score trajectories.
            if iterct%logging_interval == 0:
                lab_particles.append(self._scoreslab[self._lab_MCsamples])
                unl_particles.append(self._scoresunl[self._unl_MCsamples])
                val_particles.append(self._scoresval[self._val_MCsamples])
                preds_unl = np.clip(self._scoresunl, -1, 1)
                alg_loss_unl = 1.0 - muffled_utils.accuracy_calc(self.unlabeled_labels, preds_unl)
                alg_auc_unl = sklearn.metrics.roc_auc_score(self.unlabeled_labels, preds_unl)
                preds_out = np.clip(self._scoresout, -1, 1)
                alg_loss_out = 1.0 - muffled_utils.accuracy_calc(self.holdout_labels, preds_out)
                alg_auc_out = sklearn.metrics.roc_auc_score(self.holdout_labels, preds_out)
                preds_val = np.clip(self._scoresval, -1, 1)
                alg_loss_val = 1.0 - muffled_utils.accuracy_calc(self.validation_labels, preds_val)
                alg_auc_val = sklearn.metrics.roc_auc_score(self.validation_labels, preds_val)
                print 'After iteration  ' + str(iterct) + ':\t Time = ' + str(time.time() - inittime)
                print('Holdout: \t Error = ' + str(alg_loss_out) + '\t AUC: ' + str(alg_auc_out))
                print('Validation: \t Error = ' + str(alg_loss_val) + '\t AUC: ' + str(alg_auc_val))
                print('Length of weight vector:\t ' + str(self.weights.shape[0]))
                statauc.append((alg_loss_unl, alg_auc_unl, alg_loss_out, 
                    alg_auc_out, alg_loss_val, alg_auc_val))
        lab_particles = np.array(lab_particles)
        unl_particles = np.array(unl_particles)
        val_particles = np.array(val_particles)
        return (statauc, (lab_particles, unl_particles, val_particles))

    # def compute_AUC(self, dataset=None, labels=None):
    #     if dataset is None:
    #         dataset = self.validation_set
    #         labels = self.validation_labels
    #     return sklearn.metrics.roc_auc_score(labels, self.predict(dataset))
    # TODO(Akshay): Currently, we support the same step size for all the new specialists, or total correction i.e. 
    # different weights for each. But what about different step sizes for each which aren't totally corrected?

    def add_classifier(self, new_classifier, total_correction=False, 
        correction_duration=10, failure_prob=0.01, k=0, tol=0.00005, init_stepsize=1.0):
        """
        Adds a new classifier and its derived specialists to the ensemble. 
        Args:
            new_classifier: Classifier to add to the ensemble, following the scikit-learn API.
            total_correction: optional Boolean; True for a totally corrective update. Defaults to False.
            k: optional int; Approximate number of derived specialists to generate from the new classifier. 
            correction_duration: optional int; Number of iterations of totally corrective SGD run 
                each iteration. Only increase if total correction is not fully correcting.
            failure_prob: optional float; The allowed failure (tail) probability used to define 
                the Wilson error bound on the new classifier.
            tol: optional float; Tolerance used to terminate line search procedure.
            init_stepsize: optional float; maximum step size when the update is not totally corrective. 
        """
        cl_composite = composite_feature.CompositeFeature(
            new_classifier, self.holdout_set, self.holdout_labels, 
            failure_prob=failure_prob, use_tree_partition=self._use_tree_partition)
        self.enslist.append((cl_composite, k))
        xtra_out, newb = cl_composite.featurize(self.holdout_set, k=k)
        xtra_unl, _ = cl_composite.featurize(self.unlabeled_set, k=k)
        xtra_val, _ = cl_composite.featurize(self.validation_set, k=k)
        xtra_lab, _ = cl_composite.featurize(self.labeled_set, k=k)
        self._labeled_set_feats.append(xtra_lab)
        self._unlabeled_set_feats.append(xtra_unl)
        self._holdout_set_feats.append(xtra_out)
        self._validation_set_feats.append(xtra_val)
        feats_lab = sp.sparse.csr_matrix(sp.sparse.hstack(self._labeled_set_feats))
        feats_unl = sp.sparse.csr_matrix(sp.sparse.hstack(self._unlabeled_set_feats))
        feats_out = sp.sparse.csr_matrix(sp.sparse.hstack(self._holdout_set_feats))
        feats_val = sp.sparse.csr_matrix(sp.sparse.hstack(self._validation_set_feats))
        # print 'SHAPES:\t ' , self.b_vector.shape , '\t' , self.weights.shape
        #print 'Projected slack function evaluation:  ' + str(self._proj_slack_newdir(newb, xtra_unl)(1))
        if not total_correction:
            # Weights are the same for all the new specialists added this iteration.
            assert self.weights.shape[0] == self.b_vector.shape[0] , "Ensemble weights or label correlations incorrectly updated."
            stepsize = muffled_utils.golden_section_search(
                self._proj_slack_newdir(newb, xtra_unl), 0, 2.0*init_stepsize, tol=tol)
            self.weights = np.append(self.weights, stepsize * np.ones(len(newb)))
            self.b_vector = np.append(self.b_vector, newb)
        else:
            # Pad weight vector to required new length and use slack_minimizer for total correction.
            self.b_vector = np.append(self.b_vector, newb)
            wts = np.append(self.weights, [0]*len(newb))
            gradh = slack_minimizer.SlackMinimizer(
                self.b_vector, feats_unl, feats_out, self.holdout_labels, 
                unlabeled_labels=self.unlabeled_labels, validation_set=feats_val, 
                validation_labels=self.validation_labels, weights=wts)
            statauc = gradh.sgd(correction_duration, unl_stride_size=100, logging_interval=0)
            self.weights = gradh.weights
        self._scoresunl = feats_unl.dot(self.weights)
        self._scoreslab = feats_lab.dot(self.weights)
        self._scoresout = feats_out.dot(self.weights)
        self._scoresval = feats_val.dot(self.weights)

    def predict(self, data):
        """
        Predict on input data with the current aggregated classifier learned by Marvin.
        Args:
            data: Matrix of data on which to predict. {# examples} rows, {# features} cols. 
        Returns:
            A matrix of Marvin's predictions, with {# examples} rows, and one column for each 
            base classifier and derived specialist in the current ensemble 
            (i.e. {# cols} == len(self.weights)).
        """
        featlist = []
        for cl, k in self.enslist:
            predictions, _ = cl.featurize(data, k=k)
            featlist.append(predictions)
        ft = sp.sparse.csr_matrix(sp.sparse.hstack(featlist))
        scores = ft.dot(self.weights)
        return np.clip(scores, -1, 1)

    def learn_new_classifier(self, unl_indices_this_iteration=None):
        """
        Learn the best new classifier to add to the current ensemble.
        Args:
            indices_this_iteration: Optional list of indices in the unlabeled dataset, 
                specifying a subset to be used for learning (defaults to all of it).
        Returns:
            A classifier following the scikit-learn API, learned using self.base_learner.
        """
        if unl_indices_this_iteration is None:
            unl_indices_this_iteration = range(self.unlabeled_set.shape[0])
        dataset = self._hallucinate_set(indices_this_iteration=unl_indices_this_iteration)
        return self.base_learner.fit(dataset[0], dataset[1], sample_weight=dataset[2])

    def _proj_slack_newdir(self, new_b_vals, new_preds_unl):
        """
        Projects the slack function in the direction of one or more new classifiers.
        Args:
            new_b_vals: Vector of lower bounds on label correlations of the new ensemble classifiers.
            new_preds_unl: A (# examples) x (# classifiers) matrix of the new classifiers'
                predictions on unlabeled data.
        Returns:
            A function (closure) that evaluates the projected slack function given an 
            input step size alpha.
        """
        def _toret(alpha):
            scores = self._scoresunl + alpha*new_preds_unl.dot(
                np.ones(new_preds_unl.shape[1]))
            return -np.dot(self.b_vector, self.weights) - alpha*np.sum(new_b_vals) + np.mean(
                np.maximum(np.abs(scores), 1.0)) 
        return _toret

    def _hallucinate_labels(self, scores=None):
        """ 
        Calculate hallucinated labels for dataset using given scores, 
        which default to current unlabeled scores. We treat borderline labels as clipped, 
        to avoid problems of zero gradient upon initialization. 
        We also set labels on hedged examples to zero instead of random fair 
        binary coin flips, which reduces variance and improves performance. 
        """
        if scores is None:
            scores = self._scoresunl
        ghlabels = np.sign(scores)
        ghlabels[np.where(np.abs(scores) < 1.0)] = 0.0
        return ghlabels

    def _hallucinate_set(self, indices_this_iteration=None):
        """
        Hallucinate reweighted dataset on which to train the next classifier used by Marvin. 
        Optional arg contains indices of a subset of the unlabeled dataset to be used (defaults to all of it).
        """
        numlbl = self.labeled_set.shape[0]
        if indices_this_iteration is None:
            indices_this_iteration = range(self.unlabeled_set.shape[0])
        numunl_this_iteration = len(indices_this_iteration)
        newtrainset = np.vstack((self.labeled_set, 
            self.unlabeled_set[indices_this_iteration, :]))  # (numlbl+numunl_this_iteration) rows
        dreamed_labels = self._hallucinate_labels(scores=self._scoresunl[indices_this_iteration])
        samplbls = np.hstack(
            (np.array(self.labeled_labels), -1.0 * np.sign(dreamed_labels)))
        sampwts = np.zeros(numlbl + numunl_this_iteration)
        sampwts[0:numlbl] = (1.0/numlbl) * np.ones(numlbl)
        sampwts[numlbl:(numlbl+numunl_this_iteration)] = (1.0/numunl_this_iteration) * np.abs(dreamed_labels)
        return (newtrainset, samplbls, sampwts)



"""
Script demonstrating class usage. 

Usage: [TODO(Akshay): Change the usage to correspond to the example]
python marvin.py <data file> <total_labels> <number of iterations> 
<number of Monte Carlo trials> 

Example:
python marvin.py Data/covtype_binary_all.csv 10000 100 
"""
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    import csv, argparse

    parser = argparse.ArgumentParser(
        description="""
        Run the Marvin algorithm, using a base learner to sequentially learn an ensemble of classifiers to aggregate.
        Contact: Akshay Balsubramani <abalsubr@ucsd.edu>
        """)
    parser.add_argument('labeled_file', help='CSV file from which to read data.')
    parser.add_argument('total_labels', type=int, 
        help='Number of labeled data in total (to be divided between training set for the ensemble and holdout for calculating b).')
    parser.add_argument('--unlabeled_set_size', '-u', type=int, default=1000,
        help='Number of unlabeled data.')
    parser.add_argument('--failure_prob', '-f', type=float, default=0.05, 
        help="""Failure probability for calculating ensemble error bounds (i.e., 
        Wilson intervals on label correlations) used for step size line search.""")
    parser.add_argument('--num_trials', '-n', type=int, dest='num_MCtrials', default=1, 
        help='Number of Monte Carlo trials of the experiment to run.')
    parser.add_argument('--k', '-k', type=int, default=0, 
        help="""Approximate number of derived specialists to generate from each base classifier.
        More precisely, the best k (by Wilson error bound) are taken, along with the base classifier if it is not already 
        one of the best k. See composite_feature.featurize(...) docs for more details. Defaults to 0.""")
    parser.add_argument('--validation_set_size', '-v', type=int, default=1000, 
        help='Number of validation data.')
    args = parser.parse_args()

    root = 'marvin'
    labeled_set_size = int(args.total_labels*0.75)
    # unlabeled_set_size = max(1000, 2*args.total_labels)
    holdout_set_size = args.total_labels - labeled_set_size
    logname_auc = 'logs/' + root + '_' + str(args.total_labels) + '_tailprob' + str(args.failure_prob) + '_auc.csv'
    logname_err = 'logs/' + root + '_' + str(args.total_labels) + '_tailprob' + str(args.failure_prob) + '_err.csv'
    inittime = time.time()
    with open(logname_auc, 'a') as fo_auc, open(logname_err, 'a') as fo_err:
        writer_auc = csv.writer(fo_auc)
        writer_err = csv.writer(fo_err)
        for mctrial in range(args.num_MCtrials):
            print 'Trial ' + str(mctrial) + ':\tTime = ' + str(time.time() - inittime)
            (x_train, y_train, x_unl, y_unl, x_out, y_out, x_validate, y_validate) = muffled_utils.read_random_data_from_csv(
                args.labeled_file, labeled_set_size, args.unlabeled_set_size, holdout_set_size, args.validation_set_size)
            print('Data loaded. \tTime = ' + str(time.time() - inittime))
            marv = Marvin(DecisionTreeClassifier(max_depth=None), x_train, y_train, 
                x_unl, x_out, y_out, unlabeled_labels=y_unl, 
                validation_set=x_validate, validation_labels=y_validate)
            statauc, (lab_particles, unl_particles, val_particles) = marv.aggregate(
                50, k=0, failure_prob=args.failure_prob, correction_duration=10, 
                correction_interval=1, unl_stride_size=1000)
            _, _, _, _, cl_err, cl_auc = zip(*statauc)
            print cl_auc
            print cl_err
            writer_auc.writerow(cl_auc)
            writer_err.writerow(cl_err)
    print 'Written to files:\t' + logname_auc + ',\t ' + logname_err
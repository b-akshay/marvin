import numpy as np
import scipy as sp
import math
import time
import sklearn.metrics
import SSutils
import scipy.special
import matplotlib.pyplot as plt

""" This class is just for data that can fit in memory, which are passed to the initial instance. """

class slackMinimizer (object):
    """
    A class that aggregates a given set of classifiers, as in the paper "Scalable semi-supervised aggregation of classifiers."
    "labeled_set" and "unlabeled_set" are (# examples) x (# classifiers) matrices, one row per example. We assume that they fit in memory.
    "labels" is a vector of the labels in the labeled_set.
    """

    def __init__(self, unlabeled_set, unlabeled_labels, holdout_set, holdout_labels, labeled_set=None, labels=None, 
                 outdiag_set=None, outdiag_labels=None, b_vector=None, sigma=None, b_calc='out'):
        self.unlabeled_set = unlabeled_set
        self.unlabeled_labels = unlabeled_labels
        self.holdout_set = holdout_set
        self.holdout_labels = holdout_labels
        self.labeled_set = labeled_set
        self.labels = labels
        self.outdiag_set = outdiag_set
        self.outdiag_labels = outdiag_labels
        
        self.b_vector = b_vector

        # Initialize sigma to a random vector, and scale them so the data have average margin 1.
        if sigma is None:
            sigma = np.random.randn(self.unlabeled_set.shape[1])
            margsunl = self.unlabeled_set.dot(sigma)
            meanmarg = 1.0/np.mean(np.abs(margsunl))
            self.sigma = meanmarg*sigma
            self.margsunl = meanmarg*margsunl     # vector of <x, sigma> values, one for each of numunl examples
            self.margsout = meanmarg*self.holdout_set.dot(sigma)
            self.margdiag = meanmarg*self.outdiag_set.dot(sigma)
        else:
            self.sigma = sigma
            self.margsunl = self.unlabeled_set.dot(np.array(sigma))
            self.margsout = self.holdout_set.dot(np.array(sigma))
            self.margdiag = self.outdiag_set.dot(np.array(sigma))

        self.logloss = False
    
    
    def projslackfunc(self, search_dir):
        def toret(alpha):
            margs = self.margsunl + self.unlabeled_set.dot(alpha*search_dir)
            return -np.dot(self.b_vector, self.sigma + alpha*search_dir) + np.mean(np.maximum(np.abs(margs), 1.0)) 
        return toret


    def hallucinate_labels (self, margins=None):
        # Calculate hallucinated labels for each unlabeled data point, at given margs or unlabeled. 
        # TODO: Different losses give different gradients here. 
        if margins is None:
            margins = self.margsunl
        ghlabels = np.sign(margins)
        ghlabels[np.where(np.abs(margins) < 1)] = 0
        if self.logloss:
            ghlabels = 2.0*scipy.special.expit(margins) - 1
        return ghlabels

    
    def calc_grad (self, indices_this_iteration=[]):
        if len(indices_this_iteration) == 0:
            indices_this_iteration = range(len(self.unlabeled_labels))
        unl_set = self.unlabeled_set[indices_this_iteration,:]
        return -self.b_vector + (1.0/len(indices_this_iteration))*unl_set.transpose().dot(
            self.hallucinate_labels(margins=self.margsunl[indices_this_iteration]))


    # Assumes dataset is a sparse csr_matrix, with a number of columns equal to the actual number of hypotheses (via composite_feature.featurize()). 
    def predict (self, dataset):
        return np.clip(dataset.dot(self.sigma), -1, 1)


    def AUC (self, dataset, data_labels):
        return sklearn.metrics.roc_auc_score(data_labels, self.predict(dataset))

    
    # Run duration iterations of the algorithm, starting at iteration timestart with current sigma. 
    # Holdout info and test labels should be full already.
    def sgd (self, duration, timetostart=5, linesearch=True, linesearch_tol=0.00005, projection='pos', verbose=False, unl_stride=False, unl_stride_size=5000):
        inittime = time.time()
        statauc = []
        EMA_stepsize = 0.2  # exponential moving average, with parameter beta; initialized to some reasonable constant
        beta = 0.5 # exponential moving average decay parameter in [0,1)

        for iterct in range(timetostart, timetostart+duration):
            if not unl_stride:
                unl_stride_size = len(self.unlabeled_labels)
            indices_this_iteration = np.random.choice(self.unlabeled_set.shape[0], unl_stride_size, replace=False)
            grad = self.calc_grad(indices_this_iteration=indices_this_iteration)
            
            if linesearch:
                max_stepsize = 2*EMA_stepsize
                stepsize = SSutils.golden_section_search(self.projslackfunc(-grad), 0, max_stepsize, tol=linesearch_tol)
                EMA_stepsize = beta*EMA_stepsize + (1-beta)*stepsize
                #if verbose:
                #    print stepsize, max_stepsize
            else:
                stepsize = 1.0*np.power(iterct, -0.5)

            weight_to_add = -stepsize*grad
            self.sigma += weight_to_add
            
            # Project down if necessary, otherwise leave the sigma alone.
            if projection == 'pos':
                self.sigma = np.maximum(self.sigma, 0.0)

            self.margsunl = self.unlabeled_set.dot(self.sigma)
            self.margsout = self.holdout_set.dot(self.sigma)
            self.margdiag = self.outdiag_set.dot(self.sigma)

            preds_unl = np.clip(self.margsunl, -1, 1)
            preds_out = np.clip(self.margsout, -1, 1)
            preds_diag = np.clip(self.margdiag, -1, 1)
            alg_loss_unl = 1.0 - SSutils.accuracy_calc (self.unlabeled_labels, preds_unl)
            alg_loss_out = 1.0 - SSutils.accuracy_calc (self.holdout_labels, preds_out)
            alg_loss_diag = 1.0 - SSutils.accuracy_calc (self.outdiag_labels, preds_diag)
            alg_auc_unl = sklearn.metrics.roc_auc_score (self.unlabeled_labels, preds_unl)
            alg_auc_out = sklearn.metrics.roc_auc_score (self.holdout_labels, preds_out)
            alg_auc_diag = sklearn.metrics.roc_auc_score (self.outdiag_labels, preds_diag)
            
            if self.logloss:
                ll_preds_unl = 2.0*scipy.special.expit(self.margsunl) - 1
                ll_preds_out = 2.0*scipy.special.expit(self.margsout) - 1
                vect_preds_unl = np.array([[0.5*(1-x), 0.5*(1+x)] for x in ll_preds_unl])
                vect_preds_out = np.array([[0.5*(1-x), 0.5*(1+x)] for x in ll_preds_out])
                ll_loss_unl = sklearn.metrics.log_loss(self.unlabeled_labels, vect_preds_unl)
                ll_loss_out = sklearn.metrics.log_loss(self.holdout_labels, vect_preds_out)
            
            besthyp = np.argmax(np.abs(grad))
            bestgrad = np.max(np.abs(grad))

            if (iterct%30 == 0 or (iterct%5 == 0 and verbose)):
                print(iterct, time.time() - inittime) 
                #print('Unlabeled error:  ' + str(alg_loss_unl) + ' and AUC: ' + str(alg_auc_unl))
                print('Holdout error:  ' + str(alg_loss_out) + ' and AUC: ' + str(alg_auc_out))
                print('Diagnostic error:  ' + str(alg_loss_diag) + ' and AUC: ' + str(alg_auc_diag))
            statauc.append((alg_loss_unl, alg_auc_unl, alg_loss_out, alg_auc_out, alg_loss_diag, alg_auc_diag))
        # We must eventually return both particles and margs, because margs is over all data and not just the Monte Carloed stuff.
        return statauc



"""
Usage: 
python slack_minimizer.py <Wilson failure probability> < k (# top classifiers)> 
<mode setting {allrel, topk}> <number of Monte Carlo trials> <OPTIONAL: is_tree>

Example:
python slack_minimizer.py Data/covtype_binary_all.csv 10000 100000 covtype 0.005 0 allrel 20 False
creates file: logs/covtype_tree_0.005_allrel_10000.csv
"""

"""
if __name__ == "__main__":

    labeled_file = sys.argv[1]
    TOTAL_LABELS = int(sys.argv[2])
    UNLABEL_SET_SIZE = int(sys.argv[3])
    root = sys.argv[4]
    failure_prob = float(sys.argv[5])
    k = int(sys.argv[6])
    modesetting = sys.argv[7]
    num_MCtrials = int(sys.argv[8])

    if len(sys.argv) >= 10:
        num_iters = int(sys.argv[9])
    else:
        num_iters = 100

    LABELED_SET_SIZE = int(TOTAL_LABELS*0.3)
    HOLDOUT_SET_SIZE = TOTAL_LABELS - LABELED_SET_SIZE
    HOLDOUT_SET2_SIZE = 100

    TOTAL_SIZE = LABELED_SET_SIZE+UNLABEL_SET_SIZE+HOLDOUT_SET_SIZE+HOLDOUT_SET2_SIZE
    classstr = '_'

    logfile_auc = 'logs/' + root + classstr + str(failure_prob) + '_' + modesetting + '_' + str(TOTAL_LABELS) + 'auc.csv'
    logfile_err = 'logs/' + root + classstr + str(failure_prob) + '_' + modesetting + '_' + str(TOTAL_LABELS) + 'err.csv'

    inittime = time.time()
    with open( logfile_auc, 'a' ) as fo_auc, open( logfile_err, 'a' ) as fo_err:
        writer_auc = csv.writer( fo_auc)
        writer_err = csv.writer( fo_err)
        for mctrial in range(num_MCtrials):
            (x_all, y_all) = read_data (labeled_file, TOTAL_SIZE)
            print 'Data read: ' + str(time.time() - inittime)
            (x_train, y_train, x_unl, y_unl, x_out, y_out, x_out2, y_out2) = init_data(
                x_all, y_all, LABELED_SET_SIZE, UNLABEL_SET_SIZE, HOLDOUT_SET_SIZE, HOLDOUT_SET2_SIZE)
            print mctrial, time.time() - inittime
            skcl = init_base_classifiers(x_train, y_train, num_iters=num_iters)
            rf = skcl[0][1]
            line = []
            (xtra_b, xtra_unl, xtra_out, xtra_out2, list_compfeats) = gen_compfeats(rf.estimators_, y_out, x_out, x_unl, 
                x_out2, k=k, failure_prob=failure_prob, mode=modesetting, numsamp_min=0, is_sklearn_rf=True, is_tree=True)
            gradh = slack_minimizer.slackMinimizer(xtra_unl, y_unl, xtra_out, y_out, b_vector=xtra_b, 
                outdiag_set=xtra_unl, outdiag_labels=y_unl)
            statauc = gradh.sgd(50, verbose=True, unl_stride=True, unl_stride_size=100)
            print 'ACTIVE NODES: ' + str(np.mean([len(h.relevant_ndces) for h in list_compfeats]))
            print 'TOTAL NODES: ' + str(np.mean([h.numnodes for h in list_compfeats]))
            print 'AUC: ' + str (gradh.AUC(xtra_unl, y_unl))
            _, _, _, _, cl_err, cl_auc = zip(*statauc)
            writer_auc.writerow( cl_auc )
            writer_err.writerow( cl_err )

"""
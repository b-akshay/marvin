import numpy as np
import scipy as sp
import math
import time
import hedgeclipper
import composite_feature
import sklearn.metrics
import scipy.special
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
import SSutils

""" This class is just for data that can fit in memory, which are passed to the initial instance. """

class Marvin (object):
    """
    A class that sequentially creates and aggregates classifiers to a combined model, as in the paper "Semi-Supervised Boosting."
    "labeled_set" and "unlabeled_set" are (# examples) x (# features) matrices, one row per example. We assume that they fit in memory.
    "labels" and "unlabeled_labels" are vectors of the labels in the labeled_set and unlabeled_set.
    "wlearner" is the weak learner, an algorithm following the sklearn API.
    
    To do corrective updates, we need the holdout set, from which we calculate b.
    
    self.enslist = list of tree features comprising the ensemble
    """

    def __init__(self, wlearner, 
                 labeled_set, labels, unlabeled_set, unlabeled_labels=None, 
                 holdout_set=None, holdout_labels=None, outdiag_set=None, outdiag_labels=None, 
                 init_hyp=None, num_MC=200):
        self.labeled_set = labeled_set
        self.labels = labels
        self.unlabeled_set = unlabeled_set
        self.unlabeled_labels = unlabeled_labels
        self.holdout_set = holdout_set
        self.holdout_labels = holdout_labels
        self.outdiag_set = outdiag_set
        self.outdiag_labels = outdiag_labels

        self.wlearner = wlearner
        self.enslist = []
        self.sigma = []
        self.labeled_set_feats = []
        self.unlabeled_set_feats = []
        self.holdout_set_feats = []
        self.outdiag_set_feats = []
        
        self.num_MC = num_MC
        self.logloss = False
        
        self.margslab = np.zeros(self.labeled_set.shape[0])
        self.margsunl = np.zeros(self.unlabeled_set.shape[0])
        self.margsout = np.zeros(self.holdout_set.shape[0])
        self.margdiag = np.zeros(self.outdiag_set.shape[0])
        
        # When corrective, need to track b vector for learned features, using the holdout set.
        self.b = []

        # Initialize with the best single (generalist) hypothesis, on just the labeled data.
        if (init_hyp is None):# and (wlearner is not None):
            init_hyp = self.wlearner.fit(self.labeled_set, self.labels)
        self.add_composite_feature (init_hyp, total_correction=False, correction_duration=10, stepsize=3.0, 
                               failure_prob=0.01, numsamp_min=0, mode='topk', k=0)
        
        
    def newhyp(self, indices_this_iteration=[]):
        dataset = self.hallucinate_set(indices_this_iteration=indices_this_iteration)
        return self.wlearner.fit(dataset[0], dataset[1], sample_weight=dataset[2])

    
    def hallucinate_labels (self, margins=None):
        # Calculate hallucinated labels for each unlabeled data point. 
        # TODO: Different losses give different gradients here. 
        if margins is None:
            margins = self.margsunl
        ghlabels = np.sign(margins)
        ghlabels[np.where(np.abs(margins) < 1.0)] = 0.0
        #if self.logloss:
        #    ghlabels = 2.0*scipy.special.expit(margins) - 1
        return ghlabels

        
    def hallucinate_set (self, indices_this_iteration=[]):
        numlbl = self.labeled_set.shape[0]
        numunl = self.unlabeled_set.shape[0]
        if len(indices_this_iteration) == 0:
            indices_this_iteration = range(numunl)
        numunl_this_iteration = len(indices_this_iteration)
        
        newtrainset = np.vstack((self.labeled_set, self.unlabeled_set[indices_this_iteration, :]))  # ((numlbl+numunl) x d) matrix
        ghlabels = self.hallucinate_labels (margins=self.margsunl[indices_this_iteration])
        
        samplbls = np.hstack((np.array(self.labels), -1.0*np.sign(ghlabels)))
        sampwts = np.zeros(numlbl+numunl_this_iteration)
        sampwts[0:numlbl] = (1.0/numlbl)*np.ones(numlbl)
        # The following line zeroes out the weights on hedged examples, for zero-one loss.
        sampwts[numlbl:(numlbl+numunl_this_iteration)] = (1.0/numunl_this_iteration)*np.abs(ghlabels)
        return (newtrainset, samplbls, sampwts)
        
    
    """ 
    Totally corrective version of the algorithm, correcting every correction_interval steps, 
    for correction_duration steps of gradient descent with golden section search. 
    """
    def boost (self, duration, timetostart=-1, correction_interval=1, correction_duration=10, 
               one_at_a_time=False, mode='allrel', k=10, numsamp_min=0, failure_prob=0.001, 
               unl_stride=False, unl_stride_size=5000, tol=0.00005):
        inittime = time.time()
        numlab = self.labeled_set.shape[0]
        numunl = self.unlabeled_set.shape[0]
        numdiag = self.outdiag_set.shape[0]
        if timetostart==-1:
            timetostart = len(self.enslist)

        statauc = []
        lab_particles = []
        unl_particles = []
        out_particles = []
        
        trainMCs = np.random.choice(range(numlab), self.num_MC)
        testMCs = np.random.choice(range(numunl), self.num_MC)
        diagMCs = np.random.choice(range(numdiag), self.num_MC)
        
        self.y_trainMC = np.array(self.labels)[trainMCs]
        self.y_testMC = np.array(self.unlabeled_labels)[testMCs]
        self.y_outMC = np.array(self.outdiag_labels)[diagMCs]
        

        for iterct in range(timetostart, timetostart+duration):
            if not unl_stride:
                unl_stride_size = numunl
            unl_indices_this_iteration = np.random.choice(numunl, unl_stride_size, replace=False)
            
            newhyp = self.newhyp(indices_this_iteration=unl_indices_this_iteration)
            correct_this_iteration = (not one_at_a_time) and (iterct%correction_interval == 0)
            stepsize = 0.5*np.power(iterct, -0.5)
            if one_at_a_time:
                mode = 'topk'
                k = 0
                
            #TODO: check this line for unl_indices_this_iteration
            self.add_composite_feature (newhyp, total_correction=correct_this_iteration, correction_duration=10, 
                                   stepsize=stepsize, failure_prob=failure_prob, numsamp_min=numsamp_min, mode=mode, k=k, tol=tol)
            
            print 'Length of weight vector, iteration ' + str(iterct) + ': ' + str(self.sigma.shape[0])
            
            lab_particles.append(self.margslab[trainMCs])
            unl_particles.append(self.margsunl[testMCs])
            out_particles.append(self.margdiag[diagMCs])

            preds_unl = np.clip(self.margsunl, -1, 1)
            preds_out = np.clip(self.margsout, -1, 1)
            preds_diag = np.clip(self.margdiag, -1, 1)
            alg_acc_unl = SSutils.accuracy_calc (self.unlabeled_labels, preds_unl)
            alg_acc_out = SSutils.accuracy_calc (self.holdout_labels, preds_out)
            alg_acc_diag = SSutils.accuracy_calc (self.outdiag_labels, preds_diag)
            alg_auc_unl = sklearn.metrics.roc_auc_score (self.unlabeled_labels, preds_unl)
            alg_auc_out = sklearn.metrics.roc_auc_score (self.holdout_labels, preds_out)
            alg_auc_diag = sklearn.metrics.roc_auc_score (self.outdiag_labels, preds_diag)
            
            if (len(self.enslist)%1 == 0):
                print('ITERATION ' + str(iterct) + ': ', #time.time() - inittime, 
                      'Holdout err:  ' + str(alg_acc_out), #+ ' and AUC: ' + str(alg_auc_out), 
                      'Diagnostic err:  ' + str(alg_acc_diag) + ' and AUC: ' + str(alg_auc_diag))

            statauc.append((alg_acc_unl, alg_auc_unl, alg_acc_out, alg_auc_out, alg_acc_diag, alg_auc_diag))
            
        lab_particles = np.array(lab_particles)
        unl_particles = np.array(unl_particles)
        out_particles = np.array(out_particles)
        """ We must return all particles and also margs, because margs is over all data and not just the Monte Carloed stuff. """
        return (statauc, (lab_particles, unl_particles, out_particles))
    

    def serialize (self, filename, compress_factor=4):
        joblib.dump(self, filename, compress=compress_factor)
        
    def featurize (self, data, mode='topk', k=0):
        """Implements a featurize(data) function using the existing ensemble list. """
        featlist = []
        for treefeat in self.enslist:
            f, _ = treefeat.featurize(data, mode=mode)
            featlist.append(f)
        return csr_matrix(sp.sparse.hstack(featlist))
    
    
    def predict (self, data, mode='topk', k=0):
        f = self.featurize(data, mode=mode, k=k)
        margs = f.dot(self.sigma)
        return np.clip(margs, -1, 1)
    
    def AUC (self, dataset, data_labels):
        return sklearn.metrics.roc_auc_score(data_labels, self.predict(dataset))
    
    
    def add_composite_feature (self, newtree, total_correction=False, correction_duration=10, stepsize=None, 
                          failure_prob=0.01, numsamp_min=0, mode='topk', k=0, tol=0.00005):
        # y_transform is false because decision trees actually make the right predictions when trained directly and not through the RF class.
        treefeat = composite_feature.composite_feature(newtree, self.holdout_set, self.holdout_labels, failure_prob=failure_prob, k=k, 
                                             numsamp_min=numsamp_min, ytransform=False)
        # Update the current featurization
        xtra_out, newb = treefeat.featurize(self.holdout_set, mode=mode)
        xtra_unl, _ = treefeat.featurize(self.unlabeled_set, mode=mode)
        xtra_outdiag, _ = treefeat.featurize(self.outdiag_set, mode=mode)
        xtra_lab, _ = treefeat.featurize(self.labeled_set, mode=mode)
         
        self.labeled_set_feats.append(xtra_lab)
        self.unlabeled_set_feats.append(xtra_unl)
        self.holdout_set_feats.append(xtra_out)
        self.outdiag_set_feats.append(xtra_outdiag)
        
        feats_lab = csr_matrix(sp.sparse.hstack(self.labeled_set_feats))
        feats_unl = csr_matrix(sp.sparse.hstack(self.unlabeled_set_feats))
        feats_out = csr_matrix(sp.sparse.hstack(self.holdout_set_feats))
        feats_diag = csr_matrix(sp.sparse.hstack(self.outdiag_set_feats))
        
        self.enslist.append(treefeat)
        wts = np.append(self.sigma, [0]*len(newb))
        
        #print 'Projected slack function evaluation:  ' + str(self.projslackfunc(newb, xtra_unl)(1))
        # Now update sigma.
        if not total_correction:
            max_stepsize = 2*stepsize
            stepsize = SSutils.golden_section_search(self.projslackfunc(newb, xtra_unl), 0, max_stepsize, tol=tol)
            print stepsize, max_stepsize
            new_wts = stepsize*np.ones(len(newb))
            self.sigma = np.append(self.sigma, new_wts)
            self.b = np.append(self.b, newb)
        else:
            self.b = np.append(self.b, newb)
            gradh = hedgeclipper.Hedgeclipper(feats_unl, self.unlabeled_labels, feats_out, self.holdout_labels, 
                outdiag_set=feats_diag, outdiag_labels=self.outdiag_labels, b_vector=self.b, sigma=wts)
            statauc = gradh.sgd(correction_duration, verbose=False)
            self.sigma = gradh.sigma
        
        self.margslab = feats_lab.dot(self.sigma)
        self.margsunl = feats_unl.dot(self.sigma)
        self.margsout = feats_out.dot(self.sigma)
        self.margdiag = feats_diag.dot(self.sigma)
    

    # Here new_preds_unl should be a (# examples) x (# classifiers) matrix, sparse because the classifiers could be specialists.
    def projslackfunc(self, new_b_vals, new_preds_unl):
        def toret(alpha):
            margs = self.margsunl + alpha*new_preds_unl.dot(np.ones(new_preds_unl.shape[1]))
            return -np.dot(self.b, self.sigma) - alpha*np.sum(new_b_vals) + np.mean(np.maximum(np.abs(margs), 1.0)) 
        return toret
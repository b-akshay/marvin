import numpy as np
import scipy as sp
import math
import scipy.stats
from scipy.sparse import csr_matrix
import time
import random
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools
from sklearn.cross_validation import train_test_split




"""
This class has several todos:
- Function to munge text into indicator features for use by the algorithm (needs to be implemented sparsely if possible). 
- Function to munge libSVM data (most text datasets)
"""


def accuracy_calc (y_true, y_pred, sample_weight=None):
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)/np.sum(sample_weight)
        return 0.5 + 0.5*np.dot(np.array(np.multiply(y_pred, sample_weight)), y_true)
    else:
        return 0.5 + (0.5/len(y_true))*np.dot(y_true, y_pred)

def logloss_calc (y_true, y_pred, sample_weight=None, tolerance=10**-15):
    numpts = len(y_true)
    y_pred = np.clip(y_pred, tolerance, 1 - tolerance)
    plus_true = np.ones(numpts) + np.array(y_true)
    minus_true = np.ones(numpts) - np.array(y_true)
    plus_pred = np.log(2.0/np.max(1.0 + y_pred))
    minus_pred = np.log(2.0/np.max(1.0 - y_pred))
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)/np.sum(sample_weight)
    else:
        sample_weight = (1.0/numpts)*np.ones(numpts)
    plus_contrib = 0.5*np.dot(np.array(np.multiply(plus_pred, sample_weight)), plus_true)
    minus_contrib = 0.5*np.dot(np.array(np.multiply(minus_pred, sample_weight)), minus_true)
    return plus_contrib + minus_contrib


"""
Golden section search to find the minimum of f in [left,right]. Follows Kiefer (1953) "Sequential minimax search for a maximum", 
but memoizes intermediate function values, as evaluation of f is expensive.
"""
def golden_section_search(f, left, right, tol=0.00001):
    phi = (math.sqrt(5)-1)/2       # golden ratio
    c = right - phi*(right - left)
    d = left + phi*(right - left)
    f_memoized_c = f(c)
    f_memoized_d = f(d)
    while abs(c-d) > tol:
        if f_memoized_c < f_memoized_d:
            right = d
            d = c
            c = right - phi*(right - left)
            f_memoized_d = f_memoized_c
            f_memoized_c = f(c)
        else:
            left = c
            c = d
            d = left + phi*(right - left)
            f_memoized_c = f_memoized_d
            f_memoized_d = f(d)
    return (left + right)*0.5


"""
Assumes plugin_est >= 0, and numsamps is an array of the number of samples (from the same validation set used to calculate plugin_est) 
falling into each specialist. 
This is used in lieu of the rho_i distribution, which is always uniform over a subset of the data.

Returns 0 if the feature has no weight on it. 
So removing zero-b features removes (a) zero-weight features, (b) those which are not good enough, 
and (c) those which have too few samples to sufficiently narrow the confidence interval.
"""
def calc_b_wilson(plugin_est, numsamps, failure_prob=0.01):
    err_est = 0.5*(1.0 - plugin_est)
    z = scipy.stats.norm.ppf(0.5*(2-failure_prob))
    zsq_n = z*z/numsamps
    sleeping_specs = np.isinf(zsq_n)
    recentered_mean_errors = np.multiply(1.0/(1+zsq_n), (err_est + 0.5*zsq_n))
    recentered_plugins = 1.0 - 2*recentered_mean_errors
    stddevs = np.sqrt(np.multiply(zsq_n, np.multiply(err_est, 1.0 - err_est)) + np.square(0.5*zsq_n))
    stddevs = np.multiply(1.0/(1+zsq_n), stddevs)
    toret = np.maximum(recentered_plugins - 2*stddevs, 0.0)
    toret[sleeping_specs] = 0.0
    return toret




"""
===========================================================================================
GENERATING PLOTS, RESULTS, AND DIAGNOSTICS
===========================================================================================
"""

def diagnostic_margin_info (margdiag, true_labels, numbins=0):
    hedged_ndces = np.where(np.abs(margdiag) < 1)[0]
    clipped_ndces = np.where(np.abs(margdiag) >= 1)[0]
    preds_out = np.clip(margdiag, -1, 1)
    mv_preds_out = np.sign(margdiag)
    if (len(hedged_ndces) > 0 and len(clipped_ndces) > 0):
        print('Fraction hedged', 1.0*len(hedged_ndces)/len(margdiag))
        print('Accuracy on {hedged, clipped}', 
              accuracy_calc (true_labels[hedged_ndces], preds_out[hedged_ndces]), 
              accuracy_calc (true_labels[clipped_ndces], preds_out[clipped_ndces]))
        print('Accuracy/AUC of vote', accuracy_calc (true_labels, mv_preds_out), 
              sklearn.metrics.roc_auc_score (true_labels, mv_preds_out))
    #hist_arr, bins, _ = plt.hist(margdiag, numbins)
    #binndxs = np.digitize(margdiag, bins)
    pluses = np.where(np.sign(true_labels) == 1.0)[0]
    minuses = np.where(np.sign(true_labels) == -1.0)[0]
    print(len(pluses), len(minuses))

    if numbins == 0:
        numbins = np.clip(0.05*len(margdiag), 25, 100)
    plt.hist(margdiag[pluses], numbins, alpha = 0.5, label='+1', color='green')
    plt.hist(margdiag[minuses], numbins, alpha = 0.5, label='-1', color='red')
    print(len([m for i,m in enumerate(margdiag) if ((m > 0) and (true_labels[i] == 1.0))]))
    return plt


def plot_weights (datafeats):
    a = csr_matrix(datafeats.max(axis=0).transpose()).toarray().flatten()
    plt.hist(a, np.clip(0.2*len(a), 25, 100))
    return plt


# Plot avg true label in bins, compare to CRP
def cumulabel_plot (preds, numbins=0):
    if numbins == 0:
        numbins = max(25, 0.05*len(preds))
    #plusndxs = np.where(y_test == 1)[0]
    #minusndxs = np.where(y_test == 0)[0]
    labfreqs = np.array([np.mean(preds[np.where(binndxs == i)[0]]) for i in range(1,1+numbins)])
    #labfreqsplus = 2*(labelFreqs (margs, y_test, numbins)) - 1
    #labfreqsminus = 2*(labelFreqs (margs, y_test, numbins)) - 1
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(bins, labfreqs, 'r-', linewidth=2)
    #ax1.plot(labbins, labbins, 'g--')
    plt.title('Approximate average label vs. averaged ensemble prediction')
    return plt




"""
===========================================================================================
Reading Data 
===========================================================================================
"""

def mbatGen(fileiter, mbatsize):
    for i, lines in enumerate(itertools.islice(fileiter, mbatsize)): #add filtering condition after this line, if any
        yield lines


def init_data (labeled_file, LABELED_SET_SIZE, UNLABEL_SET_SIZE, HOLDOUT_SET_SIZE, HOLDOUT_SET2_SIZE):
    with open(labeled_file,'r') as f:
        data = np.genfromtxt(mbatGen(f, LABELED_SET_SIZE+UNLABEL_SET_SIZE+HOLDOUT_SET_SIZE+HOLDOUT_SET2_SIZE), 
                             delimiter=',', dtype='f8')
    #with open(unlab_file,'r') as f:
    #    tstdata = np.genfromtxt(mbatGen(f, UNLABEL_SET_SIZE), delimiter=',', dtype='f8')
    #    outdata = np.genfromtxt(mbatGen(f, HOLDOUT_SET_SIZE), delimiter=',', dtype='f8')

    y_all = np.array([x[0] for x in data])
    x_all = np.array([x[1:] for x in data])
    
    xtrte, x_outall, ytrte, y_outall = train_test_split(x_all, y_all, test_size=HOLDOUT_SET_SIZE+HOLDOUT_SET2_SIZE, 
                                              random_state=42)
    x_out, x_out2, y_out, y_out2 = train_test_split(x_outall, y_outall, test_size=HOLDOUT_SET2_SIZE)
    x_train, x_unl, y_train, y_unl = train_test_split(xtrte, ytrte, test_size=UNLABEL_SET_SIZE, random_state=42)
    return (x_train, y_train, x_unl, y_unl, x_out, y_out, x_out2, y_out2)



def shuf_data (labeled_file, total_size, target_file):
    buf = samp_file_to_arr (labeled_file, total_size)
    with open( target_file, 'wb' ) as fo:
        for item in buf:
            fo.write(item)


"""
The following function samples total_size rows uniformly at random from the file labeled_file, 
and returns a numpy matrix with just these rows. 
It does this in one fast pass through the file.
"""
def samp_file_to_arr (labeled_file, total_size):
    buf = []
    n = 0
    with open( labeled_file, 'rb' ) as fi:
        for _, line in enumerate(fi):
            n = n + 1
            r = random.random()
            if n <= total_size:
                buf.append(line)
            elif r < total_size/n:
                loc = random.randint(0, total_size-1)
                buf[loc] = line
    return np.array([np.fromstring(s, sep=',', dtype='f8') for s in buf])


# Input: a generator of LibSVM file lines (a minibatch)
# Output: a CSR matrix of those lines, featurized.
def libsvm_to_sparse(filegen, numfeat):
    if numfeat == 0:
        return ()
    label = 0
    row_number = 0
    labs = []
    rows = []
    cols = []
    data = []
    reader = csv.reader( filegen, delimiter = ' ' )
    for line in reader:
        label = ytrans(int(line.pop( 0 )))
        labs += [label]
        if line[-1].strip() == '':
            line.pop( -1 )
        line = map( lambda x: tuple( x.split( ":" )), line )
        ndces, vals = zip(*line)
        ndces = [int(x)-1 for x in list(ndces)]              # Convert from string
        vals = map(float, list(vals))
        #print row_number, ndces[0:100], vals[0:100]
        rows += [row_number] * len(ndces)          # Augment CSR representation
        cols += ndces
        data += vals
        row_number += 1
    mat = csr_matrix((data, (rows, cols)), shape=(row_number, numfeat))
    return (mat, labs)


# Returns minibatches of size minibatch_size, given a file object for the file of interest, and the total dimension d.
def sparse_iter_minibatches(fileiter, minibatch_size, d):
    datambat = libsvm_to_sparse(mbatGen(fileiter, minibatch_size), d)
    while len(datambat):
        yield datambat
        datambat = libsvm_to_sparse(mbatGen(fileiter, minibatch_size), d)



"""
===========================================================================================
Benchmark classifiers
===========================================================================================
"""

def init_base_classifiers(x_train, y_train, num_iters=100):
    skcl = []
    inittime = time.time()
    clrf = RandomForestClassifier(n_estimators=num_iters, n_jobs=-1)
    skcl.append(('Plain RF', clrf))
    clgbdt = AdaBoostClassifier(n_estimators=num_iters, algorithm='SAMME')
    skcl.append(('GBDT', clgbdt))
    cldt = DecisionTreeClassifier()
    skcl.append(('DT', cldt))
    cletf = GradientBoostingClassifier(n_estimators=num_iters, loss='exponential')
    skcl.append(('MART', cletf))
    cllogistic = LogisticRegression()#(loss='log')
    skcl.append(('Logistic regression', cllogistic))
    clsvm = svm.LinearSVC()
    skcl.append(clsvm)

    # Now x_train is a (LABELED_SET_SIZE x d) matrix, and y_train a vector of size LABELED_SET_SIZE.        
    for i in range(len(skcl)):
        skcl[i][1].fit(x_train, y_train)
        print(skcl[i][0] + ' trained', time.time() - inittime)
        
    return skcl
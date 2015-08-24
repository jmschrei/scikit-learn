# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# Licence: BSD 3 clause

from libc.stdlib cimport calloc, free, realloc, qsort

from libc.string cimport memcpy, memset
from libc.math cimport log as ln
from cpython cimport Py_INCREF, PyObject

from cython.parallel import prange
from joblib import Parallel, delayed

import time
import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse, csc_matrix, csr_matrix

from sklearn.tree._utils cimport Stack, StackRecord
from sklearn.tree._utils cimport PriorityHeap, SplitRecord



cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

cdef DTYPE_t MIN_IMPURITY_SPLIT = 1e-7

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})


# =============================================================================
# Criterion
# =============================================================================

cdef class Criterion:
    """Interface for impurity criteria. 
    
    This object stores methods on how to calculate how good a split is using 
    different metrics.
    """

    cdef void init(self, DTYPE_t* X, SIZE_t X_sample_stride, 
        SIZE_t X_feature_stride, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* w,
        SIZE_t n_samples, SIZE_t min_leaf_samples, DOUBLE_t min_leaf_weight,
        DOUBLE_t* w_sum, DOUBLE_t* yw_sq_sum, DOUBLE_t** node_value):
        """Initialize by passing pointers to the underlying data.

        Parameters
        ----------
        X: array-like, dtype=DTYPE_t
            A 1-d buffer of the data matrix, accessible using strides
        X_sample_stride: SIZE_t
            The number of positions in the buffer to move to access the same
            feature of the next sample (move one row down)
        X_feature_stride: SIZE_t
            The number of positions in the buffer to move to access the next
            feature of the same sample (move one column over)
        y: array-like, dtype=DOUBLE_t
            A 1-d buffer of targets for each sample.
        y_stride: SIZE_t
            The number of positions in the buffer to move over to access the
            next sample.
        w: array-like, dtype=DOUBLE_t
            The weight of each sample
        n_samples: SIZE_t
            The number of data points in the dataset
        min_leaf_samples: SIZE_t
            A constraint on the minimum number of samples a leaf must have
        min_leaf_weight: DOUBLE_t
            A constraint on the minimal total sample weight a leaf must have 
        """

        self.X = X
        self.y = y
        self.w = w

        self.min_leaf_samples = min_leaf_samples
        self.min_leaf_weight = min_leaf_weight
        self.n_samples = n_samples

        self.X_sample_stride = X_sample_stride
        self.X_feature_stride = X_feature_stride
        self.y_stride = y_stride

    cdef SplitRecord best_split(self, SIZE_t* samples, SIZE_t start, 
        SIZE_t end, SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum,
        DOUBLE_t* node_value) nogil:
        """Find the best split in samples[start:end] on a specific feature.

        Parameters
        ----------
        samples: array-like, dtype=SIZEt
            The sorted indices of samples to consider for this split
        start: SIZE_t
            The index of the first sample to consider
        end: SIZE_t
            The index of the last sample to consider
        feature: SIZE_t
            The feature to be considered for a split
        """

        pass

    cdef SplitRecord random_split(self, SIZE_t* samples, SIZE_t start, 
        SIZE_t end, SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum,
        DOUBLE_t* node_value, UINT32_t* rand_r) nogil:

        pass

cdef class ClassificationCriterion(Criterion):
    """
    This is a criterion with methods specifically used for classification.
    """
    
    cdef SIZE_t [:] n_classes
    cdef SIZE_t n
    cdef DOUBLE_t* yw_cr
    cdef DOUBLE_t* yw_cl

    def __cinit__(self, SIZE_t n_outputs, np.ndarray[SIZE_t, ndim=1] n_classes, 
        SIZE_t n_jobs):
        """
        Initialize attributes for this classifier, automatically calling the
        parent __cinit__ method as well.

        Parameters
        ----------
        n_outputs: int64
            The number of responses, the dimensionality of the prediction
        n_classes: numpy.ndarray, dtype=int64
            The number of unique classes in each response
        """

        cdef SIZE_t i

        self.n_outputs = n_outputs
        self.n_classes = n_classes
        self.n_jobs = n_jobs

        self.X = NULL
        self.y = NULL
        self.w = NULL

        self.X_sample_stride = 1
        self.X_feature_stride = 1
        self.y_stride = 1

        self.min_leaf_samples = 0
        self.min_leaf_weight = 0

        self.yw_cl = NULL
        self.yw_cr = NULL

        self.n = 0
        for i in range(n_outputs):
            self.n += n_classes[i] 

    cdef void init(self, DTYPE_t* X, SIZE_t X_sample_stride, 
        SIZE_t X_feature_stride, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* w,
        SIZE_t n_samples, SIZE_t min_leaf_samples, DOUBLE_t min_leaf_weight,
        DOUBLE_t* w_sum, DOUBLE_t* yw_sq_sum, DOUBLE_t** node_value):
        """Initialize by passing pointers to the underlying data."""

        Criterion.init(self, X, X_sample_stride, X_feature_stride, y, 
            y_stride, w, n_samples, min_leaf_samples, min_leaf_weight,
            w_sum, yw_sq_sum, node_value)

        self.yw_cl = <DOUBLE_t*> calloc(self.n, sizeof(DOUBLE_t))
        self.yw_cr = <DOUBLE_t*> calloc(self.n, sizeof(DOUBLE_t))
        memset(self.yw_cl, 0, self.n*sizeof(DOUBLE_t))
        memset(self.yw_cr, 0, self.n*sizeof(DOUBLE_t))

        cdef SIZE_t i, label

        for i in range(n_samples):
            label = <SIZE_t>y[i]
            w_sum[0] += w[i]
            self.yw_cr[label] += w[i]

        node_value[0] = <DOUBLE_t*> calloc(self.n, sizeof(DOUBLE_t))
        for i in range(self.n):
            node_value[0][i] = self.yw_cr[i]

        yw_sq_sum[0] = 0

    def __dealloc__(self):
        free(self.yw_cl)
        free(self.yw_cr)

    def __reduce__(self):
        return (ClassificationCriterion,
                (self.n_outputs, self.n_classes, self.n_jobs),
                self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

cdef class Entropy(ClassificationCriterion):
    """
    A class representing the Cross Entropy impurity criteria. This handles
    cases where the response is a classification taking values 0, 1, ...
    K-2, K-1. If node m represents a region Rm with Nm observations, then let

        pmk = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} pmk log(pmk)
    """

    cdef SplitRecord best_split(self, SIZE_t* samples, SIZE_t start, 
        SIZE_t end, SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum,
        DOUBLE_t* node_value) nogil:
        
        cdef DTYPE_t* X = self.X
        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* w = self.w

        cdef SIZE_t y_stride = self.y_stride
        cdef SIZE_t upper, lower

        cdef DOUBLE_t w_cl, w_cr
        cdef DOUBLE_t* yw_cl = self.yw_cl
        cdef DOUBLE_t* yw_cr = self.yw_cr
        memset(yw_cl, 0, self.n*sizeof(DOUBLE_t))
        memset(yw_cr, 0, self.n*sizeof(DOUBLE_t))

        cdef SIZE_t feature_offset = feature*self.X_feature_stride
        cdef SIZE_t label

        cdef SIZE_t i, j, p, n = end-start, m = self.n

        cdef DOUBLE_t impurity_left, impurity_right, improvement
        cdef SplitRecord split
        _init_split_record(&split)
        split.node_value = node_value
        split.node_value_left = <DOUBLE_t*> calloc(m, sizeof(DOUBLE_t))
        split.node_value_right = <DOUBLE_t*> calloc(m, sizeof(DOUBLE_t))

        # Get sufficient statistics for the impurity improvement and children
        # impurity calculations and cache them for all possible splits
        split.impurity = 0
        for i in range(m):
            yw_cr[i] = node_value[i]
            if node_value[i] > 0:
                split.impurity -= (node_value[i] / w_sum * 
                    log(node_value[i] / w_sum))

        w_cr = w_sum
        w_cl = 0
        # Find the best split by scanning the entire range and calculating
        # improvement for each split point.
        for i in range(self.min_leaf_samples-1, n-self.min_leaf_samples):
            p = samples[start+i]
            label = <SIZE_t>y[p]

            yw_cr[label] -= w[p]
            yw_cl[label] += w[p]  

            w_cr -= w[p]
            w_cl += w[p]

            upper = samples[start+i+1]*self.X_sample_stride + feature_offset
            lower = samples[start+i]*self.X_sample_stride + feature_offset

            if start+i+1 < end and X[upper] <= X[lower] + FEATURE_THRESHOLD:
                continue

            if w_cl < self.min_leaf_weight or w_cr < self.min_leaf_weight:
                continue

            impurity_left = 0
            impurity_right = 0
            for j in range(m):
                if yw_cl[j] > 0:
                    impurity_left -= yw_cl[j] * log(yw_cl[j] / w_cl)
                if yw_cr[j] > 0:
                    impurity_right -= yw_cr[j] * log(yw_cr[j] / w_cr)

            improvement = -impurity_left - impurity_right

            if improvement > split.improvement:
                split.improvement = improvement
                split.threshold = (X[upper] + X[lower]) / 2.0
                if split.threshold == X[upper]:
                    split.threshold = X[lower]
                split.pos = i+1

                for j in range(m):
                    split.node_value_left[j] = yw_cl[j]
                    split.node_value_right[j] = yw_cr[j]

                split.weight = w_sum
                split.weight_left = w_cl
                split.weight_right = w_cr

                split.impurity_left = impurity_left / w_cl
                split.impurity_right = impurity_right / w_cr

        split.improvement = (( 1.0 * n / self.n_samples ) * 
            split.improvement - 
            split.weight_left / n * split.impurity_left -
            split.weight_right / n * split.impurity_right )

        split.impurity /= self.n_outputs
        split.impurity_left /= self.n_outputs
        split.impurity_right /= self.n_outputs

        split.feature = feature
        if split.pos == -1:
            split.pos = end
        else:
            split.pos += start
        return split

    cdef SplitRecord random_split(self, SIZE_t* samples, SIZE_t start, 
        SIZE_t end, SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum,
        DOUBLE_t* node_value, UINT32_t* rand_r) nogil:
        """
        Random split.
        """
        
        cdef DTYPE_t* X = self.X
        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* w = self.w

        cdef SIZE_t y_stride = self.y_stride
        cdef DTYPE_t upper, lower, value

        cdef DOUBLE_t w_cl, w_cr
        cdef DOUBLE_t* yw_cl = self.yw_cl
        cdef DOUBLE_t* yw_cr = self.yw_cr
        memset(yw_cl, 0, self.n*sizeof(DOUBLE_t))
        memset(yw_cr, 0, self.n*sizeof(DOUBLE_t))

        cdef SIZE_t feature_offset = feature*self.X_feature_stride
        cdef SIZE_t label

        cdef SIZE_t i, p, n = end-start, m = self.n

        cdef DOUBLE_t improvement
        cdef SplitRecord split
        _init_split_record(&split)
        split.node_value = node_value
        split.node_value_left = <DOUBLE_t*> calloc(m, sizeof(DOUBLE_t))
        split.node_value_right = <DOUBLE_t*> calloc(m, sizeof(DOUBLE_t))

        split.impurity = 0
        for i in range(m):
            yw_cr[i] = node_value[i] 
            if yw_cr[i] > 0:
                split.impurity -= yw_cr[i] / w_sum * log(yw_cr[i] / w_sum)

        w_cr = w_sum
        w_cl = 0
        # Find the best split by scanning the entire range and calculating
        # improvement for each split point.
        upper = X[samples[start+n-1]*self.X_sample_stride + feature_offset]
        lower = X[samples[start]*self.X_sample_stride + feature_offset]

        split.threshold = rand_uniform(lower, upper, rand_r)
        split.impurity_left = 0
        split.impurity_right = 0

        for i in range(n-1):
            p = samples[start+i]
            value = X[p*self.X_sample_stride + feature_offset]

            if value > split.threshold:
                split.pos = i+start
                break

            label = <SIZE_t>y[p]
            yw_cr[label] -= w[p]
            yw_cl[label] += w[p]  
            w_cr -= w[p]
            w_cl += w[p]

        for i in range(m):
            if yw_cl[i] > 0:
                split.impurity_left -= yw_cl[i] / w_cl * log(yw_cl[i] / w_cl)
            if yw_cr[i] > 0:
                split.impurity_right -= yw_cr[i] / w_cr * log(yw_cr[i] / w_cr)

        split.improvement = (-w_cl * split.impurity_left - 
            w_cr * split.impurity_right)

        for i in range(m):
            split.node_value_left[i] = yw_cl[i]
            split.node_value_right[i] = yw_cr[i]

        split.weight = w_sum
        split.weight_left = w_cl
        split.weight_right = w_cr

        split.improvement = (( 1.0 * n / self.n_samples ) * 
            split.improvement - 
            split.weight_left / n * split.impurity_left -
            split.weight_right / n * split.impurity_right )

        split.impurity /= self.n_outputs
        split.impurity_left /= self.n_outputs
        split.impurity_right /= self.n_outputs

        split.feature = feature
        return split

cdef class Gini(ClassificationCriterion):
    """
    A class representing the Gini Index impurity criteria. This handles
    cases where the response is a classification taking values 0, 1, ...
    K-2, K-1. If node m represents a region Rm with Nm observations, then let

        pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} pmk (1 - pmk)
              = 1 - \sum_{k=0}^{K-1} pmk ** 2
    """
    cdef SplitRecord best_split(self, SIZE_t* samples, SIZE_t start, 
        SIZE_t end, SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum,
        DOUBLE_t* node_value) nogil:

        cdef DTYPE_t* X = self.X
        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* w = self.w

        cdef SIZE_t y_stride = self.y_stride
        cdef SIZE_t upper, lower

        cdef DOUBLE_t w_cl, w_cr
        cdef DOUBLE_t* yw_cl = self.yw_cl
        cdef DOUBLE_t* yw_cr = self.yw_cr
        memset(yw_cl, 0, self.n*sizeof(DOUBLE_t))
        memset(yw_cr, 0, self.n*sizeof(DOUBLE_t))

        cdef SIZE_t feature_offset = feature*self.X_feature_stride
        cdef SIZE_t label

        cdef SIZE_t i, j, p, n = end-start, m = self.n

        cdef DOUBLE_t impurity_left, impurity_right, improvement
        cdef SplitRecord split
        _init_split_record(&split)
        split.node_value = node_value
        split.node_value_left = <DOUBLE_t*> calloc(m, sizeof(DOUBLE_t))
        split.node_value_right = <DOUBLE_t*> calloc(m, sizeof(DOUBLE_t))

        split.impurity = 0
        for i in range(m):
            yw_cr[i] = node_value[i]
            split.impurity += yw_cr[i] ** 2

        split.impurity = 1.0 - split.impurity / (w_sum ** 2.0) 

        w_cr = w_sum
        w_cl = 0
        # Find the best split by scanning the entire range and calculating
        # improvement for each split point.
        for i in range(self.min_leaf_samples-1, n-self.min_leaf_samples):
            p = samples[start+i]
            label = <SIZE_t>y[p]

            yw_cr[label] -= w[p]
            yw_cl[label] += w[p]  

            w_cr -= w[p]
            w_cl += w[p]

            upper = samples[start+i+1]*self.X_sample_stride + feature_offset
            lower = samples[start+i]*self.X_sample_stride + feature_offset

            if start+i+1 < end and X[upper] <= X[lower] + FEATURE_THRESHOLD:
                continue

            if w_cl < self.min_leaf_weight or w_cr < self.min_leaf_weight:
                continue

            impurity_left = 0
            impurity_right = 0
            for j in range(m):
                impurity_left += yw_cl[j] ** 2.0
                impurity_right += yw_cr[j] ** 2.0

            improvement = impurity_left / w_cl + impurity_right / w_cr

            if improvement > split.improvement:
                split.improvement = improvement
                split.threshold = (X[upper] + X[lower]) / 2.0
                if split.threshold == X[upper]:
                    split.threshold = X[lower]
                split.pos = i+1

                for j in range(m):
                    split.node_value_left[j] = yw_cl[j]
                    split.node_value_right[j] = yw_cr[j]

                split.weight = w_sum
                split.weight_left = w_cl
                split.weight_right = w_cr

                split.impurity_left = 1.-impurity_left/split.weight_left**2
                split.impurity_right = 1.-impurity_right/split.weight_right**2

        split.improvement = (( 1.0 * n / self.n_samples ) * 
            split.improvement - 
            split.weight_left / n * split.impurity_left -
            split.weight_right / n * split.impurity_right )

        split.impurity /= self.n_outputs
        split.impurity_left /= self.n_outputs
        split.impurity_right /= self.n_outputs

        split.feature = feature
        if split.pos == -1:
            split.pos = end
        else:
            split.pos += start
        return split

    cdef SplitRecord random_split(self, SIZE_t* samples, SIZE_t start, 
        SIZE_t end, SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum,
        DOUBLE_t* node_value, UINT32_t* rand_r) nogil:
        """
        Random split.
        """
        
        cdef DTYPE_t* X = self.X
        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* w = self.w

        cdef SIZE_t y_stride = self.y_stride
        cdef DTYPE_t upper, lower, value

        cdef DOUBLE_t w_cl, w_cr
        cdef DOUBLE_t* yw_cl = self.yw_cl
        cdef DOUBLE_t* yw_cr = self.yw_cr
        memset(yw_cl, 0, self.n*sizeof(DOUBLE_t))
        memset(yw_cr, 0, self.n*sizeof(DOUBLE_t))

        cdef SIZE_t feature_offset = feature*self.X_feature_stride
        cdef SIZE_t label

        cdef SIZE_t i, p, n = end-start, m = self.n

        cdef DOUBLE_t impurity_left, impurity_right, improvement
        cdef SplitRecord split
        _init_split_record(&split)
        split.node_value = node_value
        split.node_value_left = <DOUBLE_t*> calloc(m, sizeof(DOUBLE_t))
        split.node_value_right = <DOUBLE_t*> calloc(m, sizeof(DOUBLE_t))

        split.impurity = 0
        for i in range(m):
            yw_cr[i] = node_value[i]
            split.impurity += yw_cr[i] ** 2

        split.impurity = 1.0 - split.impurity / (w_sum ** 2.0)

        w_cr = w_sum
        w_cl = 0
        # Find the best split by scanning the entire range and calculating
        # improvement for each split point.
        upper = X[samples[start+n-1]*self.X_sample_stride + feature_offset]
        lower = X[samples[start]*self.X_sample_stride + feature_offset]

        split.threshold = rand_uniform(lower, upper, rand_r)
        split.impurity_left = 0
        split.impurity_right = 0

        for i in range(n):
            p = samples[start+i]
            value = X[p*self.X_sample_stride + feature_offset]

            if value > split.threshold:
                split.pos = i+start
                break

            label = <SIZE_t>y[p]
            yw_cr[label] -= w[p]
            yw_cl[label] += w[p]  
            w_cr -= w[p]
            w_cl += w[p]

        for i in range(m):
            split.impurity_left += yw_cl[i] ** 2.0
            split.impurity_right += yw_cr[i] ** 2.0

        for i in range(m):
            split.node_value_left[i] = yw_cl[i]
            split.node_value_right[i] = yw_cr[i]

        split.weight = w_sum
        split.weight_left = w_cl
        split.weight_right = w_cr

        split.impurity_left = 1. - split.impurity_left/split.weight_left**2
        split.impurity_right = 1. - split.impurity_right/split.weight_right**2

        split.improvement = (( 1.0 * n / self.n_samples ) * 
            split.improvement - 
            split.weight_left / n * split.impurity_left -
            split.weight_right / n * split.impurity_right )

        split.impurity /= self.n_outputs
        split.impurity_left /= self.n_outputs
        split.impurity_right /= self.n_outputs

        split.feature = feature
        return split

cdef class RegressionCriterion(Criterion):
    """
    A class representing a regression criteria. This handles cases where the
    response is a continuous value, and is evaluated by computing the variance
    of the target values left and right of the split point. The computation
    takes linear time with `n_samples` by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2 
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_jobs):
        self.n_outputs = n_outputs
        self.n_jobs = n_jobs

        self.X = NULL
        self.y = NULL
        self.w = NULL

        self.X_sample_stride = 1
        self.X_feature_stride = 1
        self.y_stride = 1

        self.min_leaf_samples = 0
        self.min_leaf_weight = 0

    cdef void init(self, DTYPE_t* X, SIZE_t X_sample_stride, 
        SIZE_t X_feature_stride, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* w,
        SIZE_t n_samples, SIZE_t min_leaf_samples, DOUBLE_t min_leaf_weight,
        DOUBLE_t* w_sum, DOUBLE_t* yw_sq_sum, DOUBLE_t** node_value):
        """Initialize by passing pointers to the underlying data."""

        Criterion.init(self, X, X_sample_stride, X_feature_stride, y, 
            y_stride, w, n_samples, min_leaf_samples, min_leaf_weight,
            w_sum, yw_sq_sum, node_value)

        cdef SIZE_t i
        node_value[0] = <DOUBLE_t*>calloc(1, sizeof(DOUBLE_t))
        memset(node_value[0], 0, 1*sizeof(DOUBLE_t))

        for i in range(n_samples):
            w_sum[0] += w[i]
            node_value[0][0] += w[i] *y[i]
            yw_sq_sum[0] += w[i] * y[i] * y[i]

        node_value[0][0] /= w_sum[0]

    def __reduce__(self):
        return (RegressionCriterion, (self.n_outputs, self.n_jobs), self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """
    cdef SplitRecord best_split(self, SIZE_t* samples, SIZE_t start, 
        SIZE_t end, SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum,
        DOUBLE_t* node_value) nogil:

        cdef DTYPE_t* X = self.X
        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* w = self.w

        cdef SIZE_t y_stride = self.y_stride
        cdef SIZE_t upper, lower

        cdef SIZE_t feature_offset = feature*self.X_feature_stride

        cdef DOUBLE_t yw_sum = node_value[0] * w_sum
        cdef DOUBLE_t w_cl = 0, yw_cl = 0, yw_sq = 0
        cdef DOUBLE_t w_cr = w_sum, yw_cr = yw_sum, yw_sq_r = yw_sq_sum

        cdef int i, p, n = end-start

        cdef DOUBLE_t improvement
        cdef SplitRecord split
        _init_split_record(&split)
        split.node_value = node_value
        split.node_value_left = <DOUBLE_t*> calloc(1, sizeof(DOUBLE_t))
        split.node_value_right = <DOUBLE_t*> calloc(1, sizeof(DOUBLE_t))

        split.impurity = yw_sq_sum / w_sum - (yw_sum / w_sum) ** 2.0
        split.node_value[0] = node_value[0]

        w_cl = 0
        w_cr = w_sum

        for i in range(self.min_leaf_samples-1, n-self.min_leaf_samples):
            p = samples[start+i]

            w_cl += w[p]
            yw_cl += w[p] * y[p]
            yw_sq += w[p] * y[p] * y[p]

            w_cr -= w[p]
            yw_cr -= w[p] * y[p]
            yw_sq_r -= w[p] * y[p] * y[p]

            upper = samples[start+i+1]*self.X_sample_stride + feature_offset
            lower = samples[start+i]*self.X_sample_stride + feature_offset

            if start+i+1 < end and X[upper] <= X[lower] + FEATURE_THRESHOLD:
                continue

            if w_cl < self.min_leaf_weight or w_cr < self.min_leaf_weight:
                continue

            improvement = yw_cl**2.0 / w_cl + yw_cr**2.0 / w_cr

            if improvement > split.improvement:
                split.improvement = improvement
                split.threshold = (X[upper] + X[lower]) / 2.0
                split.pos = i+1

                split.weight = w_sum
                split.weight_left = w_cl
                split.weight_right = w_cr

                split.yw_sq_sum = yw_sq_sum
                split.yw_sq_sum_left = yw_sq
                split.yw_sq_sum_right = yw_sq_r
                
                split.impurity_left = yw_sq / w_cl - (yw_cl / w_cl) ** 2.0
                split.impurity_right =  yw_sq_r / w_cr - (yw_cr / w_cr) ** 2.0

                split.node_value_left[0] = yw_cl / w_cl
                split.node_value_right[0] = yw_cr / w_cr

        split.feature = feature

        split.improvement = (( 1.0 * n / self.n_samples ) * 
            split.improvement - 
            split.weight_left / n * split.impurity_left -
            split.weight_right / n * split.impurity_right )

        if split.pos == -1:
            split.pos = end
        else:
            split.pos += start

        return split

    cdef SplitRecord random_split(self, SIZE_t* samples, SIZE_t start, 
        SIZE_t end, SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum,
        DOUBLE_t* node_value, UINT32_t* rand_r) nogil:

        cdef DTYPE_t* X = self.X
        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* w = self.w

        cdef SIZE_t y_stride = self.y_stride
        cdef DTYPE_t upper, lower, value

        cdef DOUBLE_t yw_sum = node_value[0] * w_sum
        cdef DOUBLE_t w_cl = 0, yw_cl = 0, yw_sq = 0
        cdef DOUBLE_t w_cr = w_sum, yw_cr = yw_sum, yw_sq_r = yw_sq_sum

        cdef int i, p, n = end-start

        cdef DOUBLE_t impurity_left, impurity_right, improvement
        cdef SplitRecord split
        _init_split_record(&split)
        split.node_value = node_value
        split.node_value_left = <DOUBLE_t*> calloc(1, sizeof(DOUBLE_t))
        split.node_value_right = <DOUBLE_t*> calloc(1, sizeof(DOUBLE_t))

        cdef SIZE_t feature_offset = feature*self.X_feature_stride

        upper = X[samples[start+n-1]*self.X_sample_stride + feature_offset]
        lower = X[samples[start]*self.X_sample_stride + feature_offset]

        split.threshold = rand_uniform(lower, upper, rand_r)
        split.impurity = yw_sq_sum / w_sum - (yw_sum / w_sum) ** 2.0

        # Now find the best split using sufficient statistics
        for i in range(n-1):
            p = samples[start+i]
            value = X[p*self.X_sample_stride + feature_offset]

            if value > split.threshold:
                split.pos = i+start
                break

            w_cl += w[p]
            yw_cr += w[p] * y[p]
            yw_sq += w[p] * y[p] * y[p]

            w_cr -= w[p]
            yw_cr -= w[p] * y[p]
            yw_sq_r -= w[p] * y[p] * y[p]

        split.weight = w_sum 
        split.weight_left = w_cl
        split.weight_right = w_cr
        
        split.impurity_left = yw_sq / w_cl - (yw_cl / w_cl) ** 2.0
        split.impurity_right =  yw_sq_r / w_cr - (yw_cr / w_cr) ** 2.0

        split.improvement = (( 1.0 * n / self.n_samples ) * 
            split.improvement - 
            split.weight_left / n * split.impurity_left -
            split.weight_right / n * split.impurity_right )

        split.yw_sq_sum = yw_sq_sum
        split.yw_sq_sum_left = yw_sq
        split.yw_sq_sum_right = yw_sq_r

        split.node_value_left[0] = yw_cl / w_cl
        split.node_value_right[0] = yw_cr / w_cr 

        split.feature = feature

        return split


# =============================================================================
# Splitter
# =============================================================================

cdef inline void _init_split_record(SplitRecord* split) nogil:
    split.improvement = -INFINITY
    split.pos = -1
    split.threshold = -INFINITY
    split.feature = 0
    split.impurity = 0
    split.impurity_right = 0
    split.impurity_left = 0
    split.weight = 0
    split.weight_left = 0
    split.weight_right = 0
    split.yw_sq_sum = 0
    split.yw_sq_sum_left = 0
    split.yw_sq_sum_right = 0
    split.node_value = NULL
    split.node_value_left = NULL
    split.node_value_right = NULL

cdef class Splitter:
    """
    Interface for the splitter class. This is an object which handles efficient
    storage and splitting of a feature in the process of building a decision
    tree.
    """

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf, 
                  object random_state, SIZE_t n_jobs, SIZE_t best):

        self.criterion = criterion
        self.n_jobs = n_jobs

        self.samples = NULL
        self.n_samples = 0
        self.sample_weight = NULL
        self.sample_mask = NULL

        self.features = NULL
        self.n_features = 0

        self.y = NULL
        self.y_stride = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0

        self.X = NULL
        self.X_feature_stride = 1
        self.X_sample_stride = 1

        self.best = best

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.sample_mask)

        if self.presort:
            free(self.X_i)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __reduce(self):
        return (Splitter, (self.criterion,
                           self.max_features,
                           self.min_samples_leaf,
                           self.min_weight_leaf,
                           self.random_state), self.__getstate__() )

    cdef void init(self, object X, np.ndarray[INT32_t, ndim=2] X_idx_sorted,
                   bint presort, np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight, DOUBLE_t* w_sum,
                   DOUBLE_t* yw_sq_sum, DOUBLE_t** node_value,
                   SIZE_t* n_node_samples):
        """
        Initialize the values in this object.
        """

        self.presort = presort
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        
        self.n_samples = X.shape[0]
        safe_realloc(&self.samples, self.n_samples)

        cdef SIZE_t i, j = 0

        # In order to only use positively weighted samples, we must go through
        # each sample and check its associated weight.
        for i in range(self.n_samples):
            if sample_weight[i] != 0.0:
                self.samples[j] = i
                j += 1

        n_node_samples[0] = j

        self.n_features = X.shape[1]
        safe_realloc(&self.features, self.n_features)
        
        for i in range(self.n_features):
            self.features[i] = i

        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize
        
        self.sample_weight = sample_weight

        cdef np.ndarray X_ndarray = X
        self.X = <DTYPE_t*> X_ndarray.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_feature_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize

        if presort == 1:
            safe_realloc(&self.sample_mask, self.n_samples)
            memset(self.sample_mask, 0, self.n_samples*sizeof(SIZE_t))

            self.X_idx_sorted = X_idx_sorted
            self.X_idx_sorted_ptr = <INT32_t*> self.X_idx_sorted.data
            self.X_idx_sorted_stride = (<SIZE_t> self.X_idx_sorted.strides[1] /
                                           <SIZE_t> self.X_idx_sorted.itemsize)
        else:
            safe_realloc(&self.X_i, self.n_samples)

        self.criterion.init(self.X, self.X_sample_stride, 
            self.X_feature_stride, self.y, self.y_stride,
            self.sample_weight, self.n_samples,
            self.min_samples_leaf, self.min_weight_leaf,
            w_sum, yw_sq_sum, node_value)

    cdef SplitRecord _split(self, SIZE_t start, SIZE_t end, 
        SIZE_t feature, DOUBLE_t w_sum, DOUBLE_t yw_sq_sum, DOUBLE_t* node_value,
        SIZE_t best) nogil:
        """Find the best split for a specific feature.

        This is a helper for the best_split method to allow parallel
        computation of the best split, by scanning multiple features
        at the same time.
        """

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* X_i = self.X_i
        cdef INT32_t* X_idx_sorted = self.X_idx_sorted_ptr
        cdef SIZE_t* sample_mask = self.sample_mask
        cdef SIZE_t* samples = self.samples
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SIZE_t i, j, p = start
        cdef SIZE_t feature_offset = self.X_idx_sorted_stride * feature
        cdef SIZE_t argmin, argmax

        cdef SplitRecord split

        if self.presort == 1:
            for i in range(self.n_samples): 
                j = X_idx_sorted[i + feature_offset]
                if sample_mask[j] == 1:
                    samples[p] = j
                    p += 1
        else:
            for i in range(start, end):
                X_i[i] = X[self.X_sample_stride * samples[i] +
                           self.X_feature_stride * feature]

            sort(X_i + start, samples + start, end - start)


        feature_offset = feature*self.X_feature_stride

        argmax = samples[end-1]*self.X_sample_stride + feature_offset
        argmin = samples[start]*self.X_sample_stride + feature_offset

        if X[argmax] > X[argmin] + FEATURE_THRESHOLD:
            if best == 1:
                split = self.criterion.best_split(samples, start, end, 
                    feature, w_sum, yw_sq_sum, node_value)
            else:
                split = self.criterion.random_split(samples, start, end, 
                    feature, w_sum, yw_sq_sum, node_value, random_state)
        else:
            _init_split_record(&split)

        return split

    cdef SplitRecord split(self, SIZE_t start, SIZE_t end, DOUBLE_t w_sum, 
        DOUBLE_t yw_sq_sum, DOUBLE_t* node_value) nogil:
        """Find the best split for this node."""

        cdef SIZE_t* features = self.features
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t* sample_mask = self.sample_mask

        cdef DTYPE_t* X = self.X
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_feature_stride = self.X_feature_stride
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SIZE_t n_visited_features = 0
        cdef SIZE_t tmp, partition_end
        cdef SIZE_t i=0, j, p

        cdef SplitRecord best, current
        _init_split_record(&best)

        # Set a mask to indicate which samples we are considering.
        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 1

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm. To allow for parallelism,
        # we sample batches at a time, and count the number of
        # non-constant features, until we converge at max_features
        # number of non-constant features
        while n_visited_features < self.max_features and i < self.n_features:
            j = rand_int(i, self.n_features, random_state)
            features[i], features[j] = features[j], features[i]

            current = self._split(start, end, features[i], w_sum, yw_sq_sum, 
                node_value, self.best)

            i += 1
            if current.improvement > -INFINITY:
                n_visited_features += 1

            if current.improvement > best.improvement and best.pos < end:
                if best.node_value is not NULL:
                    free(best.node_value_left)
                    free(best.node_value_right)

                best = current
            else:
                free(current.node_value_left)
                free(current.node_value_right)

        best.start = start
        best.end = end
        if best.pos == -1:
            best.pos = end

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] + X_feature_stride 
                    * best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Reset sample mask
        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 0

        return best

# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)

# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples, SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r

cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples, SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b

cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind

cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1

# =============================================================================
# Tree builders
# =============================================================================
cdef class TreeBuilder:
    """Interface for different tree building strategies. """

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, SIZE_t max_leaf_nodes):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

    cpdef depth_first(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight, bint presort,
                np.ndarray X_idx_sorted):
        """Build a decision tree from the training set (X, y)."""
        X, y, sample_weight = self._check_input(X, y, sample_weight)
        cdef DOUBLE_t* sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split

        cdef SIZE_t start, end, depth, parent, node_id
        cdef bint is_left, is_leaf
        cdef DOUBLE_t* node_value = NULL
        cdef SplitRecord split

        cdef SIZE_t n_node_samples = 0
        cdef DOUBLE_t w_sum = 0, yw_sq_sum = 0
        splitter.init(X, X_idx_sorted, presort, y, sample_weight_ptr, &w_sum, 
            &yw_sq_sum, &node_value, &n_node_samples)

        cdef double threshold, impurity = INFINITY 
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        # push root node onto stack
        rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 
            w_sum, yw_sq_sum, node_value)
        if rc == -1:
            # got return code -1 - out-of-memory
            raise MemoryError()

        with nogil:
            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                w_sum = stack_record.weight
                yw_sq_sum = stack_record.yw_sq_sum
                impurity = stack_record.impurity
                node_value = stack_record.node_value

                n_node_samples = end - start

                is_leaf = ((depth >= max_depth) or
                           (n_node_samples < min_samples_split) or
                           (n_node_samples < 2 * min_samples_leaf) or
                           (w_sum <= 2 * min_weight_leaf) or 
                           (impurity <= MIN_IMPURITY_SPLIT))

                if not is_leaf:
                    split = splitter.split(start, end, w_sum, yw_sq_sum, node_value)
                    is_leaf = is_leaf or (split.pos >= end)
                    impurity = split.impurity
                    w_sum = split.weight
                    node_value = split.node_value

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         w_sum, node_value)

                if node_id == <SIZE_t>(-1):
                    rc = -1
                    break

                if not is_leaf:
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, split.weight_right,
                                    split.yw_sq_sum_right, split.node_value_right)
                    if rc == -1:
                        break

                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, split.weight_left,
                                    split.yw_sq_sum_left, split.node_value_left)
                    if rc == -1:
                        break

                free(node_value)

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

    cpdef best_first(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight, bint presort,
                np.ndarray X_idx_sorted):
        """Build a decision tree from the training set (X, y)."""

        X, y, sample_weight = self._check_input(X, y, sample_weight)
        cdef DOUBLE_t* sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_leaf_nodes = self.max_leaf_nodes
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split

        cdef SIZE_t n_node_samples
        cdef DOUBLE_t w_sum, yw_sq_sum
        cdef DOUBLE_t* node_value

        # Recursive partition (without actual recursion)
        splitter.init(X, X_idx_sorted, presort, y, sample_weight_ptr, &w_sum, 
            &yw_sq_sum, &node_value, &n_node_samples)

        cdef PriorityHeap frontier = PriorityHeap(INITIAL_STACK_SIZE)
        cdef SplitRecord parent, split

        cdef SIZE_t max_split_nodes = max_leaf_nodes - 1
        cdef bint is_leaf
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        # Initial capacity
        cdef SIZE_t init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize_c(init_capacity)

        with nogil:
            split = splitter.split(0, n_node_samples, w_sum, yw_sq_sum, 
                node_value)
            split.depth = 0
            split.parent = _TREE_UNDEFINED
            split.is_left = 0

            frontier.push(split)

            while not frontier.is_empty():
                frontier.pop(&parent)

                n_node_samples = parent.end - parent.start

                is_leaf = ((parent.depth >= self.max_depth) or
                           (n_node_samples < min_samples_split) or
                           (n_node_samples < 2 * min_samples_leaf) or
                           (parent.weight <= 2 * min_weight_leaf) or 
                           (parent.impurity <= MIN_IMPURITY_SPLIT) or
                           (max_split_nodes <= 0))

                node_id = tree._add_node(parent.parent, parent.is_left, 
                    is_leaf, parent.feature, parent.threshold, 
                    parent.impurity, n_node_samples, parent.weight,
                    parent.node_value) 

                if not is_leaf:
                    max_split_nodes -= 1

                    if ((parent.impurity_left < MIN_IMPURITY_SPLIT) or 
                        (max_split_nodes <=0)):
                        tree._add_node(node_id, 1, 1, 0, 0, 
                            parent.impurity_left, parent.pos-parent.start, 
                            parent.weight_left, parent.node_value_left)
                    else: 
                        split = splitter.split(parent.start, parent.pos, 
                            parent.weight_left, parent.yw_sq_sum_left, 
                            parent.node_value_left)

                        split.is_left = 1
                        split.parent = node_id
                        split.depth = parent.depth + 1

                        frontier.push(split)

                    if ((parent.impurity_right < MIN_IMPURITY_SPLIT) or 
                        (max_split_nodes <=0)):
                        tree._add_node(node_id, 0, 1, 0, 0, 
                            parent.impurity_right, parent.end-parent.pos, 
                            parent.weight_right, parent.node_value_right)
                    else: 
                        split = splitter.split(parent.pos, parent.end, 
                            parent.weight_right, parent.yw_sq_sum_right, 
                            parent.node_value_right)

                        split.is_left = 0
                        split.parent = node_id
                        split.depth = parent.depth + 1

                        frontier.push(split)

                    if split.depth > max_depth_seen:
                        max_depth_seen = split.depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()


# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The maximal depth of the tree.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            # it's small; copy for memory safety
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs).copy()

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes, 
        int n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef void _resize(self, SIZE_t capacity) except *:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays."""
        if self._resize_c(capacity) != 0:
            raise MemoryError()

    # XXX using (size_t)(-1) is ugly, but SIZE_MAX is not available in C89
    # (i.e., older MSVC).
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil:
        """Guts of _resize. Returns 0 for success, -1 for error."""
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        # XXX no safe_realloc here because we need to grab the GIL
        cdef void* ptr = realloc(self.nodes, capacity * sizeof(Node))
        if ptr == NULL:
            return -1
        self.nodes = <Node*> ptr
        ptr = realloc(self.value,
                      capacity * self.value_stride * sizeof(double))
        if ptr == NULL:
            return -1
        self.value = <double*> ptr

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples, double weighted_n_node_samples,
                          DOUBLE_t* value) nogil:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count
        cdef SIZE_t i

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        for i in range(self.value_stride):
            self.value[node_id*self.value_stride + i] = value[i]

        self.node_count += 1
        return node_id

    cpdef np.ndarray predict(self, object X):
        """Predict target for X."""

        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)


    cdef inline np.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset
        return out

    cdef inline np.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.

        """
        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                       dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features * sizeof(DTYPE_t))
        safe_realloc(&feature_to_sample, n_features * sizeof(SIZE_t))

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out

    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

# =============================================================================
# Utils
# =============================================================================

# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (DOUBLE_t*)
    (SIZE_t*)
    (unsigned char*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        raise MemoryError("could not allocate (%d * %d) bytes"
                          % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        raise MemoryError("could not allocate %d bytes" % nbytes)
    p[0] = tmp
    return tmp  # for convenience


def _realloc_test():
    # Helper for tests. Tries to allocate <size_t>(-1) / 2 * sizeof(size_t)
    # bytes, which will always overflow.
    cdef SIZE_t* p = NULL
    safe_realloc(&p, <size_t>(-1) / 2)
    if p != NULL:
        free(p)
        assert False


# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)

cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Encapsulate data into a 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data)

cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return low + our_rand_r(random_state) % (high - low)

cdef inline double rand_uniform(double low, double high,
                                UINT32_t* random_state) nogil:
    """Generate a random double in [low; high)."""
    return ((high - low) * <double> our_rand_r(random_state) /
            <double> RAND_R_MAX) + low

cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)


"""
Baseline Model Helpers
"""

######################
### Imports
######################

## External Libraries
import numpy as np
from scipy import sparse

######################
### Classes
######################

class ConditionalMajorityClassifier(object):

    """
    Look at a binary matrix (1 entry max per row) and classify
    based on the conditional criteria (e.g., keyword-conditional majority)
    """

    def __init__(self, alpha=1, **kwargs):
        """
        
        """
        ## Working Variable Space
        self._classes = None
        self._counts = None
        ## Properties
        self.alpha = alpha
        self.coef_ = None
    
    def fit(self, X, y, **kwargs):
        """
        
        """
        ## Check Type
        if not isinstance(X, sparse.csr_matrix) and not isinstance(X, np.ndarray):
            raise TypeError("Input data type not recongized")
        ## Check Type
        if not (np.logical_or(X.sum(axis=1) == 1, X.sum(axis=1) == 0)).all():
            raise ValueError("Expected binary rows.")
        ## Initiaize
        self._classes = sorted(set(y))
        self._counts = np.zeros((X.shape[1], len(self._classes)), dtype=int)
        ## Class to Ind
        class2ind = dict(zip(self._classes, range(len(self._classes))))
        ## Drop Null Rows
        nn_ind = (X.sum(axis=1) > 0).nonzero()[0]
        X = X[nn_ind]
        y = y[nn_ind]
        ## Get Nonzero Entries
        X_ind = X.nonzero()[1]
        ## Update Counts
        for _x, _y in zip(X_ind, y):
            self._counts[_x, class2ind[_y]] += 1 
        ## Transformation into Probabilities
        self.coef_ = (self._counts + self.alpha) / (self._counts + self.alpha).sum(axis=1, keepdims=True)
        self.coef_ = self.coef_.T
        return self
    
    def predict_proba(self, X, **kwargs):
        """
        
        """
        ## Check Type
        if not isinstance(X, sparse.csr_matrix) and not isinstance(X, np.ndarray):
            raise TypeError("Input data type not recongized")
        ## Check Sum
        if not (np.logical_or(X.sum(axis=1) == 1, X.sum(axis=1) == 0)).all():
            raise ValueError("Expected binary rows.")
        ## Check Shape
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError("Shape Mismatch.")
        ## Initialize Outputs
        prob = np.zeros((X.shape[0], self.coef_.shape[0]))
        ## Get Nonzero Entries
        row_nn, col_nn = X.nonzero()        
        ## Fill Non-zero Probabilities
        prob[row_nn,:] = self.coef_[:,col_nn].T
        ## Fill Uniform Probs
        prob[(prob.max(axis=1) == 0).nonzero()[0],:] += (1 / self.coef_.shape[0])
        ## Return
        return prob
    
    def predict(self, X):
        """
        
        """
        X_prob = self.predict_proba(X)
        X_prob_argmax = X_prob.argmax(axis=1)
        y = np.array([self._classes[ind] for ind in X_prob_argmax])
        return y
    
    def fit_predict(self, X, y, **kwargs):
        """
        
        """
        self = self.fit(X, y)
        y_pred = self.predict_proba(X)
        return y_pred



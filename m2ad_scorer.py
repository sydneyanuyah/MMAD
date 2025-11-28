"""
M2AD (Multivariate Anomaly Detection) scoring module.
"""
import numpy as np
from utils import (
    fit_gmm_1d,
    gmm_two_sided_pvals,
    fisher_aggregate,
    fit_gamma_moments,
    gamma_quantile,
    point_error,
    area_error,
    smooth_errors
)


class M2ADScorer:
    """
    M2AD anomaly scorer combining GMM fitting, Fisher aggregation,
    and Gamma threshold calibration.
    """
    
    def __init__(
        self,
        error_mode="area",
        area_l=2,
        ewma_alpha=0.2,
        kmax=3,
        target_fpr=0.01,
        min_p=1e-12,
        weights=None,
        seed=42
    ):
        assert error_mode in ("point", "area")
        self.error_mode = error_mode
        self.area_l = int(area_l)
        self.ewma_alpha = float(ewma_alpha)
        self.kmax = int(kmax)
        self.target_fpr = float(target_fpr)
        self.min_p = float(min_p)
        self.weights = weights
        self.seed = int(seed)
        
        self.gmms = None
        self.gamma = None
        self.threshold = None
    
    @staticmethod
    def _ewma_1d(x, alpha):
        """Apply EWMA to 1D array."""
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
        return out
    
    def _smooth(self, E):
        """Apply EWMA smoothing to error matrix."""
        out = np.zeros_like(E)
        for j in range(E.shape[1]):
            out[:, j] = self._ewma_1d(E[:, j], self.ewma_alpha)
        return out
    
    def _compute_errors(self, X, Y):
        """Compute raw errors between X and Y."""
        if self.error_mode == "point":
            return point_error(X, Y)
        else:
            return area_error(X, Y, l=self.area_l)
    
    def _residuals(self, X, Y):
        """Compute smoothed residuals."""
        E = self._compute_errors(X, Y)
        return self._smooth(E)
    
    def fit(self, X_train, Yhat_train):
        """
        Fit M2AD scorer on training data.
        
        Args:
            X_train: True training data (T, d)
            Yhat_train: Predicted training data (T, d)
        """
        assert X_train.shape == Yhat_train.shape
        
        # Compute training errors
        Etr = self._residuals(X_train, Yhat_train)
        
        # Fit GMMs per dimension
        gmms = []
        Ptr = np.zeros_like(Etr)
        
        for j in range(Etr.shape[1]):
            gmm = fit_gmm_1d(Etr[:, j], k_max=self.kmax, seed=self.seed)
            gmms.append(gmm)
            Ptr[:, j] = gmm_two_sided_pvals(gmm, Etr[:, j], min_p=self.min_p)
        
        # Fisher aggregation
        S_tr = fisher_aggregate(Ptr, weights=self.weights, min_p=self.min_p)
        
        # Fit Gamma distribution
        gamma_params = fit_gamma_moments(S_tr)
        
        # Compute threshold
        thr = gamma_quantile(gamma_params, 1.0 - self.target_fpr)
        
        self.gmms = gmms
        self.gamma = (gamma_params.alpha, gamma_params.theta)
        self.threshold = float(thr)
        
        return self
    
    def score(self, X_test, Yhat_test):
        """
        Score test data.
        
        Args:
            X_test: True test data (T, d)
            Yhat_test: Predicted test data (T, d)
        
        Returns:
            Dictionary with scores, labels, and threshold
        """
        assert self.gmms is not None and self.threshold is not None, \
            "Call fit() first."
        assert X_test.shape == Yhat_test.shape
        
        # Compute test errors
        Ete = self._residuals(X_test, Yhat_test)
        
        # Compute p-values
        Pte = np.zeros_like(Ete)
        for j in range(Ete.shape[1]):
            Pte[:, j] = gmm_two_sided_pvals(
                self.gmms[j],
                Ete[:, j],
                min_p=self.min_p
            )
        
        # Fisher aggregation
        S_te = fisher_aggregate(Pte, weights=self.weights, min_p=self.min_p)
        
        # Threshold
        label = (S_te >= self.threshold).astype(np.int32)
        
        return {
            "score": S_te,
            "label": label,
            "threshold": float(self.threshold)
        }

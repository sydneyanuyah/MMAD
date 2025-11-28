"""
Utility functions for anomaly detection.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.mixture import GaussianMixture
from scipy.stats import gamma as sp_gamma
from dataclasses import dataclass


# ===================== Datasets =====================
class PredDataset(Dataset):
    """For LSTM Predictor: input window t..t-w+1 → predict x_{t+1}."""
    def __init__(self, X, w):
        self.X = X
        self.w = w
        self.T, self.d = X.shape
        self.n = max(0, self.T - w)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        x_win = self.X[i:i+self.w]
        y_next = self.X[i+self.w]
        return torch.from_numpy(x_win), torch.from_numpy(y_next)


class RecDataset(Dataset):
    """For AE/VAE: input window → reconstruct the same window."""
    def __init__(self, X, w):
        self.X = X
        self.w = w
        self.T, self.d = X.shape
        self.n = max(0, self.T - w + 1)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        x_win = self.X[i:i+self.w]
        return torch.from_numpy(x_win), torch.from_numpy(x_win)


# ===================== Error Computation =====================
def ewma(arr, alpha):
    """Exponentially weighted moving average."""
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def point_error(x, y):
    """Point-wise absolute error."""
    return np.abs(x - y)


def area_error(x, y, l=2):
    """Area-based error with window integration."""
    resid = x - y
    k = 2 * l + 1
    from numpy.lib.stride_tricks import sliding_window_view
    pad = l
    rpad = np.pad(resid, ((pad, pad), (0, 0)), mode='edge')
    sw = sliding_window_view(rpad, (k, resid.shape[1]))[:, 0, :]
    area = sw.mean(axis=1)
    return np.abs(area)


def smooth_errors(E, alpha):
    """Apply EWMA smoothing to error matrix."""
    out = np.zeros_like(E)
    for j in range(E.shape[1]):
        out[:, j] = ewma(E[:, j], alpha)
    return out


# ===================== GMM & Statistical Functions =====================
def fit_gmm_1d(errors_train, k_max=3, seed=42):
    """Fit 1D GMM with BIC selection."""
    best = None
    best_bic = np.inf
    X = np.asarray(errors_train, dtype=np.float64).reshape(-1, 1)
    
    for k in range(1, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='diag',
            random_state=seed
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best = gmm
    
    return best


def gmm_two_sided_pvals(gmm, x, min_p=1e-12):
    """Compute two-sided p-values from GMM."""
    x = np.asarray(x, dtype=np.float64)
    wt = gmm.weights_
    mu = gmm.means_.ravel()
    sigma = np.sqrt(gmm.covariances_.ravel())
    
    xt = torch.from_numpy(x).float()
    p = torch.zeros_like(xt)
    
    for wj, muj, sj in zip(wt, mu, sigma):
        dist = torch.distributions.Normal(
            loc=torch.tensor(muj, dtype=torch.float32),
            scale=torch.tensor(max(sj, 1e-6), dtype=torch.float32)
        )
        p += torch.tensor(wj, dtype=torch.float32) * dist.cdf(xt)
    
    p = p.numpy()
    two_sided = 2.0 * np.minimum(p, 1.0 - p)
    two_sided = np.clip(two_sided, min_p, 1.0)
    
    return two_sided


# ===================== Fisher & Gamma =====================
@dataclass
class GammaParams:
    alpha: float
    theta: float


def fisher_aggregate(pvals_mat, weights=None, min_p=1e-12):
    """Fisher's method for combining p-values."""
    pvals_mat = np.asarray(pvals_mat, dtype=np.float64)
    T, d = pvals_mat.shape
    
    if weights is None:
        weights = np.ones(d, dtype=np.float64) / d
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)
    
    p = np.clip(pvals_mat, min_p, 1.0)
    St = -2.0 * np.sum(
        np.log(p) * weights.reshape(1, -1).repeat(T, 0),
        axis=1
    )
    
    return St


def fit_gamma_moments(samples, eps=1e-12):
    """Fit Gamma distribution using method of moments."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    
    if x.size == 0:
        return GammaParams(alpha=1.0, theta=eps)
    
    m = x.mean()
    v = x.var(ddof=1) if x.size > 1 else (m ** 2)
    v = max(v, eps)
    
    alpha = (m * m) / v
    theta = v / m
    
    return GammaParams(alpha=float(alpha), theta=float(theta))


def gamma_quantile(params, q):
    """Compute quantile of Gamma distribution."""
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1). Got {q}")
    return float(sp_gamma.ppf(q, a=params.alpha, scale=params.theta))


# ===================== Event Processing =====================
def ranges_from_labels(y):
    """Convert 0/1 sequence to list of [start, end] ranges."""
    starts, ends = [], []
    in_ev = False
    
    for i, v in enumerate(y):
        if v and not in_ev:
            in_ev = True
            cur_s = i
        if in_ev and (i == len(y) - 1 or y[i + 1] == 0):
            in_ev = False
            starts.append(cur_s)
            ends.append(i)
    
    return list(zip(starts, ends))


def overlap(a, b):
    """Compute overlap length between two ranges."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def dilate_binary(y, radius=1):
    """Morphological dilation of binary sequence."""
    y = np.asarray(y).astype(int)
    T = len(y)
    if radius <= 0 or T == 0:
        return y
    
    out = np.zeros_like(y)
    idx = np.where(y == 1)[0]
    
    for i in idx:
        s = max(0, i - radius)
        e = min(T - 1, i + radius)
        out[s:e+1] = 1
    
    return out


def remove_short_events(y, min_len=50):
    """Remove events shorter than min_len timesteps."""
    y = np.asarray(y).astype(int)
    out = y.copy()
    
    for s, e in ranges_from_labels(out):
        if (e - s + 1) < min_len:
            out[s:e+1] = 0
    
    return out


def canonical_postprocess_labels(y, dilate_radius=1, min_event_len=5):
    """Apply canonical post-processing to prediction labels."""
    y = dilate_binary(y, radius=dilate_radius)
    y = remove_short_events(y, min_len=min_event_len)
    return y.astype(int)


# ===================== Metrics =====================
def event_scores(y_true, y_pred):
    """Compute event-wise TP, FP, FN."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    true_ev = ranges_from_labels(y_true)
    pred_ev = ranges_from_labels(y_pred)
    
    tp = 0
    matched_true = np.zeros(len(true_ev), dtype=bool)
    
    for pe in pred_ev:
        hit = False
        for i, te in enumerate(true_ev):
            if overlap(pe, te) > 0:
                hit = True
                matched_true[i] = True
        tp += int(hit)
    
    fp = len(pred_ev) - tp
    fn = np.sum(~matched_true)
    
    return tp, fp, fn


def prf(tp, fp, fn, beta=1.0):
    """Compute precision, recall, F-beta score."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    
    if p == 0 and r == 0:
        return p, r, 0.0
    
    f = (1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) else 0.0
    
    return p, r, f

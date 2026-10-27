"""NICE: Negative-control Invariant Concept Erasure.

Shortcut-robust detection when every case is collected through one pathway (hospital biobank,
h=1) and the usual controls through another (volunteers, h=0), so that the cell (case, h=0) is
structurally empty.  Sera of patients WITHOUT the target disease that went through the hospital
pathway (other cancers) are used as negative controls to
  (1) identify the pathway direction on the y=0 stratum only  (conditional concept erasure),
  (2) fit a pathway/type-balanced ridge-logistic detector on the erased features,
  (3) set a conformal threshold that controls the false-positive rate in both pathways
      (worst-pathway threshold).
Only numpy is needed.
"""
import numpy as np
from math import ceil


# ----------------------------------------------------------------------------- basic tools
def auc(pos, neg):
    """Mann-Whitney AUC of positives against negatives (ties count 1/2)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    s = np.r_[pos, neg]
    o = np.argsort(s, kind='mergesort'); r = np.empty(len(s))
    ss = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ridge_logistic(X, y, w=None, alpha=1e-2, iters=60, tol=1e-9):
    """min_b  sum_i w_i logloss_i + alpha/2 ||beta||^2  (weights normalised to 1), Newton."""
    n, p = X.shape
    w = np.ones(n) / n if w is None else w / w.sum()
    Xb = np.c_[X, np.ones(n)]; beta = np.zeros(p + 1)
    R = np.eye(p + 1) * alpha; R[-1, -1] = 0.0
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        g = Xb.T @ (w * (mu - y)) + R @ beta
        H = (Xb * (w * np.maximum(mu * (1 - mu), 1e-7))[:, None]).T @ Xb + R + 1e-10 * np.eye(p + 1)
        step = np.linalg.solve(H, g); beta -= step
        if np.abs(step).max() < tol:
            break
    return beta[:-1], beta[-1]


def group_weights(masks):
    """Each mask gets total weight 1/len(masks)."""
    w = np.zeros(len(masks[0]))
    for m in masks:
        if m.any():
            w[m] += 1.0 / (m.sum() * len(masks))
    return w


def conformal_quantile(scores, alpha):
    """Split-conformal upper quantile: P(new score > q) <= alpha for exchangeable scores."""
    s = np.sort(np.asarray(scores, float)); n = len(s)
    k = ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else s[k - 1]


# ----------------------------------------------------------------------------- concept erasure
def fit_erasure(A, Z, w=None, ridge=1e-3):
    """Least-squares concept erasure (LEACE form) fitted on the rows of A.
    A: [n,p] features, Z: [n,k] concept indicators.  Returns r(x) = mu + P (x - mu) with
    P = I - Sigma^{1/2} Q Sigma^{-1/2},  Q = orthogonal projector on colspace(Sigma^{-1/2} Sigma_xz)."""
    Z = Z.reshape(len(A), -1).astype(float)
    w = np.ones(len(A)) / len(A) if w is None else w / w.sum()
    mu = (A * w[:, None]).sum(0); Ac = A - mu
    S = (Ac * w[:, None]).T @ Ac + ridge * np.eye(A.shape[1])
    ev, U = np.linalg.eigh(S)
    W = (U / np.sqrt(ev)) @ U.T; Wi = (U * np.sqrt(ev)) @ U.T
    Zc = Z - (Z * w[:, None]).sum(0)
    a = W @ ((Ac * w[:, None]).T @ Zc)
    u, s, _ = np.linalg.svd(a, full_matrices=False); u = u[:, s > 1e-10 * max(s.max(), 1e-300)]
    P = np.eye(A.shape[1]) - Wi @ (u @ u.T) @ W
    return mu, P


def apply_erasure(X, mu, P):
    return (X - mu) @ P.T + mu


# ----------------------------------------------------------------------------- NICE
class NICE:
    """fit(A, y, h, types): A features, y in {0,1} disease, h in {0,1} pathway, types: negative-type labels.
    erasure: 'conditional' (default, y=0 stratum only) | 'naive' (all samples) | 'multiclass' | None
    weights: 'type' (pathway halves, types equal inside the hospital half) | 'pathway' | 'class' | 'binary'
    """

    def __init__(self, alpha=1e-2, erasure='conditional', weights='type'):
        self.alpha, self.erasure, self.weights = alpha, erasure, weights

    def _w(self, y, h, types):
        pc, nc, hn = y == 1, (y == 0) & (h == 0), (y == 0) & (h == 1)
        w = np.zeros(len(y))
        if self.weights == 'class':
            w[pc] = 0.5 / pc.sum(); w[~pc] = 0.5 / (~pc).sum()
        elif self.weights == 'binary':
            w[pc] = 0.5 / pc.sum(); w[nc] = 0.5 / nc.sum()
        elif self.weights == 'pathway' or not hn.any():
            w[pc] = 0.5 / pc.sum()
            if hn.any():
                w[nc] = 0.25 / nc.sum(); w[hn] = 0.25 / hn.sum()
            else:
                w[nc] = 0.5 / nc.sum()
        else:  # 'type'
            w[pc] = 0.5 / pc.sum(); w[nc] = 0.25 / nc.sum()
            ts = np.unique(types[hn])
            for t in ts:
                k = hn & (types == t); w[k] = 0.25 / (len(ts) * k.sum())
        return w

    def fit(self, A, y, h, types):
        y = np.asarray(y); h = np.asarray(h); neg = y == 0
        self.mu_, self.P_ = np.zeros(A.shape[1]), np.eye(A.shape[1])
        if self.erasure == 'conditional' and (neg & (h == 1)).any():
            w0 = group_weights([neg[neg] & (h[neg] == 0), neg[neg] & (h[neg] == 1)])
            self.mu_, self.P_ = fit_erasure(A[neg], h[neg], w0)
        elif self.erasure == 'naive':
            self.mu_, self.P_ = fit_erasure(A, h, group_weights([h == 0, h == 1]))
        elif self.erasure == 'multiclass' and (neg & (h == 1)).any():
            ts = np.unique(types[neg & (h == 1)])
            Z = np.stack([(types[neg] == t) & (h[neg] == 1) for t in ts], 1)
            w0 = group_weights([h[neg] == 0] + [Z[:, i] for i in range(Z.shape[1])])
            self.mu_, self.P_ = fit_erasure(A[neg], Z, w0)
        w = self._w(y, h, types); keep = w > 0
        self.beta_, self.b0_ = ridge_logistic(self.transform(A[keep]), y[keep].astype(float), w[keep], self.alpha)
        return self

    def transform(self, A):
        return apply_erasure(A, self.mu_, self.P_)

    def decision_function(self, A):
        return self.transform(A) @ self.beta_ + self.b0_


def worst_group_threshold(scores, y, h, fpr=0.05):
    """Largest of the two pathway-wise conformal (1-fpr) quantiles of the negative calibration scores:
    the false-positive rate is controlled among volunteer controls AND among hospital negatives."""
    neg = y == 0
    qs = [conformal_quantile(scores[neg & (h == k)], fpr) for k in (0, 1) if (neg & (h == k)).any()]
    return max(qs)


# ----------------------------------------------------------------------------- audit quantities
def shortcut_audit(A, y, h, ridge=1e-3):
    """Mahalanobis geometry of the disease effect D_y = mu(1,1)-mu(0,1) and the pathway effect
    D_h = mu(0,1)-mu(0,0), with the pooled within-group covariance."""
    pc, nc, hn = y == 1, (y == 0) & (h == 0), (y == 0) & (h == 1)
    mus = [A[k].mean(0) for k in (pc, nc, hn)]
    S = sum((A[k] - m).T @ (A[k] - m) for k, m in zip((pc, nc, hn), mus)) / (len(y) - 3) + ridge * np.eye(A.shape[1])
    Dy, Dh = mus[0] - mus[2], mus[2] - mus[1]
    Si = np.linalg.inv(S)
    dy2, dh2, c = Dy @ Si @ Dy, Dh @ Si @ Dh, Dy @ Si @ Dh
    return dict(dy=float(np.sqrt(dy2)), dh=float(np.sqrt(dh2)), cos2=float(c * c / (dy2 * dh2)), cos=float(c / np.sqrt(dy2 * dh2)))

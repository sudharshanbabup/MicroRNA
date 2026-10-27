"""Semi-synthetic stress test in which the structurally empty cell becomes observable.

Only REAL hospital sera of one study are used (PC and other cancers; one real pathway).  A known additive
pathway effect v is injected in the PCA representation:
   pathway 1 = serum + v,   pathway 0 = serum (unchanged).
Training follows the case-control design: all training cases receive v (cell (1,1)); the other-cancer sera are
split at random into "controls" (0,0) and "negative controls" (0,1).  At test time every serum is evaluated in
BOTH pathways, so the sensitivity of volunteer-pathway cases, cell (1,0), is measured directly.
v has Mahalanobis length d_h and cosine rho with the disease direction (pooled within-group covariance):
   v = d_h * Sigma^{1/2} ( rho * u_y + sqrt(1-rho^2) * u_perp ).
Usage: python semi_synthetic.py <repeats> [study] [first_rep]      -> results/semi.jsonl
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_experiments as R
import baselines as B
from nice import NICE, conformal_quantile, worst_group_threshold

FPR = 0.05
METHODS = {'ERM-LR': (B.erm, False), 'OvR': (B.ovr, True), 'Matched': (B.matched, True), 'RWG': (B.rwg, True),
           'GroupDRO': (B.groupdro, True), 'IRM': (B.irm, True), 'V-REx': (B.vrex, True), 'CORAL': (B.coral, True),
           'DANN': (B.dann, True), 'LEACE-naive': (None, True), 'NICE': (None, True)}


def sqrtm_psd(S):
    ev, U = np.linalg.eigh(S); return (U * np.sqrt(np.maximum(ev, 1e-12))) @ U.T


def one(study, rep, fold, rho, dh, npc=100):
    rng = np.random.default_rng(100000 * rep + 100 * fold + int(1000 * rho) + int(10 * dh))
    idx = np.where((R.ST == study) & np.isin(R.G, ['PC', 'OC']))[0]
    frng = np.random.default_rng(5000 + rep); f = np.zeros(len(idx), int)
    for grp in ('PC', 'OC'):
        k = np.where(R.G[idx] == grp)[0]; frng.shuffle(k); f[k] = np.arange(len(k)) % 5
    tr, te = idx[f != fold], idx[f == fold]
    A, At = R.front_end(R.X[tr], R.X[te], npc)
    y = (R.G[tr] == 'PC').astype(int); yt = (R.G[te] == 'PC').astype(int); types = R.DX[tr]
    # pathway assignment in training: cases -> 1, other-cancer sera -> 0/1 at random
    h = np.ones(len(y), int); neg = np.where(y == 0)[0]; h[rng.permutation(neg)[:len(neg) // 2]] = 0
    # injected effect with prescribed Mahalanobis length and cosine to the disease direction
    S = (np.cov(A[y == 1].T) * (y.sum() - 1) + np.cov(A[y == 0].T) * ((1 - y).sum() - 1)) / (len(y) - 2) + 1e-3 * np.eye(A.shape[1])
    Sh = sqrtm_psd(S); uy = np.linalg.solve(Sh, A[y == 1].mean(0) - A[y == 0].mean(0)); uy /= np.linalg.norm(uy)
    g = rng.standard_normal(A.shape[1]); g -= (g @ uy) * uy; g /= np.linalg.norm(g)
    v = dh * Sh @ (rho * uy + np.sqrt(1 - rho ** 2) * g)
    Atr = A + np.outer(h, v)
    inner = R.inner_folds(R.G[tr], np.where(y == 1, 'pc', np.where(h == 1, 'n1', 'n0')), rng)
    out = {}
    for nm, (fn, aware) in METHODS.items():
        if nm == 'NICE':
            fit = lambda A_, y_, h_, t_: NICE().fit(A_, y_, h_, t_).decision_function
        elif nm == 'LEACE-naive':
            fit = lambda A_, y_, h_, t_: NICE(erasure='naive', weights='pathway').fit(A_, y_, h_, t_).decision_function
        else:
            fit = lambda A_, y_, h_, t_, fn=fn: fn(A_, y_, h_, t_, rep)
        oof = np.zeros(len(y))
        for k in range(4):
            a, b = inner != k, inner == k
            oof[b] = fit(Atr[a], y[a], h[a], types[a])(Atr[b])
        s = fit(Atr, y, h, types)
        thr = {'nc': conformal_quantile(oof[(y == 0) & (h == 0)], FPR)}
        if aware:
            thr['wg'] = worst_group_threshold(oof, y, h, FPR)
        s0, s1 = s(At), s(At + v)                          # every test serum in both pathways
        res = {}
        for tn, t in thr.items():
            c = dict(se11=float((s1[yt == 1] > t).mean()), se10=float((s0[yt == 1] > t).mean()),
                     sp00=float((s0[yt == 0] <= t).mean()), sp01=float((s1[yt == 0] <= t).mean()))
            c['wga4'] = min(c.values()); res.update({f'{k}@{tn}': x for k, x in c.items()})
        out[nm] = res
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]); study = sys.argv[2] if len(sys.argv) > 2 else 'B'; rep0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    for rep in range(rep0, rep0 + reps):
        for dh in (3.0,):
            for rho in (0.0, 0.2, 0.4, 0.6, 0.8):
                for fold in range(5):
                    r = one(study, rep, fold, rho, dh)
                    with open(os.path.join(R.RES, 'semi.jsonl'), 'a') as fh:
                        fh.write(json.dumps(dict(study=study, rep=rep, fold=fold, rho=rho, dh=dh, res=r)) + '\n')
        print('semi rep', rep, 'done', flush=True)

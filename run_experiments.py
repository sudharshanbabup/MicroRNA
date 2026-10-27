"""All real-data experiments.  Usage:  python run_experiments.py <protocol> <repeats> [seed0]
protocols
  cv     repeated stratified 5-fold CV inside each study (benign-disease sera are never used for
         training or calibration and serve as unseen hospital negatives)
  lonto  leave-one-negative-type-out: one hospital negative type is removed from training and
         calibration and used only for testing (3-fold on the remaining sera)
  cross  train on one study / test on another (B->C, C->B, BC->A, BC->D)
  abl    NICE ablations (cv protocol)
  mneg   number of negative-control sera available for training (cv protocol, study B)
Results are appended as JSON lines to results/<protocol>.jsonl
"""
import sys, os, json, time, warnings
import numpy as np
from scipy.stats import norm
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from nice import NICE, auc, conformal_quantile, worst_group_threshold, shortcut_audit
import baselines as B

FPR = 0.05; NPC = 100
DATA_PATH = os.path.join(HERE, 'data', 'mirna_dataset.npz') if os.path.exists(os.path.join(HERE, 'data', 'mirna_dataset.npz')) else os.path.join(HERE, '..', 'data', 'mirna_dataset.npz')
d = np.load(DATA_PATH)
X, ST, G, DX = d['X'].astype(np.float64), d['study'], d['group'], d['dx']
RES = os.path.join(HERE, 'results') if not os.path.exists(os.path.join(HERE, '..', 'results')) else os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)


def front_end(Xtr, Xte, k=NPC):
    """standardise -> PCA(k) -> unit-variance scores; fitted on the training sera only."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    U, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    k = min(k, len(s) - 1); V = Vt[:k].T; c = Z.mean(0)
    A = (Z - c) @ V; sc = A.std(0) + 1e-9
    return A / sc, (((Xte - mu) / sd) - c) @ V / sc


def metrics(s_te, g_te, unseen, thr):
    pc, nc = s_te[g_te == 'PC'], s_te[g_te == 'NC']
    hn, un = s_te[(g_te == 'OC') & ~unseen], s_te[unseen]
    m = dict(auc_nc=auc(pc, nc), auc_hn=auc(pc, hn), auc_un=auc(pc, un),
             se95hn=float((pc > np.quantile(hn, 0.95)).mean()) if len(hn) else np.nan,
             se95un=float((pc > np.quantile(un, 0.95)).mean()) if len(un) else np.nan)
    for nm, t in thr.items():
        v = dict(se=float((pc > t).mean()), sp_nc=float((nc <= t).mean()),
                 sp_hn=float((hn <= t).mean()) if len(hn) else np.nan,
                 sp_un=float((un <= t).mean()) if len(un) else np.nan)
        v['wga'] = float(np.nanmin(list(v.values())))
        m.update({f'{k}@{nm}': x for k, x in v.items()})
    return m


def inner_folds(g, types, rng, K=4):
    """stratified (by group and type) inner folds used for cross-conformal calibration"""
    f = np.zeros(len(g), int)
    for key in set(zip(g, types)):
        idx = np.where((g == key[0]) & (types == key[1]))[0]; rng.shuffle(idx)
        f[idx] = (np.arange(len(idx)) + rng.integers(K)) % K
    return f


def run_fold(Xtr, gtr, ttr, Xte, gte, tte, unseen, seed, methods='all', n_neg=None, npc=NPC):
    """gtr in {PC, NC, OC}; unseen: boolean mask over the test sera (unseen hospital negatives).
    Every method is fitted on all training sera; its thresholds come from out-of-fold (K=4) scores of the
    training negatives (cross-conformal calibration), so no method is calibrated on resubstitution scores."""
    rng = np.random.default_rng(seed)
    if n_neg is not None:                      # keep only n_neg negative-control sera
        oc = np.where(gtr == 'OC')[0]
        if n_neg < len(oc):
            keep = np.ones(len(gtr), bool); keep[rng.choice(oc, len(oc) - n_neg, replace=False)] = False
            Xtr, gtr, ttr = Xtr[keep], gtr[keep], ttr[keep]
    A, Bt = front_end(Xtr, Xte, npc)
    y = (gtr == 'PC').astype(int); h = (gtr != 'NC').astype(int)
    inner = inner_folds(gtr, ttr, rng)
    out = {}; keepf = {}

    def add(name, fn, aware):
        """fn(A, y, h, types) -> scoring function"""
        oof = np.zeros(len(y))
        for k in range(inner.max() + 1):
            tr = inner != k
            oof[~tr] = fn(A[tr], y[tr], h[tr], ttr[tr])(A[~tr])
        f = fn(A, y, h, ttr)
        thr = {'nc': conformal_quantile(oof[(y == 0) & (h == 0)], FPR)}
        if aware:
            thr['wg'] = worst_group_threshold(oof, y, h, FPR)
        st = f(Bt); out[name] = metrics(st, gte, unseen, thr); keepf[name] = (oof, st, thr)

    nice = lambda **kw: (lambda A_, y_, h_, t_: NICE(**kw).fit(A_, y_, h_, t_).decision_function)
    base = lambda fn: (lambda A_, y_, h_, t_: fn(A_, y_, h_, t_, seed))
    if methods in ('all', 'main'):
        for nm, fn in B.GROUP_UNAWARE.items():
            add(nm, base(fn), False)
        # audit of the conventional classifier (Theorem 1 / Corollary 1) from training-side scores only
        oof, st, thr = keepf['ERM-LR']; t = thr['nc']
        snc, shn = oof[(y == 0) & (h == 0)], oof[(y == 0) & (h == 1)]
        kap = (shn.mean() - snc.mean()) / snc.std(ddof=1); pc_te = st[gte == 'PC']
        out['ERM-LR'].update(kappa=float(kap),
                             pred_sp_hn=float(norm.cdf((t - shn.mean()) / shn.std(ddof=1))),
                             pred_sp_hn_eqvar=float(norm.cdf(norm.ppf(1 - FPR) - kap)),
                             cf_se=float((pc_te - (shn.mean() - snc.mean()) > t).mean()))
        out['ERM-LR'].update({'audit_' + k: v for k, v in shortcut_audit(A, y, h).items()})
        out['ERM-LR+WGcal'] = metrics(st, gte, unseen, {'nc': t, 'wg': worst_group_threshold(oof, y, h, FPR)})
        for nm, fn in B.GROUP_AWARE.items():
            add(nm, base(fn), True)
        add('LEACE-naive', nice(erasure='naive', weights='pathway'), True)
    if methods in ('all', 'main', 'nice'):
        add('NICE', nice(), True)
    if methods == 'abl':
        cfg = {'NICE': {}, 'no erasure': dict(erasure=None), 'naive erasure': dict(erasure='naive'),
               'multiclass erasure': dict(erasure='multiclass'), 'pathway weights': dict(weights='pathway'),
               'class weights': dict(weights='class'), 'binary (PC vs NC)': dict(weights='binary'),
               'no erasure, binary': dict(erasure=None, weights='binary')}
        for a in (1e-3, 1e-1, 1.0):
            cfg[f'alpha={a:g}'] = dict(alpha=a)
        for nm, kw in cfg.items():
            add(nm, nice(**kw), True)
    if methods == 'mneg':
        for nm, fn in [('ERM-LR', B.erm), ('OvR', B.ovr), ('Matched', B.matched), ('RWG', B.rwg), ('GroupDRO', B.groupdro)]:
            add(nm, base(fn), nm != 'ERM-LR')
        add('NICE', nice(), True)
    return out


def dump(proto, rec):
    with open(os.path.join(RES, proto + '.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')


def cv_protocol(proto, reps, seed0, methods, studies='ABCD', n_negs=(None,), npcs=(NPC,)):
    for rep in range(seed0, seed0 + reps):
        for s in studies:
            idx = np.where(ST == s)[0]; rng = np.random.default_rng(1000 * rep + ord(s))
            fold = np.zeros(len(idx), int)
            for grp in np.unique(G[idx]):                       # stratified by group
                k = np.where(G[idx] == grp)[0]; rng.shuffle(k); fold[k] = np.arange(len(k)) % 5
            for f in range(5):
                tr = idx[(fold != f) & (G[idx] != 'BEN')]
                te = idx[(fold == f) | (G[idx] == 'BEN')]       # all benign sera are unseen negatives
                for nn in n_negs:
                    for k in npcs:
                        t0 = time.time()
                        r = run_fold(X[tr], G[tr], DX[tr], X[te], G[te], DX[te], G[te] == 'BEN',
                                     seed=10000 * rep + 10 * f + ord(s), methods=methods, n_neg=nn, npc=k)
                        dump(proto, dict(study=s, rep=rep, fold=f, n_neg=nn, npc=k, sec=round(time.time() - t0, 2), res=r))
        print(proto, 'rep', rep, 'done', flush=True)


def lonto_protocol(reps, seed0):
    for rep in range(seed0, seed0 + reps):
        for s in 'ABCD':
            idx = np.where(ST == s)[0]
            types = [t for t in np.unique(DX[idx]) if G[idx][DX[idx] == t][0] in ('OC', 'BEN') and (DX[idx] == t).sum() >= 20]
            for T in types:
                rng = np.random.default_rng(1000 * rep + ord(s) + abs(hash(T)) % 997)
                fold = np.zeros(len(idx), int)
                for grp in np.unique(G[idx]):
                    k = np.where(G[idx] == grp)[0]; rng.shuffle(k); fold[k] = np.arange(len(k)) % 3
                for f in range(3):
                    isT = DX[idx] == T
                    tr = idx[(fold != f) & ~isT & (G[idx] != 'BEN')]
                    te = idx[((fold == f) & np.isin(G[idx], ['PC', 'NC', 'OC']) & ~isT) | isT]
                    r = run_fold(X[tr], G[tr], DX[tr], X[te], G[te], DX[te], DX[te] == T, seed=10000 * rep + 10 * f + ord(s))
                    dump('lonto', dict(study=s, rep=rep, fold=f, heldout=T, heldout_group=str(G[idx][isT][0]),
                                       n_heldout=int(isT.sum()), res=r))
        print('lonto rep', rep, 'done', flush=True)


def cross_protocol(reps, seed0):
    pairs = [('B', 'C'), ('C', 'B'), ('BC', 'A'), ('BC', 'D')]
    for rep in range(seed0, seed0 + reps):
        for a, b in pairs:
            tr = np.where(np.isin(ST, list(a)) & (G != 'BEN'))[0]; te = np.where(np.isin(ST, list(b)))[0]
            r = run_fold(X[tr], G[tr], DX[tr], X[te], G[te], DX[te], G[te] == 'BEN', seed=777 * rep + ord(b[0]))
            dump('cross', dict(train=a, test=b, rep=rep, res=r))
        print('cross rep', rep, 'done', flush=True)


if __name__ == '__main__':
    proto, reps = sys.argv[1], int(sys.argv[2]); seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if proto == 'cv':
        cv_protocol('cv', reps, seed0, 'all')
    elif proto == 'lonto':
        lonto_protocol(reps, seed0)
    elif proto == 'cross':
        cross_protocol(reps, seed0)
    elif proto == 'abl':
        cv_protocol('abl', reps, seed0, 'abl')
    elif proto == 'mneg':
        cv_protocol('mneg', reps, seed0, 'mneg', studies='B', n_negs=(5, 10, 20, 50, 100, 200, 500, None))
    elif proto == 'npc':
        cv_protocol('npc', reps, seed0, 'mneg', studies='ABD', npcs=(10, 25, 50, 100, 200))

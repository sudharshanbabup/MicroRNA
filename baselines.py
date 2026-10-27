"""Baselines.  Every function takes the training representation and returns a scoring function.
A: [n,p] features; y: disease label (1 = pancreatic cancer); h: pathway (1 = hospital biobank);
types: negative-type labels.  Group-unaware baselines only see sera with (y=1) or (y=0,h=0),
i.e. the case-versus-volunteer design used by the published serum-miRNA classifiers.
"""
import numpy as np, torch, torch.nn as nn
from nice import ridge_logistic, group_weights, fit_erasure, apply_erasure

torch.set_num_threads(1)
STEPS, LR, WD, HID = 300, 1e-2, 1e-3, 64


def _lin(b, b0):
    return lambda Z: Z @ b + b0


def _t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


# ----------------------------------------------------------------------------- group-unaware
def erm(A, y, h, types, seed=0, alpha=1e-2):
    k = (y == 1) | (h == 0)
    return _lin(*ridge_logistic(A[k], y[k].astype(float), group_weights([y[k] == 1, y[k] == 0]), alpha))


def xgb(A, y, h, types, seed=0):
    import xgboost
    k = (y == 1) | (h == 0)
    m = xgboost.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                              n_jobs=1, random_state=seed, scale_pos_weight=float((y[k] == 0).sum() / (y[k] == 1).sum()),
                              verbosity=0)
    m.fit(A[k], y[k])
    return lambda Z: m.predict(Z, output_margin=True).astype(float)


def _train(model, loss_fn, steps=STEPS, seed=0, extra_params=()):
    opt = torch.optim.Adam(list(model.parameters()) + list(extra_params), lr=LR, weight_decay=WD)
    for t in range(steps):
        opt.zero_grad(); loss = loss_fn(t); loss.backward(); opt.step()
    return model


def _mlp(p, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(p, HID), nn.ReLU(), nn.Linear(HID, 1))


def _score(model):
    return lambda Z: model(_t(Z)).detach().numpy().ravel().astype(float)


def _bce(logit, y, w=None):
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction='none')
    return l.mean() if w is None else (l * w).sum() / w.sum()


def mlp(A, y, h, types, seed=0):
    k = (y == 1) | (h == 0); X, Y = _t(A[k]), _t(y[k]); W = _t(group_weights([y[k] == 1, y[k] == 0]))
    m = _mlp(A.shape[1], seed)
    return _score(_train(m, lambda t: _bce(m(X).squeeze(1), Y, W)))


def jtt(A, y, h, types, seed=0, up=20.0):
    """Just Train Twice (Liu et al., 2021): up-weight the errors of a heavily regularised first model."""
    k = (y == 1) | (h == 0); Ak, yk = A[k], y[k].astype(float); w = group_weights([yk == 1, yk == 0])
    b, b0 = ridge_logistic(Ak, yk, w, alpha=1.0)
    err = ((Ak @ b + b0) > 0) != (yk == 1)
    return _lin(*ridge_logistic(Ak, yk, w * np.where(err, up, 1.0), 1e-2))


# ----------------------------------------------------------------------------- negative-control aware
def ovr(A, y, h, types, seed=0, alpha=1e-2):
    return _lin(*ridge_logistic(A, y.astype(float), group_weights([y == 1, y == 0]), alpha))


def matched(A, y, h, types, seed=0, alpha=1e-2):
    k = h == 1
    return _lin(*ridge_logistic(A[k], y[k].astype(float), group_weights([y[k] == 1, y[k] == 0]), alpha))


def rwg(A, y, h, types, seed=0, alpha=1e-2):
    """Group-balanced reweighting (Idrissi et al., 2022) over the three observed groups."""
    w = np.zeros(len(y)); pc, nc, hn = y == 1, (y == 0) & (h == 0), (y == 0) & (h == 1)
    w[pc] = 0.5 / pc.sum(); w[nc] = 0.25 / nc.sum(); w[hn] = 0.25 / hn.sum()
    return _lin(*ridge_logistic(A, y.astype(float), w, alpha))


def groupdro(A, y, h, types, seed=0, eta=0.05):
    """Group DRO (Sagawa et al., 2020), linear model, groups = {PC, volunteer NC, hospital negatives}."""
    torch.manual_seed(seed); X, Y = _t(A), _t(y); m = nn.Linear(A.shape[1], 1)
    G = [torch.tensor(k) for k in (y == 1, (y == 0) & (h == 0), (y == 0) & (h == 1))]
    q = torch.ones(3) / 3

    def loss(t):
        nonlocal q
        l = nn.functional.binary_cross_entropy_with_logits(m(X).squeeze(1), Y, reduction='none')
        lg = torch.stack([l[g].mean() for g in G])
        q = q * torch.exp(eta * lg.detach()); q = q / q.sum()
        return (q * lg).sum()
    return _score(_train(m, loss))


def _envs(y, h, seed):
    """Two environments that differ in the control pathway: PC split at random between them."""
    rng = np.random.default_rng(seed); pc = np.where(y == 1)[0]; rng.shuffle(pc)
    e1 = np.zeros(len(y), bool); e2 = np.zeros(len(y), bool)
    e1[pc[::2]] = True; e2[pc[1::2]] = True
    e1 |= (y == 0) & (h == 0); e2 |= (y == 0) & (h == 1)
    return e1, e2


def irm(A, y, h, types, seed=0, lam=100.0, warm=100):
    """IRMv1 (Arjovsky et al., 2019)."""
    torch.manual_seed(seed); m = nn.Linear(A.shape[1], 1); E = _envs(y, h, seed)
    D = [(_t(A[e]), _t(y[e]), _t(group_weights([y[e] == 1, y[e] == 0]))) for e in E]

    def loss(t):
        tot = 0.0; pen = 0.0
        for X, Y, W in D:
            s = torch.ones(1, requires_grad=True)
            l = _bce(m(X).squeeze(1) * s, Y, W)
            g = torch.autograd.grad(l, s, create_graph=True)[0]
            tot = tot + l; pen = pen + (g ** 2).sum()
        lw = lam if t >= warm else 1.0
        out = tot + lw * pen
        return out / lw if lw > 1 else out
    return _score(_train(m, loss))


def vrex(A, y, h, types, seed=0, beta=10.0, warm=100):
    """V-REx (Krueger et al., 2021)."""
    torch.manual_seed(seed); m = nn.Linear(A.shape[1], 1); E = _envs(y, h, seed)
    D = [(_t(A[e]), _t(y[e]), _t(group_weights([y[e] == 1, y[e] == 0]))) for e in E]

    def loss(t):
        r = torch.stack([_bce(m(X).squeeze(1), Y, W) for X, Y, W in D])
        return r.mean() + (beta if t >= warm else 1.0) * r.var()
    return _score(_train(m, loss))


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


def dann(A, y, h, types, seed=0, lam=1.0):
    """Domain-adversarial training (Ganin et al., 2016); the adversary predicts the pathway on y=0 sera."""
    torch.manual_seed(seed); p = A.shape[1]
    feat = nn.Sequential(nn.Linear(p, HID), nn.ReLU()); clf = nn.Linear(HID, 1); adv = nn.Linear(HID, 1)
    X, Y = _t(A), _t(y); W = _t(group_weights([y == 1, (y == 0) & (h == 0), (y == 0) & (h == 1)]))
    neg = torch.tensor(y == 0); Hn = _t(h[y == 0]); Wn = _t(group_weights([h[y == 0] == 0, h[y == 0] == 1]))
    model = nn.Sequential(feat, clf)

    def loss(t):
        f = feat(X)
        return _bce(clf(f).squeeze(1), Y, W) + lam * _bce(adv(_GRL.apply(f[neg])).squeeze(1), Hn, Wn)
    _train(model, loss, extra_params=adv.parameters())
    return _score(model)


def coral(A, y, h, types, seed=0, lam=1.0):
    """Deep CORAL (Sun & Saenko, 2016) aligning hidden means/covariances of the two negative pathways."""
    torch.manual_seed(seed); p = A.shape[1]
    feat = nn.Sequential(nn.Linear(p, HID), nn.ReLU()); clf = nn.Linear(HID, 1); model = nn.Sequential(feat, clf)
    X, Y = _t(A), _t(y); W = _t(group_weights([y == 1, (y == 0) & (h == 0), (y == 0) & (h == 1)]))
    a = torch.tensor((y == 0) & (h == 0)); b = torch.tensor((y == 0) & (h == 1))

    def loss(t):
        f = feat(X); fa, fb = f[a], f[b]
        ca = torch.cov(fa.T); cb = torch.cov(fb.T)
        pen = ((ca - cb) ** 2).sum() / (4 * HID ** 2) + ((fa.mean(0) - fb.mean(0)) ** 2).mean()
        return _bce(clf(f).squeeze(1), Y, W) + lam * pen
    return _score(_train(model, loss))


def inlp(A, y, h, types, seed=0, iters=5, alpha=1e-2):
    """Iterative null-space projection (Ravfogel et al., 2020) of the pathway on y=0 sera, then RWG."""
    neg = y == 0; P = np.eye(A.shape[1]); An = A[neg]; hn = h[neg].astype(float); w = group_weights([hn == 0, hn == 1])
    for _ in range(iters):
        b, _ = ridge_logistic(An @ P, hn, w, alpha)
        b = P @ b; nb = np.linalg.norm(b)
        if nb < 1e-10:
            break
        b /= nb; P = P - np.outer(P @ b, b)
    f = rwg(A @ P, y, h, types, seed, alpha)
    return lambda Z: f(Z @ P)


GROUP_UNAWARE = {'ERM-LR': erm, 'XGBoost': xgb, 'MLP': mlp, 'JTT': jtt}
GROUP_AWARE = {'OvR': ovr, 'Matched': matched, 'RWG': rwg, 'GroupDRO': groupdro, 'IRM': irm, 'V-REx': vrex,
               'DANN': dann, 'CORAL': coral, 'INLP': inlp}

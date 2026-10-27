"""Monte-Carlo checks of the analytical results (Gaussian pathway model).
  x | y,h ~ N(y*D_y + h*D_h, I_p)   (whitened coordinates; d_y=|D_y|, d_h=|D_h|, rho=cos angle)
(a) Corollary 1: specificity of the case-vs-volunteer rule among hospital negatives, Phi(z - kappa),
    and its counterfactual sensitivity, versus the pathway shift d_h.
(b) Theorem 2: discriminability after conditional erasure, d_y*sqrt(1-rho^2), versus rho.
(c) Proposition 1: residual pathway shift after erasure estimated from m negative controls.
(d) Theorem 3: false-positive rate of the worst-pathway conformal threshold.
Writes results/theory_sim.json
"""
import json, os, sys
import numpy as np
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from nice import NICE, ridge_logistic, group_weights, conformal_quantile, worst_group_threshold, fit_erasure, apply_erasure

rng = np.random.default_rng(2026)
ALPHA = 0.05; z = norm.ppf(1 - ALPHA)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
out = {}


def effects(p, dy, dh, rho):
    Dh = np.zeros(p); Dh[0] = dh
    Dy = np.zeros(p); Dy[0] = dy * rho; Dy[1] = dy * np.sqrt(1 - rho ** 2)
    return Dy, Dh


def draw(n, y, h, Dy, Dh):
    return rng.standard_normal((n, len(Dy))) + y * Dy + h * Dh


# ---------------------------------------------------------------- (a) specificity collapse
p, dy, rho, n = 20, 2.0, 0.3, 400
res = []
for dh in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]:
    Dy, Dh = effects(p, dy, dh, rho)
    D = np.sqrt(dy ** 2 + dh ** 2 + 2 * rho * dy * dh); kap = (dh ** 2 + rho * dy * dh) / D
    th_sp, th_se, th_cf = norm.cdf(z - kap), norm.cdf(D - z), norm.cdf(D - kap - z)
    sp, se, cf, sp_n, spv_n, se_n, cf_n = [], [], [], [], [], [], []
    for _ in range(R):
        Xc, Xv, Xn = draw(n, 1, 1, Dy, Dh), draw(n, 0, 0, Dy, Dh), draw(n, 0, 1, Dy, Dh)
        # conventional rule: LDA direction from cases and volunteers, threshold at the volunteers' 95th percentile
        S = (np.cov(Xc.T) + np.cov(Xv.T)) / 2; w = np.linalg.solve(S, Xc.mean(0) - Xv.mean(0))
        T = [draw(4000, a, b, Dy, Dh) for a, b in ((0, 0), (0, 1), (1, 1), (1, 0))]
        t = np.quantile(Xv @ w, 1 - ALPHA)
        sp.append(((T[1] @ w) <= t).mean()); se.append(((T[2] @ w) > t).mean()); cf.append(((T[3] @ w) > t).mean())
        # NICE with the same sera plus the negative controls
        A = np.vstack([Xc, Xv, Xn]); y = np.r_[np.ones(n), np.zeros(2 * n)].astype(int); h = np.r_[np.ones(n), np.zeros(n), np.ones(n)].astype(int)
        m = NICE(alpha=1e-3, weights='pathway').fit(A, y, h, np.array(['x'] * len(y)))
        s = m.decision_function(A); tn = np.quantile(s[(y == 0) & (h == 0)], 1 - ALPHA)
        f = m.decision_function
        spv_n.append((f(T[0]) <= tn).mean()); sp_n.append((f(T[1]) <= tn).mean()); se_n.append((f(T[2]) > tn).mean()); cf_n.append((f(T[3]) > tn).mean())
    dperp = dy * np.sqrt(1 - rho ** 2)
    res.append(dict(dh=dh, kappa=kap, th_sp=th_sp, th_se=th_se, th_cf=th_cf, th_nice_se=norm.cdf(dperp - z),
                    mc_sp=np.mean(sp), mc_se=np.mean(se), mc_cf=np.mean(cf), mc_sp_sd=np.std(sp),
                    nice_sp=np.mean(sp_n), nice_spv=np.mean(spv_n), nice_se=np.mean(se_n), nice_cf=np.mean(cf_n)))
    print('a', res[-1], flush=True)
out['collapse'] = res

# ---------------------------------------------------------------- (b) price of invariance
res = []; dh = 2.0
for rho in [0, 0.2, 0.4, 0.6, 0.8, 0.9]:
    Dy, Dh = effects(p, dy, dh, rho); a_nice, a_match, sh_match = [], [], []
    for _ in range(max(R // 4, 100)):
        Xc, Xv, Xn = draw(2000, 1, 1, Dy, Dh), draw(2000, 0, 0, Dy, Dh), draw(2000, 0, 1, Dy, Dh)
        A = np.vstack([Xc, Xv, Xn]); y = np.r_[np.ones(2000), np.zeros(4000)].astype(int); h = np.r_[np.ones(2000), np.zeros(2000), np.ones(2000)].astype(int)
        m = NICE(alpha=1e-4, weights='pathway').fit(A, y, h, np.array(['x'] * len(y)))
        Tc, Tn = draw(4000, 1, 1, Dy, Dh), draw(4000, 0, 1, Dy, Dh)
        sc, sn = m.decision_function(Tc), m.decision_function(Tn)
        a_nice.append((sc.mean() - sn.mean()) / np.sqrt((sc.var() + sn.var()) / 2))
        w = np.linalg.solve((np.cov(Xc.T) + np.cov(Xn.T)) / 2, Xc.mean(0) - Xn.mean(0))
        sc, sn, sv = Tc @ w, Tn @ w, draw(4000, 0, 0, Dy, Dh) @ w
        a_match.append((sc.mean() - sn.mean()) / np.sqrt((sc.var() + sn.var()) / 2)); sh_match.append((sn.mean() - sv.mean()) / sn.std())
    res.append(dict(rho=rho, th_nice=dy * np.sqrt(1 - rho ** 2), mc_nice=np.mean(a_nice), th_match=dy, mc_match=np.mean(a_match),
                    th_match_shift=rho * dh, mc_match_shift=np.mean(sh_match)))
    print('b', res[-1], flush=True)
out['price'] = res

# ---------------------------------------------------------------- (c) leakage of the estimated eraser
res = []; n0 = 2000
for p_ in (20, 100):
    for m_ in [10, 20, 50, 100, 200, 500, 1000, 2000]:
        Dy, Dh = effects(p_, dy, 2.0, 0.3); eta = 1 / m_ + 1 / n0; leak = []
        for _ in range(max(R // 4, 100)):
            Xv, Xn = draw(n0, 0, 0, Dy, Dh), draw(m_, 0, 1, Dy, Dh)
            mu, P = fit_erasure(np.vstack([Xv, Xn]), np.r_[np.zeros(n0), np.ones(m_)], group_weights([np.r_[np.ones(n0), np.zeros(m_)] > 0, np.r_[np.zeros(n0), np.ones(m_)] > 0]), ridge=1e-6 if m_ + n0 > 5 * p_ else 1e-2)
            leak.append(np.linalg.norm(P @ Dh))            # residual pathway shift (true Sigma = I)
        th = np.sqrt((p_ - 1) * eta * 4.0 / (4.0 + p_ * eta))
        res.append(dict(p=p_, m=m_, th=th, mc=float(np.mean(leak)), mc_rms=float(np.sqrt(np.mean(np.square(leak))))))
        print('c', res[-1], flush=True)
out['leak'] = res

# ---------------------------------------------------------------- (d) conformal worst-pathway threshold
res = []
Dy, Dh = effects(p, dy, 2.0, 0.3)
w = np.zeros(p); w[0], w[1] = 0.5, 1.0; w /= np.linalg.norm(w)   # a fixed score that is NOT pathway invariant (shift 0.89 sd)
for ncal in [20, 40, 80, 160, 320, 640]:
    f0, f1, f0n, f1n = [], [], [], []
    for _ in range(R * 2):
        c0, c1 = draw(ncal, 0, 0, Dy, Dh) @ w, draw(ncal, 0, 1, Dy, Dh) @ w
        t = max(conformal_quantile(c0, ALPHA), conformal_quantile(c1, ALPHA)); tn = conformal_quantile(c0, ALPHA)
        t0, t1 = draw(2000, 0, 0, Dy, Dh) @ w, draw(2000, 0, 1, Dy, Dh) @ w
        f0.append((t0 > t).mean()); f1.append((t1 > t).mean()); f0n.append((t0 > tn).mean()); f1n.append((t1 > tn).mean())
    res.append(dict(ncal=ncal, fpr_vol=np.mean(f0), fpr_hosp=np.mean(f1), fpr_worst=max(np.mean(f0), np.mean(f1)),
                    conv_fpr_vol=np.mean(f0n), conv_fpr_hosp=np.mean(f1n)))
    print('d', res[-1], flush=True)
out['conformal'] = res
RES = os.path.join(HERE, 'results') if not os.path.exists(os.path.join(HERE, '..', 'results')) else os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)
json.dump(dict(R=R, p=p, dy=dy, rho=0.3, alpha=ALPHA, **out), open(os.path.join(RES, 'theory_sim.json'), 'w'), indent=1, default=float)

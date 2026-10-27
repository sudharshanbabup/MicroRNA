"""Aggregate results/*.jsonl into summary tables (results/summary.json, printed text)."""
import json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results') if os.path.exists(os.path.join(HERE, 'results')) or not os.path.exists(os.path.join(HERE, '..', 'results')) else os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)
ORDER = ['ERM-LR', 'XGBoost', 'MLP', 'JTT', 'ERM-LR+WGcal', 'OvR', 'Matched', 'RWG', 'GroupDRO', 'IRM', 'V-REx', 'DANN', 'CORAL',
         'INLP', 'LEACE-naive', 'NICE']
AWARE = set(ORDER[4:])


def load(name):
    p = os.path.join(RES, name + '.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def pooled(recs, key_fn):
    """mean over folds within (key, rep), then mean and sd over reps"""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in recs:
        for m, v in r['res'].items():
            for k, x in v.items():
                if x is not None and not (isinstance(x, float) and np.isnan(x)):
                    acc[key_fn(r)][(m, k)][r['rep']].append(x)
    out = {}
    for key, d in acc.items():
        out[key] = {}
        for (m, k), reps in d.items():
            v = np.array([np.mean(x) for x in reps.values()])
            out[key].setdefault(m, {})[k] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))
    return out


def macro(tab, keys, m, k):
    v = [tab[s][m][k][0] for s in keys if s in tab and m in tab[s] and k in tab[s][m]]
    return float(np.mean(v)) if v else float('nan')


def shipped(m):
    return 'wg' if m in AWARE else 'nc'


if __name__ == '__main__':
    S = {}
    cv = load('cv')
    if cv:
        tab = pooled(cv, lambda r: r['study']); S['cv'] = {s: tab[s] for s in tab}
        print('CV  reps:', {s: tab[s]['NICE']['auc_nc'][2] for s in tab})
        cols = ['auc_nc', 'auc_hn', 'auc_un', 'se95hn', 'se@nc', 'sp_nc@nc', 'sp_hn@nc', 'sp_un@nc', 'wga@nc', 'se@S', 'sp_nc@S', 'sp_hn@S', 'sp_un@S', 'wga@S']
        for s in list('ABCD') + ['macro']:
            print('--', s); print('%-13s' % '', ' '.join('%8s' % c[-8:] for c in cols))
            for m in ORDER:
                row = []
                for c in cols:
                    c2 = c.replace('@S', '@' + shipped(m))
                    row.append(macro(tab, 'ABCD', m, c2) if s == 'macro' else tab[s][m].get(c2, (np.nan,))[0])
                print('%-13s' % m, ' '.join('%8.3f' % x for x in row))
        print('audit (ERM-LR):')
        for s in 'ABCD':
            e = tab[s]['ERM-LR']
            print(s, ' '.join('%s %.3f' % (k, e[k][0]) for k in ['kappa', 'pred_sp_hn', 'pred_sp_hn_eqvar', 'sp_hn@nc', 'se@nc', 'cf_se', 'audit_dy', 'audit_dh', 'audit_cos', 'audit_cos2']))
    lo = load('lonto')
    if lo:
        tab = pooled(lo, lambda r: (r['study'], r['heldout'])); S['lonto'] = {'|'.join(k): v for k, v in tab.items()}
        keys = list(tab)
        print('LONTO: %d held-out types' % len(keys), 'reps', tab[keys[0]]['NICE']['auc_un'][2])
        cols = ['auc_un', 'se95un', 'se@nc', 'sp_un@nc', 'se@S', 'sp_nc@S', 'sp_hn@S', 'sp_un@S', 'wga@S']
        print('%-13s' % '', ' '.join('%8s' % c[-8:] for c in cols))
        for m in ORDER:
            print('%-13s' % m, ' '.join('%8.3f' % macro(tab, keys, m, c.replace('@S', '@' + shipped(m))) for c in cols))
        for st in 'ABCD':
            ks = [k for k in keys if k[0] == st]
            print(' study', st, len(ks), ' | '.join('%s au %.3f spU %.3f se %.3f' % (m, macro(tab, ks, m, 'auc_un'), macro(tab, ks, m, 'sp_un@' + shipped(m)), macro(tab, ks, m, 'se@' + shipped(m))) for m in ['ERM-LR', 'OvR', 'Matched', 'RWG', 'NICE']))
    cr = load('cross')
    if cr:
        tab = pooled(cr, lambda r: r['train'] + '>' + r['test']); S['cross'] = tab
        for k in tab:
            print('CROSS', k)
            for m in ORDER:
                e = tab[k][m]; sh = shipped(m)
                print('  %-13s' % m, ' '.join('%s %.3f' % (c, e.get(c.replace('@S', '@' + sh), (np.nan,))[0]) for c in ['auc_nc', 'auc_hn', 'auc_un', 'se95hn', 'se@S', 'sp_nc@S', 'sp_hn@S', 'sp_un@S', 'wga@S']))
    ab = load('abl')
    if ab:
        tab = pooled(ab, lambda r: r['study']); S['abl'] = tab
        print('ABLATION (macro over studies)')
        for m in tab['A']:
            print('  %-22s' % m, ' '.join('%s %.3f' % (c, macro(tab, 'ABCD', m, c)) for c in ['auc_nc', 'auc_hn', 'auc_un', 'se95hn', 'se@nc', 'sp_nc@nc', 'sp_hn@nc', 'sp_un@nc', 'se@wg', 'sp_hn@wg', 'sp_un@wg', 'wga@wg']))
    for nm, key in (('mneg', 'n_neg'), ('npc', 'npc')):
        mn = load(nm)
        if mn:
            tab = pooled(mn, lambda r: (r['study'], r[key] if r[key] is not None else -1)); S[nm] = {'%s|%s' % k: v for k, v in tab.items()}
            print(nm.upper())
            for k in sorted(tab, key=lambda t: (t[0], t[1] if t[1] > 0 else 10 ** 9)):
                print(' ', k, ' | '.join('%s au %.3f spH %.3f se %.3f se95 %.3f' % (m, tab[k][m]['auc_hn'][0], tab[k][m]['sp_hn@nc'][0], tab[k][m]['se@nc'][0], tab[k][m]['se95hn'][0]) for m in tab[k]))
    json.dump(S, open(os.path.join(RES, 'summary.json'), 'w'), indent=0, default=lambda o: {str(k): v for k, v in o.items()} if isinstance(o, dict) else float(o))

"""Download the four public GEO serum-miRNA series (3D-Gene platform) and build one tidy dataset.

Series (all public, NCBI GEO):
  A  GSE59856   Kojima et al., PLoS ONE 2015      (GPL18941)
  B  GSE106817  Yokoi et al., Nat. Commun. 2018   (GPL21263)
  C  GSE113486  Usuba et al., Cancer Sci. 2019    (GPL21263)
  D  GSE124158  Asano et al., Nat. Commun. 2019   (GPL21263 part)
Output: data/mirna_dataset.npz  with
  X      float32 [n, p]  log2 intensities as deposited by the authors (no re-normalisation)
  study  <U1    [n]      A/B/C/D
  group  <U3    [n]      PC (pancreatic cancer) | NC (non-cancer control) | OC (other cancer) | BEN (benign/borderline)
  sid    str    [n]      sample label from the GEO title, gsm: GEO accession, feat: miRBase MIMAT accessions
Samples of series C whose patient label also occurs in series B are removed (re-measured sera).
"""
import gzip, os, re, sys, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data') if os.path.isdir(os.path.join(HERE, 'data')) or not os.path.isdir(os.path.join(HERE, '..', 'data')) else os.path.join(HERE, '..', 'data')
RAW = os.path.join(DATA, 'raw')
SERIES = {'A': ('GSE59856', 'GSE59856_series_matrix.txt.gz'),
          'B': ('GSE106817', 'GSE106817_series_matrix.txt.gz'),
          'C': ('GSE113486', 'GSE113486_series_matrix.txt.gz'),
          'D': ('GSE124158', 'GSE124158-GPL21263_series_matrix.txt.gz')}
NC_LABELS = {'healthy control', 'non-cancer', 'non-cancer control', 'healthy'}

def fetch(gse, fn):
    os.makedirs(RAW, exist_ok=True); p = os.path.join(RAW, fn)
    if not os.path.exists(p):
        url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/matrix/{fn}'
        print('downloading', url); urllib.request.urlretrieve(url, p)
    return p

def parse(path):
    ch = []
    with gzip.open(path, 'rt', errors='ignore') as h:
        for l in h:
            f = [x.strip('"') for x in l.rstrip('\n').split('\t')]
            if f[0] == '!Sample_title': title = f[1:]
            elif f[0] == '!Sample_geo_accession': gsm = f[1:]
            elif f[0] == '!Sample_characteristics_ch1': ch.append(f[1:])
            elif f[0] == '!series_matrix_table_begin': break
        df = pd.read_csv(h, sep='\t', index_col=0, comment='!', na_values=['null', 'NA', ''])
    meta = pd.DataFrame({'gsm': gsm, 'title': title})
    for c in ch:
        for i, v in enumerate(c):
            if ': ' in v:
                k, val = v.split(': ', 1); meta.loc[i, k.lower()] = val
    return meta, df.astype('float32')

def main():
    parts = {}
    for k, (gse, fn) in SERIES.items():
        meta, df = parse(fetch(gse, fn))
        col = [c for c in meta.columns if c in ('disease state', 'disease status', 'patient diagnosis')]
        dx = (meta[col[0]] if col else meta.title.str.replace(r'\s*[A-Z]+\d+$', '', regex=True)).str.lower()
        grp = np.where(dx.str.contains('pancreatic cancer'), 'PC',
              np.where(dx.isin(NC_LABELS), 'NC',
              np.where(dx.str.contains('benign|borderline|intermediate|others'), 'BEN', 'OC')))
        sid = meta.title.str.extract(r'([A-Z]{1,3}\d[\dA-Z_]*(?:_Cell\d)?)')[0].fillna(meta.title).values
        parts[k] = dict(meta=meta, df=df, grp=grp, sid=sid, dx=dx.values)
        print(k, gse, df.shape, pd.Series(grp).value_counts().to_dict())
    feats = sorted(set.intersection(*[set(p['df'].index) for p in parts.values()]))
    dupC = np.isin(parts['C']['sid'], parts['B']['sid'])
    print('common features', len(feats), '| C samples removed (patient label also in B):', int(dupC.sum()))
    X, study, group, sid, gsm, dx = [], [], [], [], [], []
    for k, p in parts.items():
        keep = ~dupC if k == 'C' else np.ones(len(p['grp']), bool)
        v = p['df'].loc[feats].values.T[keep]
        med = np.nanmedian(v, 0); nn = int(np.isnan(v).sum()); v = np.where(np.isnan(v), med, v)
        print(k, 'missing values imputed by feature median:', nn)
        X.append(v.astype('float32')); study += [k] * keep.sum(); group += list(p['grp'][keep]); sid += list(p['sid'][keep])
        gsm += list(p['meta'].gsm.values[keep]); dx += list(p['dx'][keep])
    X = np.vstack(X)
    np.savez_compressed(os.path.join(DATA, 'mirna_dataset.npz'), X=X, study=np.array(study), group=np.array(group),
                        sid=np.array(sid), gsm=np.array(gsm), dx=np.array(dx), feat=np.array(feats))
    print('saved', X.shape); print(pd.crosstab(np.array(study), np.array(group)))

if __name__ == '__main__':
    main()

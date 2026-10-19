"""Escribe una entrada por estrella (13<Tmag<=15) con los pares TIC-sector que aún no tienen .dat."""
import os, re, sys, pandas as pd
base = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(os.path.join(base, '../membresia/pleyades_tessextractor_miembros.txt'),
                sep=r'\s+', comment='#', names=['TIC', 'RA', 'Dec', 'Sector', 'Tmag'])
dat = {tuple(map(int, re.match(r'TIC_(\d+)_S_(\d+)_LC', f).groups()))
       for f in os.listdir(os.path.join(base, 'resultados/dat')) 
       if os.path.getsize(os.path.join(base, 'resultados/dat', f)) > 1000}
m = m[(m.Tmag > 13) & (m.Tmag <= 15)]
m = m[[(t, s) not in dat for t, s in zip(m.TIC, m.Sector)]].sort_values(['TIC', 'Sector'])
d = os.path.join(base, 'entradas/cola_13_15'); os.makedirs(d, exist_ok=True)
for f in os.listdir(d): os.remove(os.path.join(d, f))
for t, g in m.groupby('TIC'):
    with open(os.path.join(d, 'TIC_%d.txt' % t), 'w') as fh:
        fh.write('# TIC       Tmag   Sector\n')
        for r in g.itertuples(): fh.write('  %-9d %-6.3f %-4d\n' % (r.TIC, r.Tmag, r.Sector))
print('pares pendientes:', len(m), 'estrellas:', m.TIC.nunique())

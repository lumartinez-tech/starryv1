import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
tz=np.trapezoid
def load(pref,t):
    d=np.loadtxt(f'{pref}_{t}_massage.csv',delimiter=',',skiprows=1)
    a=d[:,1]; return a[np.isfinite(a)&(a>0)]
casos={'A: Teff de TIC (4 estrellas)':[('mc_mist',26017840),('mc_mist',35159657),('mc_mist',125754991),('mc_mist',346541355)],
       'B: clase espectral donde existe':[('mc_mist',26017840),('mc_SpT_mist',35159657),('mc_SpT_mist',125754991),('mc_SpT_mist',346541355)]}
g=np.linspace(0,3,1500); x=10**g
fig,ax=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
for k,(nom,lst) in enumerate(casos.items()):
    logp=np.zeros_like(g); print('\n',nom); ms=[]
    for pref,t in lst:
        a=load(pref,t); p=gaussian_kde(np.log10(a))(g); p/=tz(p,g)
        print(t,'N=%d media %.0f mediana %.0f sd %.0f; 16-84%%: %.0f-%.0f Myr'%(len(a),a.mean(),np.median(a),a.std(),*np.percentile(a,[16,84])))
        ax[k].plot(x,p,lw=1.3,label=str(t)); logp+=np.log(p+1e-300); ms.append((a.mean(),a.std()))
    P=np.exp(logp-logp.max()); P/=tz(P,g)
    cdf=np.cumsum(P)*(g[1]-g[0]); cdf/=cdf[-1]; q=lambda f:10**np.interp(f,cdf,g)
    print('COEVAL (producto): media %.0f, mediana %.0f, moda %.0f; 16-84%%: %.0f-%.0f; 2.5-97.5%%: %.0f-%.0f Myr'%(tz(x*P,g),q(.5),x[P.argmax()],q(.16),q(.84),q(.025),q(.975)))
    m=np.array(ms); w=1/m[:,1]**2; print('promedio ponderado de medias: %.0f ± %.0f Myr'%((w*m[:,0]).sum()/w.sum(),w.sum()**-.5))
    ax[k].plot(x,P,'k',lw=2.5,label='combinada (coeval)')
    ax[k].axvline(125,color='r',ls='--',label='Pléyades ~125 Myr'); ax[k].set_xscale('log'); ax[k].set_title(nom); ax[k].set_xlabel('edad (Myr)')
ax[0].set_ylabel('densidad en log(edad)'); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig('coevalidad_edades.png',dpi=130)

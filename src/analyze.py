import os, pandas as pd, numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
M = ['viol_urllc','viol_wavg','viol_all','p99_urllc','util','goodput','jain','dec_us']
main = pd.read_csv('res_main.csv')
print('== MAIN mean over seeds, per dataset (viol_urllc / viol_wavg / util / p99)')
for m in ['viol_urllc','viol_wavg','util','p99_urllc']:
    print(m); print(main.pivot_table(index='method', columns='ds', values=m, aggfunc='mean').round(3).assign(ALL=main.groupby('method')[m].mean().round(3)))
print('== sd of viol_urllc/wavg overall'); print(main.groupby('method')[['viol_urllc','viol_wavg','util','p99_urllc','goodput','jain','dec_us']].agg(['mean','std']).round(3))

def holm(p):
    p=np.array(p); o=np.argsort(p); n=len(p); adj=np.empty(n); run=0
    for k,i in enumerate(o): run=max(run,(n-k)*p[i]); adj[i]=min(1,run)
    return adj
def cliff(a,b):
    a=np.asarray(a);b=np.asarray(b); return (np.sum(a[:,None]>b[None,:])-np.sum(a[:,None]<b[None,:]))/(len(a)*len(b))
rows=[]
for met in ['viol_urllc','viol_wavg','util','p99_urllc']:
    pw=main[main.method=='Proposed'].sort_values(['ds','seed'])[met].values
    ps=[];tmp=[]
    for b in ['B1 Static hard','B2 Threshold soft','B3 Backlog-weighted','B4 Q-learning']:
        bw=main[main.method==b].sort_values(['ds','seed'])[met].values
        try: w=stats.wilcoxon(pw,bw); p=w.pvalue
        except Exception: p=1.0
        ps.append(p); tmp.append((met,b,pw.mean(),bw.mean(),cliff(pw,bw)))
    adj=holm(ps)
    for t,p,a in zip(tmp,ps,adj): rows.append(dict(metric=t[0],vs=t[1],prop=round(t[2],4),base=round(t[3],4),cliffs_delta=round(t[4],3),p=p,p_holm=a))
print('== Wilcoxon (pooled ds x seed, n=150)'); print(pd.DataFrame(rows).to_string())

piv = main.pivot_table(index=['ds','seed'], columns='method', values='viol_wavg')
print('Friedman viol_wavg', stats.friedmanchisquare(*[piv[c].values for c in piv.columns]))
piv = main.pivot_table(index=['ds','seed'], columns='method', values='viol_urllc')
print('Friedman viol_urllc', stats.friedmanchisquare(*[piv[c].values for c in piv.columns]))

rng=np.random.default_rng(0)
for b in ['B1 Static hard','B2 Threshold soft','B3 Backlog-weighted','B4 Q-learning']:
    d = main[main.method=='Proposed'].sort_values(['ds','seed'])['viol_wavg'].values - main[main.method==b].sort_values(['ds','seed'])['viol_wavg'].values
    bs=[rng.choice(d,len(d)).mean() for _ in range(2000)]
    print('diff viol_wavg Prop-',b, round(d.mean(),4), np.percentile(bs,[2.5,97.5]).round(4))

print('== per dataset proposed vs best baseline (viol_wavg)')
pv = main.pivot_table(index='ds', columns='method', values='viol_wavg').round(3); print(pv)
cond = pd.read_csv('res_cond.csv')
print('== COND'); print(cond.pivot_table(index='method', columns=['cond','ds'], values='viol_wavg').round(3)); print(cond.pivot_table(index='method', columns=['cond','ds'], values='viol_urllc').round(3)); print(cond.pivot_table(index='method', columns=['cond','ds'], values='util').round(3))
ab = pd.read_csv('res_ablate.csv')
print('== ABLATION'); 
for m in ['viol_urllc','viol_wavg','util','p99_urllc']:
    print(m); print(ab.pivot_table(index='method', columns=['cond','ds'], values=m).round(3).assign(MEAN=ab.groupby('method')[m].mean().round(3)))
full = ab[ab.method=='Full'].sort_values(['cond','ds','seed'])
for v in ab.method.unique():
    if v=='Full': continue
    o = ab[ab.method==v].sort_values(['cond','ds','seed'])
    for met in ['viol_wavg','viol_urllc']:
        try: p=stats.wilcoxon(full[met].values,o[met].values).pvalue
        except Exception: p=1
        print('ablation Full vs',v,met,round(full[met].mean(),4),round(o[met].mean(),4),'p=',p)
sw = pd.read_csv('res_sweep.csv')
print('== SWEEP viol_wavg'); print(sw.pivot_table(index='rho', columns='method', values='viol_wavg').round(3))
print('SWEEP viol_urllc'); print(sw.pivot_table(index='rho', columns='method', values='viol_urllc').round(3))
print('SWEEP util'); print(sw.pivot_table(index='rho', columns='method', values='util').round(3))
sc = pd.read_csv('res_scale.csv')
print('== SCALE viol_wavg'); print(sc.pivot_table(index='N', columns='method', values='viol_wavg').round(3))
print('SCALE viol_urllc'); print(sc.pivot_table(index='N', columns='method', values='viol_urllc').round(3))
print('SCALE util'); print(sc.pivot_table(index='N', columns='method', values='util').round(3))
print('SCALE dec_us'); print(sc.pivot_table(index='N', columns='method', values='dec_us').round(1))

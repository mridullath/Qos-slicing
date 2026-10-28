import sys, numpy as np, pandas as pd, time, pickle, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim
sim.load_real()
os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
os.chdir(os.path.join(HERE, '..', 'results'))  # outputs (res_*.csv) and caches go to results/
MODELS = sim.train_forecaster(seeds=range(1000,1008))
pickle.dump(MODELS, open('models.pkl','wb'))
if os.path.exists('qtab.npy'):
    QL = sim.QLearn(); QL.Qt = np.load('qtab.npy')
else:
    QL = sim.train_qlearn(episodes=60); np.save('qtab.npy', QL.Qt)
KW = dict(z_crit=0.0, z_other=0.0, floor=0.5)
def methods(include_q=True):
    m = {'B1 Static hard': sim.Static, 'B2 Threshold soft': lambda: sim.Threshold(0.2,0.9), 'B3 Backlog-weighted': sim.Backlog}
    if include_q: m['B4 Q-learning'] = lambda: QL
    m['Proposed'] = lambda: sim.Proposed(MODELS, **KW)
    return m
def run_grid(tag, seeds, N, rhos, dss, conds, meths, obs_noise=0.0, out=None):
    rows=[]
    for cond in conds:
        for ds in dss:
            for rho in rhos:
                for sd in seeds:
                    scn = sim.gen_scenario(sd, N, rho, ds, 'degraded' if cond=='degraded' else 'nominal')
                    on = 0.3 if cond=='noisy' else 0.0
                    for mn, mk in meths.items():
                        r = sim.run(scn, mk(), obs_noise=on, noise_seed=sd)
                        r.update(exp=tag, method=mn, seed=sd, N=N, rho=rho, ds=ds, cond=cond); rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(out, index=False); return df
if __name__=='__main__':
    part = sys.argv[1]; t0=time.time()
    S30 = range(0,30)
    if part=='main':
        run_grid('main', S30, 3, [0.65], ['D1','D2','D3','D4','D5'], ['nominal'], methods(), out='res_main.csv')
    elif part=='cond':
        run_grid('cond', S30, 3, [0.65], ['D1','D4'], ['degraded','noisy'], methods(), out='res_cond.csv')
    elif part=='sweep':
        run_grid('sweep', range(0,15), 3, [0.4,0.5,0.6,0.7,0.8,0.9,1.0], ['D1','D4'], ['nominal'], methods(), out='res_sweep.csv')
    elif part=='scale':
        rows=[]
        for N in [3,6,12,25,50]:
            rows.append(run_grid('scale', range(0,10), N, [0.65], ['D1'], ['nominal'], methods(False), out=f'res_scale_{N}.csv'))
        pd.concat(rows).to_csv('res_scale.csv', index=False)
    elif part=='ablate':
        ab = {'Full': lambda: sim.Proposed(MODELS, **KW),
              'w/o cross-layer': lambda: sim.Proposed(MODELS, cl=False, **KW),
              'w/o forecasting': lambda: sim.Proposed(MODELS, pred=False, **KW),
              'w/o hybrid isolation': lambda: sim.Proposed(MODELS, hybrid=False, **KW),
              'hybrid only': lambda: sim.Proposed(MODELS, cl=False, pred=False, **KW),
              'none (soft, reactive)': lambda: sim.Proposed(MODELS, cl=False, pred=False, hybrid=False, **KW)}
        run_grid('ablate', S30, 3, [0.65], ['D1','D4','D5'], ['nominal','degraded'], ab, out='res_ablate.csv')
    print(part, 'done', round(time.time()-t0,1), 's')

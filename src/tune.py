import sys, os, numpy as np, pandas as pd, itertools, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
sim.load_real()
MODELS = sim.train_forecaster(seeds=range(1000,1006))
TUNE = range(3000,3003); RHOS=(0.5,0.65,0.8); DS=('D1','D2','D3','D4','D5')
def ev(mk):
    out=[sim.run(sim.gen_scenario(sd,3,rho,d), mk()) for rho in RHOS for d in DS for sd in TUNE]
    m=pd.DataFrame(out).mean(); return float(m['viol_wavg']),float(m['viol_urllc']),float(m['p99_urllc']),float(m['util'])
which = sys.argv[1]
t0=time.time()
if which=='B2':
    for st,hi in itertools.product([0.05,0.1,0.2],[0.8,0.9]):
        print('B2',st,hi,[round(x,4) for x in ev(lambda: sim.Threshold(st,hi))],flush=True)
else:
    fl_list = [float(x) for x in sys.argv[2].split(',')]
    for fl in fl_list:
        for zc,zo in itertools.product([0,0.5,1.0],[0,0.5]):
            print('P',zc,zo,fl,[round(x,4) for x in ev(lambda: sim.Proposed(MODELS,z_crit=zc,z_other=zo,floor=fl))],round(time.time()-t0),flush=True)

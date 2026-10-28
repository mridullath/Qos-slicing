import numpy as np, glob, time, os
import pandas as pd

T_E = 5.0          # decision epoch (ms)
T_STEPS = 3000     # epochs per run
WARM = 50
QMAX_FACTOR = 10.0
CLS = {0: dict(name='URLLC', prio=3.0, D=10.0, share=0.15, W=0.30),
       1: dict(name='eMBB', prio=1.0, D=100.0, share=0.60, W=0.50),
       2: dict(name='mMTC', prio=1.0, D=200.0, share=0.25, W=0.20)}

# ---------------- real trace loading ----------------
REAL = []
def load_real(root=None):
    global REAL
    REAL = []
    root = root or os.environ.get('QOS_DATA') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', '5Gdataset', 'data')
    for f in sorted(glob.glob(root + '/**/*.csv', recursive=True)):
        try:
            df = pd.read_csv(f, encoding='latin-1', usecols=['DL_bitrate', 'CQI'])
        except Exception:
            continue
        dl = pd.to_numeric(df['DL_bitrate'], errors='coerce').ffill().bfill().values
        cq = pd.to_numeric(df['CQI'], errors='coerce').ffill().bfill().values
        if len(dl) >= 200 and np.isfinite(dl).all() and np.isfinite(cq).all():
            REAL.append((dl.astype(float), cq.astype(float)))
    return len(REAL)

def real_segment(rng, n_samples=T_STEPS // 5):
    dl, cq = REAL[rng.integers(len(REAL))]
    st = rng.integers(0, len(dl))
    idx = (st + np.arange(n_samples)) % len(dl)
    d, c = dl[idx], cq[idx]
    rep = T_STEPS // n_samples
    xs = np.arange(n_samples)
    xi = np.linspace(0, n_samples - 1, T_STEPS)
    return np.interp(xi, xs, d), np.interp(xi, xs, c)

# ---------------- scenario generation ----------------
def class_layout(N):
    if N == 3:
        return np.array([0, 1, 2])
    nu = max(1, round(0.2 * N)); nm = max(1, round(0.3 * N)); ne = N - nu - nm
    return np.array([0] * nu + [1] * ne + [2] * nm)

def markov_onoff(rng, T, p_on, stay_on, burst):
    p10 = 1 - stay_on
    p01 = p_on * p10 / (1 - p_on)
    s = np.zeros(T, bool); cur = rng.random() < p_on
    u = rng.random(T)
    for t in range(T):
        s[t] = cur
        cur = (u[t] < stay_on) if cur else (u[t] < p01)
    lam_off = 1.0 / (1 - p_on + burst * p_on)
    return np.where(s, burst * lam_off, lam_off)   # mean 1

def pareto_agg(rng, T, n_src=16, alpha=1.4):
    out = np.zeros(T)
    for _ in range(n_src):
        t = 0; on = rng.random() < 0.5
        while t < T:
            L = int(min(T, (rng.pareto(alpha) + 1) * 8))
            if on: out[t:t + L] += 1
            t += L; on = not on
    m = out.mean()
    return out / m if m > 0 else np.ones(T)

def ar1(rng, T, phi, std):
    x = np.zeros(T); s = std * np.sqrt(1 - phi ** 2)
    e = rng.normal(0, s, T)
    x[0] = rng.normal(0, std)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x

def gen_scenario(seed, N=3, rho=0.8, dataset='D1', cond='nominal', burst_u=4.0):
    rng = np.random.default_rng(seed)
    cl = class_layout(N)
    C = 100.0 * N / 3.0
    counts = {k: (cl == k).sum() for k in CLS}
    mean = np.array([rho * C * CLS[c]['share'] / counts[c] for c in cl])
    prio = np.array([CLS[c]['prio'] for c in cl]); D = np.array([CLS[c]['D'] for c in cl])
    w = np.array([CLS[c]['W'] / counts[c] for c in cl]); w = w / w.sum()
    T = T_STEPS
    lam = np.zeros((T, N)); eta = np.zeros((T, N))
    p_fade = {'nominal': 0.002, 'degraded': 0.010}.get(cond, 0.002)
    depth = 0.35 if cond == 'degraded' else 0.45
    for i, c in enumerate(cl):
        base_eta = None
        if c == 0:
            f = markov_onoff(rng, T, 0.15, 0.9, burst_u)
        elif c == 1:
            if dataset == 'D1':
                ph = rng.uniform(0, 2 * np.pi)
                f = 1 + 0.4 * np.sin(2 * np.pi * np.arange(T) / 600 + ph) + ar1(rng, T, 0.95, 0.12)
            elif dataset == 'D2':
                f = markov_onoff(rng, T, 0.3, 0.98, 3.0)
            elif dataset == 'D3':
                f = pareto_agg(rng, T)
            elif dataset == 'D4':
                dl, cq = real_segment(rng)
                f = np.maximum(dl, 1e-3) / np.mean(np.maximum(dl, 1e-3))
                base_eta = np.clip(0.35 + 0.75 * cq / 15.0, 0.3, 1.2)
            elif dataset == 'D5':
                ph = rng.uniform(0, 2 * np.pi)
                f = 1 + 0.4 * np.sin(2 * np.pi * np.arange(T) / 600 + ph) + ar1(rng, T, 0.95, 0.12)
                t0 = rng.integers(600, 2200); f[t0:t0 + 150] *= 2.5
        else:
            ph = rng.integers(0, 100)
            spk = (((np.arange(T) + ph) % 100) < 10).astype(float)
            f = (0.7 + 2.0 * spk) / 0.9
            if dataset == 'D5':
                t0 = rng.integers(600, 2200); f[t0:t0 + 150] *= 1.8
        f = np.clip(f, 0.02, None); f = f / f.mean()
        lam[:, i] = mean[i] * f
        if base_eta is None:
            x = ar1(rng, T, 0.97, 0.2)
            e = np.clip(0.9 * np.exp(x), 0.3, 1.3)
        else:
            x = ar1(rng, T, 0.9, 0.08)
            e = np.clip(base_eta * np.exp(x), 0.3, 1.3)
        # fades
        t = 0
        while t < T:
            if rng.random() < p_fade:
                L = int(rng.geometric(1 / 25.0)); e[t:t + L] *= depth; t += L
            else:
                t += 1
        eta[:, i] = np.clip(e, 0.15, 1.3)
    A = rng.poisson(lam).astype(float)
    bler = np.minimum(0.4, 0.01 / eta ** 3)
    return dict(A=A, eta=eta, bler=bler, cls=cl, C=C, prio=prio, D=D, w=w, mean=mean, N=N, seed=seed)

# ---------------- simulation core ----------------
def run(scn, policy, obs_noise=0.0, noise_seed=0, learn=False):
    A, eta, bler = scn['A'], scn['eta'], scn['bler']
    C, D, prio, mean, cl = scn['C'], scn['D'], scn['prio'], scn['mean'], scn['cls']
    T, N = A.shape
    Qmax = mean * 4.0 * D / T_E   # deadline-based discard: work older than ~4x its delay target is dropped
    nrng = np.random.default_rng(noise_seed + 7919)
    noiseA = np.exp(nrng.normal(0, obs_noise, (T, N))) if obs_noise > 0 else np.ones((T, N))
    noiseE = np.exp(nrng.normal(0, obs_noise, (T, N))) if obs_noise > 0 else np.ones((T, N))
    policy.reset(scn)
    Q = np.zeros(N); r = scn['C'] * scn['w']
    viol = np.zeros((T, N), bool); has = np.zeros((T, N), bool); dly = np.zeros((T, N))
    used_tot = np.zeros(T); served_tot = np.zeros(T); arr_s = np.zeros(N); srv_s = np.zeros(N)
    dec_time = 0.0
    for t in range(T):
        mu = r * eta[t] * (1 - bler[t])
        tot = Q + A[t]
        served = np.minimum(mu, tot)
        Qn = tot - served
        drop = np.maximum(Qn - Qmax, 0.0); Qn = Qn - drop
        d = T_E * np.maximum(0.5, tot / np.maximum(mu, 1e-6)); d = np.minimum(d, 2000.0)
        h = tot > 1e-9
        v = h & ((d > D) | (drop > 1e-9))
        need_rb = tot / np.maximum(eta[t] * (1 - bler[t]), 1e-6)
        used = np.minimum(r, need_rb)
        viol[t] = v; has[t] = h; dly[t] = np.where(h, d, 0.0)
        used_tot[t] = used.sum() / C; served_tot[t] = served.sum()
        arr_s += A[t]; srv_s += served
        obs = dict(t=t, Q=Qn, A=A[t] * noiseA[t], eta=eta[t] * noiseE[t], bler=np.minimum(0.4, 0.01 / np.maximum(eta[t] * noiseE[t], 0.15) ** 3),
                   used=used, r_prev=r, served=served, viol=v, has=h, mu=mu, d=d)
        t0 = time.perf_counter()
        r_new = policy.decide(obs, learn=learn)
        dec_time += time.perf_counter() - t0
        r = np.maximum(r_new, 0.0)
        if r.sum() > C * (1 + 1e-9): r = r * C / r.sum()
        Q = Qn
    sl = slice(WARM, T)
    hv, vv, dd = has[sl], viol[sl], dly[sl]
    per_slice_viol = np.array([vv[:, i][hv[:, i]].mean() if hv[:, i].any() else 0.0 for i in range(N)])
    u = (cl == 0)
    urllc_d = dd[:, u][hv[:, u]]
    ratio = srv_s / np.maximum(arr_s, 1e-9)
    jain = (ratio.sum() ** 2) / (N * (ratio ** 2).sum())
    return dict(viol_urllc=per_slice_viol[u].mean(), viol_embb=per_slice_viol[cl==1].mean(), viol_mmtc=per_slice_viol[cl==2].mean(), viol_all=per_slice_viol.mean(),
                viol_wavg=float(np.average(per_slice_viol, weights=prio)),
                p99_urllc=float(np.percentile(urllc_d, 99)) if len(urllc_d) else 0.0,
                util=float(used_tot[sl].mean()), goodput=float(served_tot[sl].mean()), jain=float(jain),
                dec_us=1e6 * dec_time / T)

# ---------------- policies ----------------
class Static:
    name = 'B1 Static hard'
    def reset(self, scn): self.C = scn['C']; self.w = scn['w']
    def decide(self, o, learn=False): return self.C * self.w

class Threshold:
    name = 'B2 Threshold soft'
    def __init__(self, step=0.05, hi=0.9):
        self.step = step; self.hi = hi
    def reset(self, scn):
        self.C = scn['C']; self.w = scn['w'].copy(); self.x = scn['w'].copy()
    def decide(self, o, learn=False):
        util = o['used'] / np.maximum(o['r_prev'], 1e-9)
        need = (util > self.hi) | (o['Q'] > 1.0)
        donor = (util < 0.5) & (o['Q'] <= 1.0)
        if need.any() and donor.any():
            floor = 0.5 * self.w
            avail = np.maximum(self.x - floor, 0) * donor
            m = min(self.step, avail.sum())
            if m > 0:
                self.x = self.x - m * avail / avail.sum()
                self.x[need] += m / need.sum()
        return self.C * self.x / self.x.sum()

class Backlog:
    name = 'B3 Backlog-weighted'
    def reset(self, scn): self.C = scn['C']; self.prio = scn['prio']
    def decide(self, o, learn=False):
        p = self.prio * (o['Q'] + o['A']) + 1e-6
        return 0.01 * self.C / len(p) + 0.99 * self.C * p / p.sum()

PRESETS = np.array([[.2, .55, .25], [.3, .5, .2], [.4, .45, .15], [.5, .35, .15], [.25, .6, .15], [.3, .3, .4]])
class QLearn:
    name = 'B4 Tabular Q-learning'
    def __init__(self):
        self.Qt = np.zeros((27, len(PRESETS))); self.eps = 0.0; self.alpha = 0.1; self.gamma = 0.9
        self.rng = np.random.default_rng(0)
    def reset(self, scn):
        self.C = scn['C']; self.s = None; self.a = None
    def state(self, o):
        r = np.maximum(o['r_prev'], 1.0)
        x = (o['Q'] + o['A']) / r
        b0 = 0 if x[0] < 0.5 else (1 if x[0] < 1.5 else 2)
        b1 = 0 if x[1] < 0.8 else (1 if x[1] < 1.2 else 2)
        b2 = 0 if x[2] < 0.8 else (1 if x[2] < 1.2 else 2)
        return b0 * 9 + b1 * 3 + b2
    def decide(self, o, learn=False):
        s = self.state(o)
        if learn and self.s is not None:
            v = o['viol'].astype(float)
            reward = -(3 * v[0] + v[1] + v[2]) - 0.3 * (1 - o['used'].sum() / self.C)
            self.Qt[self.s, self.a] += self.alpha * (reward + self.gamma * self.Qt[s].max() - self.Qt[self.s, self.a])
        a = self.rng.integers(len(PRESETS)) if (learn and self.rng.random() < self.eps) else int(np.argmax(self.Qt[s]))
        self.s, self.a = s, a
        return self.C * PRESETS[a]

class Proposed:
    def __init__(self, models, cl=True, pred=True, hybrid=True, z_crit=0.0, z_other=0.0, floor=0.5, name='Proposed'):
        self.models = models; self.cl = cl; self.pred = pred; self.hybrid = hybrid
        self.z_crit = z_crit; self.z_other = z_other; self.floor = floor; self.name = name
    def reset(self, scn):
        self.C = scn['C']; self.prio = scn['prio']; self.D = scn['D']; self.mean = scn['mean']; self.cls = scn['cls']
        self.P = self.models['p']
        self.coef = np.array([self.models['coef'][c] for c in self.cls])
        self.b = np.array([self.models['b'][c] for c in self.cls])
        self.sig = np.array([self.models['sig'][c] for c in self.cls]) * self.mean
        self.hist = np.ones((self.P, scn['N'])) * self.mean
        self.z = np.where(self.prio >= 3, self.z_crit, self.z_other)
        self.k = np.maximum(1.0, self.D / T_E - 1.0)
        self.fl = np.where(self.prio >= 3, self.floor * self.C * scn['w'], 0.0)
    def decide(self, o, learn=False):
        self.hist = np.vstack([self.hist[1:], o['A']])
        if self.pred:
            x = self.hist / self.mean
            Ah = self.mean * ((x.T * self.coef).sum(1) + self.b)
            Ah = np.maximum(Ah, 0)
        else:
            Ah = o['A']
        rho = np.maximum(o['eta'] * (1 - o['bler']), 0.25) if self.cl else np.ones_like(o['A'])
        B = o['Q']
        mu_req = Ah + B / self.k + self.z * self.sig
        rreq = np.maximum(mu_req / rho, self.fl) if self.hybrid else mu_req / rho
        est_delay = T_E * (B + Ah) / np.maximum(rho * o['r_prev'], 1e-6)
        slack = est_delay / self.D
        crit = (self.prio >= 3) | (slack > 0.5)
        C = self.C
        alloc = np.zeros_like(rreq)
        if self.hybrid:
            idx = np.where(crit)[0]
            # priority tier first, then closeness to deadline violation
            key = self.prio[idx] * 1000.0 + np.minimum(slack[idx], 100.0)
            idx = idx[np.argsort(-key)]
            # step 1: guaranteed isolated reserve (floor) for top-priority critical slices
            gfl = np.where(self.prio >= 3, np.minimum(self.fl, rreq), 0.0)
            alloc = gfl.copy(); rem = C - alloc.sum()
            # step 2: remaining demand of critical slices in criticality order
            for i in idx:
                a = min(max(rreq[i] - alloc[i], 0.0), rem); alloc[i] += a; rem -= a
            oth = np.where(~crit)[0]
            if len(oth):
                s = rreq[oth].sum()
                alloc[oth] = rreq[oth] if s <= rem else rem * rreq[oth] / s
        else:
            s = rreq.sum()
            alloc = rreq if s <= C else C * rreq / s
        left = C - alloc.sum()
        if left > 1e-9:
            alloc = alloc + left * (rreq + 1e-9) / (rreq + 1e-9).sum()
        return alloc

# ---------------- forecaster training ----------------
def train_forecaster(datasets=('D1', 'D2', 'D3', 'D4', 'D5'), seeds=range(1000, 1008), p=8, lam=1e-2):
    X = {0: [], 1: [], 2: []}; Y = {0: [], 1: [], 2: []}
    for ds in datasets:
        for sd in seeds:
            scn = gen_scenario(sd, 3, 0.8, ds)
            A = scn['A']; m = scn['mean']
            for c in range(3):
                a = A[:, c] / m[c]
                for t in range(p, len(a)):
                    X[c].append(a[t - p:t]); Y[c].append(a[t])
    coef, b, sig = {}, {}, {}
    for c in range(3):
        x = np.array(X[c]); y = np.array(Y[c])
        xm = np.hstack([x, np.ones((len(x), 1))])
        w = np.linalg.solve(xm.T @ xm + lam * np.eye(p + 1), xm.T @ y)
        coef[c] = w[:p]; b[c] = w[p]
        sig[c] = float(np.std(y - xm @ w))
    return dict(p=p, coef=coef, b=b, sig=sig)

def train_qlearn(episodes=60, datasets=('D1', 'D2', 'D3', 'D4', 'D5')):
    ql = QLearn(); rng = np.random.default_rng(1)
    for ep in range(episodes):
        ql.eps = max(0.02, 0.3 * (1 - ep / episodes))
        ds = datasets[ep % len(datasets)]
        scn = gen_scenario(2000 + ep, 3, 0.8, ds)
        run(scn, ql, learn=True)
    ql.eps = 0.0
    return ql

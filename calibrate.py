"""Step 1 -- calibrate the risk boundary from independent baseline episodes.

Each of N episodes is an independent run of the NOMINAL controller only (no supervisor):
random safe initial condition (time-headway start), normal lead driving
(lead acceleration piecewise constant, redrawn every 3 s from U(-2, 1.5) m/s^2),
uniform disturbance d_v ~ U(-DELTA_V, DELTA_V). One state is sampled per episode at a
uniformly random time, so the N risk samples are i.i.d. by construction
(Assumption 2 holds exactly; no temporal subsampling or mixing argument needed).

Output: calibration.json  (B = empirical 0.95-quantile, eps_N, B_hat_plus, margins in metres)
Run:    python calibrate.py
"""
import json, numpy as np, model as M

SEED, N, T_EP, ALPHA, BETA = 1000, 3000, 60.0, 0.05, 0.05

def main():
    rng = np.random.default_rng(SEED)
    n = int(T_EP/M.DT)
    v  = rng.uniform(5, 30, N)
    vl = np.clip(v + rng.uniform(-3, 3, N), 0, 30)
    s  = 5 + 1.5*v + rng.uniform(0, 30, N)
    t_sample = rng.integers(0, n, N)
    samples = np.full(N, np.nan)
    collided = np.zeros(N, bool)
    wl = rng.uniform(-2.0, 1.5, N)
    for i in range(n):
        if i % 300 == 0:
            wl = rng.uniform(-2.0, 1.5, N)
        hit = (t_sample == i)
        samples[hit] = M.risk(s[hit], v[hit], vl[hit])
        u  = M.u_nominal(s, v, vl)
        dv = rng.uniform(-M.DELTA_V, M.DELTA_V, N)
        wl_eff = np.where((vl >= 30) & (wl > 0), 0.0, wl)
        s, v, vl = M.step(s, v, vl, u, dv, wl_eff)
        collided |= (s <= 0)
    B     = float(np.quantile(samples, 1-ALPHA, method="inverted_cdf"))
    eps   = float(np.sqrt(np.log(2/BETA)/(2*N)))
    Bplus = float(np.quantile(samples, 1-ALPHA+eps, method="inverted_cdf"))
    out = dict(seed=SEED, N=N, episode_length_s=T_EP, alpha=ALPHA, beta=BETA,
               B=B, eps_N=eps, B_hat_plus=Bplus,
               margin_at_B_m=float(M.margin_of_level(B)),
               margin_at_B_hat_plus_m=float(M.margin_of_level(Bplus)),
               sample_quantiles={str(q): float(np.quantile(samples, q)) for q in (0.5, 0.9, 0.95, 0.99)},
               baseline_collision_fraction=float(collided.mean()),
               sampling="one state per independent episode at a uniformly random time")
    json.dump(out, open("calibration.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()

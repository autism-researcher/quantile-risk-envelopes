"""Double integrator: calibrate -> analytic certificate (+ grid cross-check) -> simulate.
Run: python di_run.py   -> di_results.json, di_mc_*.csv, di_traj_*.csv
"""
import json, numpy as np, pandas as pd, di_model as D

TAU = 0.5; N = 3000; ALPHA = BETA = 0.05; T_EP = 30.0

def calibrate(tau, seed=1000):
    """One state per independent episode; nominal PD tracking random set-points
    r ~ U(0, X_BAR - 2) (normal operation, away from the wall); w ~ U(-DELTA, DELTA)."""
    rng = np.random.default_rng(seed); n = int(T_EP/D.DT)
    x1 = rng.uniform(0, 6, N); x2 = rng.uniform(-1, 1, N)
    r = rng.uniform(0, D.X_BAR - 2, N); ts = rng.integers(0, n, N); out = np.empty(N)
    for i in range(n):
        if i % 500 == 0: r = rng.uniform(0, D.X_BAR - 2, N)
        hit = ts == i; out[hit] = D.risk_grad(x1[hit], x2[hit], tau)[0]
        x1, x2 = D.step(x1, x2, D.u_nominal(x1, x2, r), rng.uniform(-D.DELTA, D.DELTA, N))
    eps = np.sqrt(np.log(2/BETA)/(2*N))
    return dict(B=float(np.quantile(out, 1-ALPHA, method="inverted_cdf")), eps_N=float(eps),
                B_hat_plus=float(np.quantile(out, 1-ALPHA+eps, method="inverted_cdf")))

def grid_check(b, tau, k0, n=20001, vmax=20.0):
    """Cross-check of the analytic constants on {R=b}, x2 in [-vmax, vmax]."""
    x2 = np.linspace(-vmax, vmax, n)
    x1 = D.X_BAR - tau*x2 - x2*np.abs(x2)/(2*D.BRK) - D.h_of_level(b)
    R, R1, a = D.risk_grad(x1, x2, tau)
    dist = np.abs(a)*D.DELTA
    feas = R1*x2 - np.abs(a)*D.U_BAR + dist
    worst = -np.inf
    for r in (-50.0, 0.0, 50.0):                    # nominal set-points far either side
        u = np.clip(D.u_nominal(x1, x2, r) - k0*a, -D.U_BAR, D.U_BAR)
        worst = max(worst, float(np.max(R1*x2 + a*u + dist)))
    return dict(max_abs_R_minus_b=float(np.max(np.abs(R - b))), eta_grid=float(-feas.max()),
                a_min_grid=float(np.abs(a).min()), closed_loop_worst=worst)

def simulate(name, B, tau, k0, x1, x2, rfun, wfun, T=40.0, record=False):
    n = int(T/D.DT); maxdev = np.full(len(x1), -np.inf); rows = []; viol_wall = np.zeros(len(x1), bool)
    for i in range(n):
        r = rfun(i)
        if name == "nominal": u = D.u_nominal(x1, x2, r)
        elif name == "proposed": u = D.u_supervisor(x1, x2, r, B, tau, k0)[0]
        else: u = D.u_cbf_robust(x1, x2, r, B, tau)[0]
        R = D.risk_grad(x1, x2, tau)[0]; maxdev = np.maximum(maxdev, R - B)
        if record: rows.append([i*D.DT, x1[0], x2[0], R[0], u[0]])
        x1, x2 = D.step(x1, x2, u, wfun(i)); viol_wall |= x1 > D.X_BAR
    return maxdev, viol_wall, rows

def main():
    res = {"parameters": dict(X_BAR=D.X_BAR, BRK=D.BRK, ELL=D.ELL, U_BAR=D.U_BAR, DELTA=D.DELTA,
                              TAU=TAU, dt=D.DT, N=N, alpha=ALPHA, beta=BETA)}
    for tau, tag in ((TAU, "tau_pos"), (0.0, "tau_zero")):
        cal = calibrate(tau); B = cal["B_hat_plus"]
        an = D.analytic_constants(B, tau)
        k0 = int(np.ceil(an["k_bar"])) if np.isfinite(an["k_bar"]) else 200   # tau=0: no finite certified gain
        entry = dict(calibration=cal, analytic=an, k0_used=k0,
                     k0_note="ceil(k_bar)" if np.isfinite(an["k_bar"]) else "uncertified (k_bar = inf); 200 used for illustration",
                     grid=grid_check(B, tau, k0))
        # Monte Carlo: start inside S, set-point beyond the wall (nominal pushes into it)
        rng = np.random.default_rng(3000); m = 1000
        x2 = rng.uniform(-1, 2, m)
        x1b = D.X_BAR - tau*x2 - x2*np.abs(x2)/(2*D.BRK) - D.h_of_level(B)
        x1 = x1b - rng.uniform(0.05, 5.0, m)
        rT = rng.uniform(D.X_BAR - 1, D.X_BAR + 5, m)
        for scen, wf in (("random", lambda i, g=np.random.default_rng(3001): g.uniform(-D.DELTA, D.DELTA, m)),
                         ("adversarial", lambda i: np.full(m, D.DELTA))):
            for name in ("nominal", "proposed", "cbf_robust"):
                md, vw, _ = simulate(name, B, tau, k0, x1.copy(), x2.copy(), lambda i: rT, wf)
                entry[f"{scen}_{name}"] = dict(violations=int(np.sum(md > 0)), worst=float(md.max()),
                                               wall_crossings=int(vw.sum()))
                if tau == TAU and scen == "random":
                    pd.DataFrame({"run": np.arange(m), "max_R_minus_B": md}).to_csv(f"di_mc_{name}.csv", index=False)
        if tau == TAU:   # deterministic run for figures
            for name in ("nominal", "proposed", "cbf_robust"):
                g = np.random.default_rng(2000)
                _, _, rows = simulate(name, B, tau, k0, np.array([0.0]), np.array([0.0]),
                                      lambda i: np.array([D.X_BAR + 3.0]), lambda i: g.uniform(-D.DELTA, D.DELTA, 1),
                                      T=20.0, record=True)
                pd.DataFrame(rows, columns=["time", "x1", "x2", "risk", "u"]).to_csv(f"di_traj_{name}.csv", index=False)
        res[tag] = entry
    json.dump(res, open("di_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()

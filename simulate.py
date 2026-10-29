"""Step 3 -- closed-loop validation on the SAME plant as calibrate.py / certify.py.

Controllers: nominal ACC only; proposed supervisor (deployed level B_hat_plus, k0 from
certificate.json); exact closed-form 1-D CBF-QP (gamma = GAMMA) on the same plant, in a standard
(disturbance-free) and a robust (worst-case disturbance term added) version.

Scenarios
  det : cruise 20 m/s at nominal headway; lead brakes at A_LEAD to standstill at t=10 s,
        re-accelerates at 1.5 m/s^2 to 20 m/s at t=25 s; random disturbance (seed 2000).
  mc  : 1000 runs, random start inside S (between the boundary and nominal headway),
        one random hard-braking event (decel U(2, A_LEAD), to U(0, vl)) at U(5, 30) s,
        random disturbance per step (seeds 3000 + i).
  adv : same 1000 initial conditions with ADVERSARIAL disturbance: d_v = +DELTA_V always,
        lead braking at the full A_LEAD to standstill.
Metrics: violations (max_t R - B > 0), worst max_t(R - B), fraction of runs entering the
boundary layer (max_t R >= B - 0.01), collisions (s <= 0), projection-active fraction,
controller runtime per evaluation.
Run: python simulate.py            (k0 = k_bar from certificate.json, theorem-consistent)
     python simulate.py --k0 50    (smaller gain, certified only by the direct closed-loop
                                    boundary check "smallest_ladder_gain_passing")
Outputs are overwritten; simulation_summary.json records the k0 used.
"""
import json, sys, time, numpy as np, pandas as pd, model as M

T_END, GAMMA, K1, MU = 60.0, 5.0, 10.0, 0.05
STEPS = int(T_END/M.DT)

def controller(name, s, v, vl, B, k0):
    if name == "nominal":
        return M.u_nominal(s, v, vl), np.zeros_like(s, bool)
    if name == "proposed":
        u, R, a, pa = M.u_supervisor(s, v, vl, B, k0, K1, MU); return u, pa
    if name == "cbf":
        u, R = M.u_cbf(s, v, vl, B, GAMMA); return u, np.zeros_like(s, bool)
    if name == "cbf_robust":
        u, R = M.u_cbf(s, v, vl, B, GAMMA, robust=True); return u, np.zeros_like(s, bool)

def run_batch(name, B, k0, s, v, vl, lead_profile, dist, record=False):
    """Vectorized over runs. lead_profile(i, vl) -> w_l ; dist(i) -> d_v."""
    s, v, vl = s.copy(), v.copy(), vl.copy()
    maxdev = np.full(len(s), -np.inf); coll = np.zeros(len(s), bool)
    proj = 0; rows = []; tsum = 0.0
    for i in range(STEPS):
        R = M.risk(s, v, vl)
        maxdev = np.maximum(maxdev, R - B)
        t0 = time.perf_counter()
        u, pa = controller(name, s, v, vl, B, k0)
        tsum += time.perf_counter() - t0
        proj += int(pa.sum())
        if record:
            rows.append([i*M.DT, s[0], v[0], vl[0], R[0], u[0]])
        s, v, vl = M.step(s, v, vl, u, dist(i), lead_profile(i, vl))
        coll |= (s <= 0)
    return dict(maxdev=maxdev, coll=coll, proj_frac=proj/(STEPS*len(s)),
                us_per_eval_batch=1e6*tsum/STEPS, rows=rows)

def det_lead(i, vl):
    t = i*M.DT
    if 10.0 <= t < 25.0: return np.where(vl > 0, -M.A_LEAD, 0.0)
    if t >= 25.0:        return np.where(vl < 20.0, 1.5, 0.0)
    return np.zeros_like(vl)

def main():
    cal = json.load(open("calibration.json")); cert = json.load(open("certificate.json"))
    B = cal["B_hat_plus"]; k0 = cert["B_hat_plus"]["k0_used"]
    if "--k0" in sys.argv:
        k0 = float(sys.argv[sys.argv.index("--k0") + 1])
    summary = dict(level_used="B_hat_plus", B=B, k0=k0, k1=K1, mu=MU, gamma_cbf=GAMMA,
                   dt=M.DT, T=T_END, U=[M.U_MIN, M.U_MAX], delta_v=M.DELTA_V, a_lead=M.A_LEAD)
    # deterministic scenario
    s0, v0 = np.array([5 + 1.5*20.0]), np.array([20.0])
    for name in ("nominal", "proposed", "cbf", "cbf_robust"):
        rng = np.random.default_rng(2000)
        r = run_batch(name, B, k0, s0, v0, v0.copy(), det_lead,
                      lambda i: rng.uniform(-M.DELTA_V, M.DELTA_V, 1), record=True)
        pd.DataFrame(r["rows"], columns=["time", "spacing", "velocity", "lead_velocity",
                                         "risk", "control"]).to_csv(f"traj_{name}.csv", index=False)
        summary[f"det_{name}"] = dict(max_R_minus_B=float(r["maxdev"][0]), collision=bool(r["coll"][0]))
    # Monte Carlo initial conditions (inside S)
    n = 1000; rng = np.random.default_rng(3000)
    v = rng.uniform(5, 30, n); vl = np.clip(v + rng.uniform(-5, 5, n), 0, 30)
    s_bd = M.margin_of_level(B) + M.S0 + M.H*v + (v*v - vl*vl)/(2*M.BD)
    s_nom = np.maximum(5 + 1.5*v, s_bd + 0.5)
    s = s_bd + rng.uniform(0.1, 1.0, n)*(s_nom - s_bd)
    assert np.all(M.risk(s, v, vl) < B)
    tb = rng.uniform(5, 30, n); dec = rng.uniform(2, M.A_LEAD, n); vfin = rng.uniform(0, 1, n)*vl
    def mc_lead(i, vlc):
        t = i*M.DT
        return np.where((t >= tb) & (vlc > vfin), -dec, 0.0)
    def adv_lead(i, vlc):
        t = i*M.DT
        return np.where((t >= tb) & (vlc > 0), -M.A_LEAD, 0.0)
    for scen, lead, distf in (
            ("mc",  mc_lead,  lambda i, r=np.random.default_rng(3001): r.uniform(-M.DELTA_V, M.DELTA_V, n)),
            ("adv", adv_lead, lambda i: np.full(n, M.DELTA_V))):
        for name in ("nominal", "proposed", "cbf", "cbf_robust"):
            r = run_batch(name, B, k0, s, v, vl, lead, distf)
            summary[f"{scen}_{name}"] = dict(
                runs=n, violations=int(np.sum(r["maxdev"] > 0)),
                worst_max_R_minus_B=float(r["maxdev"].max()),
                reached_boundary_layer=float(np.mean(r["maxdev"] >= -0.01)),
                collisions=int(r["coll"].sum()), projection_active_fraction=r["proj_frac"])
            if scen == "mc":
                pd.DataFrame({"run": np.arange(n), "max_R_minus_B": r["maxdev"]}).to_csv(
                    f"mc_{name}.csv", index=False)
            print(scen, name, summary[f"{scen}_{name}"])
    # runtime: single-state evaluations
    xs = (np.array([20.0]), np.array([15.0]), np.array([10.0]))
    for name in ("proposed", "cbf", "cbf_robust"):
        t0 = time.perf_counter()
        for _ in range(20000): controller(name, *xs, B, k0)
        summary[f"runtime_us_{name}"] = 1e6*(time.perf_counter() - t0)/20000
    json.dump(summary, open("simulation_summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in summary if k.startswith("det") or k.startswith("runtime")}, indent=1))

if __name__ == "__main__":
    main()

"""Closed-loop validation on the plant of calibrate.py / certify.py.

Controllers: nominal ACC only; proposed supervisor (deployed level B_hat_plus, k0 from
certificate.json); exact closed-form 1-D CBF-QP (gamma = GAMMA) on the same plant, in a standard
(disturbance-free) and a robust (worst-case disturbance term added) version.

Scenarios
  det : cruise 20 m/s at nominal headway; lead brakes at A_LEAD to standstill at t=10 s,
        re-accelerates at 1.5 m/s^2 to 20 m/s at t=25 s; random disturbance (seed 2000).
  mc  : 1000 runs, random start inside S (between the boundary and nominal headway),
        one random hard-braking event (decel U(2, A_LEAD), to U(0, vl)) at U(5, 30) s,
        random disturbance per step; initial states and braking events seed 3000,
        disturbances seed 3001, re-seeded for each controller so all controllers see
        the same disturbance realization.
  adv : same 1000 initial conditions with ADVERSARIAL disturbance: d_v = +DELTA_V always,
        lead braking at the full A_LEAD to standstill.
Metrics: violations (max_t R - B > 0), worst max_t(R - B), fraction of runs entering the
boundary layer (max_t R >= B - 0.01), collisions (s <= 0), projection-active fraction,
controller runtime per evaluation.
Run: python simulate.py              (k0 = ceil(k_bar) from certificate.json, Theorem 1 case (ii))
     python simulate.py --k0 50      (smaller gain: passes the direct grid check only, not Theorem 1)
     python simulate.py --dt 0.001   (time-step refinement: writes simulation_summary_dt0.001.json only)
simulation_summary.json records the k0 and dt used.
"""
import json, sys, time, numpy as np, pandas as pd, model as M

T_END, GAMMA, K1, MU = 60.0, 5.0, 10.0, 0.05
DT = float(sys.argv[sys.argv.index("--dt") + 1]) if "--dt" in sys.argv else M.DT
STEPS = int(round(T_END/DT))

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
    proj = 0; rows = []; vmax = float(v.max())
    for i in range(STEPS):
        R = M.risk(s, v, vl)
        maxdev = np.maximum(maxdev, R - B)
        u, pa = controller(name, s, v, vl, B, k0)
        proj += int(pa.sum())
        if record:
            rows.append([i*DT, s[0], v[0], vl[0], R[0], u[0]])
        s, v, vl = M.step(s, v, vl, u, dist(i), lead_profile(i, vl), dt=DT)
        coll |= (s <= 0); vmax = max(vmax, float(v.max()))
    return dict(maxdev=maxdev, coll=coll, proj_frac=proj/(STEPS*len(s)), vmax=vmax, rows=rows)

def det_lead(i, vl):
    t = i*DT
    if 10.0 <= t < 25.0: return np.where(vl > 0, -M.A_LEAD, 0.0)
    if t >= 25.0:        return np.where(vl < 20.0, 1.5, 0.0)
    return np.zeros_like(vl)

def main():
    cal = json.load(open("calibration.json")); cert = json.load(open("certificate.json"))
    B = cal["B_hat_plus"]; k0 = cert["B_hat_plus"]["k0_used"]
    if "--k0" in sys.argv:
        k0 = float(sys.argv[sys.argv.index("--k0") + 1])
    refine = "--dt" in sys.argv          # refinement run: summary only, no CSVs / runtimes
    summary = dict(level_used="B_hat_plus", B=B, k0=k0, k1=K1, mu=MU, gamma_cbf=GAMMA,
                   dt=DT, T=T_END, U=[M.U_MIN, M.U_MAX], delta_v=M.DELTA_V, a_lead=M.A_LEAD)
    # deterministic scenario
    s0, v0 = np.array([5 + 1.5*20.0]), np.array([20.0])
    for name in ("nominal", "proposed", "cbf", "cbf_robust"):
        rng = np.random.default_rng(2000)
        r = run_batch(name, B, k0, s0, v0, v0.copy(), det_lead,
                      lambda i: rng.uniform(-M.DELTA_V, M.DELTA_V, 1), record=not refine)
        if not refine:
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
        t = i*DT
        return np.where((t >= tb) & (vlc > vfin), -dec, 0.0)
    def adv_lead(i, vlc):
        t = i*DT
        return np.where((t >= tb) & (vlc > 0), -M.A_LEAD, 0.0)
    def mc_dist():                       # fresh generator per controller: common random numbers
        g = np.random.default_rng(3001)
        return lambda i: g.uniform(-M.DELTA_V, M.DELTA_V, n)
    def adv_dist():
        return lambda i: np.full(n, M.DELTA_V)
    for scen, lead, distmaker in (("mc", mc_lead, mc_dist), ("adv", adv_lead, adv_dist)):
        for name in ("nominal", "proposed", "cbf", "cbf_robust"):
            r = run_batch(name, B, k0, s, v, vl, lead, distmaker())
            summary[f"{scen}_{name}"] = dict(
                runs=n, violations=int(np.sum(r["maxdev"] > 0)),
                worst_max_R_minus_B=float(r["maxdev"].max()),
                reached_boundary_layer=float(np.mean(r["maxdev"] >= -0.01)),
                collisions=int(r["coll"].sum()), projection_active_fraction=r["proj_frac"],
                max_ego_speed=r["vmax"])
            if scen == "mc" and not refine:
                pd.DataFrame({"run": np.arange(n), "max_R_minus_B": r["maxdev"]}).to_csv(
                    f"mc_{name}.csv", index=False)
            print(scen, name, summary[f"{scen}_{name}"])
    if refine:
        out = f"simulation_summary_dt{DT:g}.json"
        json.dump(summary, open(out, "w"), indent=1); print("written", out); return
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

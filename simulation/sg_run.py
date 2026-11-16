"""Calibrate, certify and simulate the vanishing-authority example of sg_model.py
(Supporting Information S-II).

Sections of the output:
  1. calibration          3000 independent baseline episodes -> B, eps_N, B_hat_plus
  2. singular geometry    a_min on the boundary, the singular point, the near-singular band
  3. conditions at a = 0  slack of condition (15) and drift-only slack, compared pointwise
  4. certificate          case (i) and case (ii) gains, with and without the band
  5. Monte Carlo          nominal / supervisor / CBF-QP / robust CBF-QP
  6. band exercise        how often closed-loop trajectories enter the near-singular band
  7. conditioning         implied gain of the CBF-QP vs the fixed supervisor gain

Output: sg_results.json      Run: python sg_run.py [--dt 0.001]
"""
import argparse, json, math
import numpy as np
import sg_model as S

SEED_CAL, N_CAL, T_EP = 4000, 3000, 30.0
ALPHA, BETA_DKW = 0.05, 0.05
SEED_MC_IC, SEED_MC_W = 4100, 4101
N_MC, T_MC = 1000, 40.0
K1, GAMMA = 10.0, 5.0
# Boundary starts begin at R = B, so R - B is 0 +/- 1e-16 in floating point. Violations are
# counted with this tolerance; the raw maxima are reported as well.
VIOL_TOL = 1e-12
# Gain-scheduling width as a fraction of the level: with MU > B the scheduled gain
# k0 clip(1 - M/MU, 0, 1) would not reach zero inside S (M = B - R <= B).
MU_FRAC = 0.25
GRIDN = 400001           # x2 grid on [-X2MAX, X2MAX]; V_W is inserted exactly (see boundary_grid)


def boundary_grid():
    """Dense x2 grid that includes the singular speed V_W."""
    x2 = np.linspace(-S.X2MAX, S.X2MAX, GRIDN)
    x2 = np.unique(np.concatenate([x2, [S.V_W]]))
    return x2


# --------------------------------------------------------------------- 1. calibration
def calibrate(dt):
    """One state per independent episode at an independently drawn time (Assumption 2)."""
    rng = np.random.default_rng(SEED_CAL)
    n = int(T_EP/dt)
    x1 = rng.uniform(0.0, 6.0, N_CAL)
    x2 = rng.uniform(-1.0, 1.0, N_CAL)
    r = rng.uniform(0.0, 8.0, N_CAL)
    t_sample = rng.integers(0, n, N_CAL)
    samples = np.full(N_CAL, np.nan)
    for i in range(n):
        if i % int(5.0/dt) == 0:
            r = rng.uniform(0.0, 8.0, N_CAL)
        hit = (t_sample == i)
        samples[hit] = S.risk(x1[hit], x2[hit])
        u = S.u_nominal(x1, x2, r)
        w = rng.uniform(-S.DELTA, S.DELTA, N_CAL)
        x1, x2 = S.step(x1, x2, u, w, dt)
    eps = float(math.sqrt(math.log(2/BETA_DKW)/(2*N_CAL)))
    return dict(seed=SEED_CAL, N=N_CAL, episode_length_s=T_EP, alpha=ALPHA, beta=BETA_DKW,
                B=float(np.quantile(samples, 1-ALPHA, method="inverted_cdf")),
                eps_N=eps,
                B_hat_plus=float(np.quantile(samples, 1-ALPHA+eps, method="inverted_cdf")),
                sample_quantiles={str(q): float(np.quantile(samples, q, method="inverted_cdf"))
                                  for q in (0.5, 0.9, 0.95, 0.99)},
                max_abs_x2_seen=float(np.nanmax(np.abs(x2))))


# --------------------------------------------------------------------- 2-4. certificate
def certify(b, dt):
    """Hypotheses of Theorem 1 on {R = b} over the operating domain |x2| <= X2MAX."""
    x2 = boundary_grid()
    ngrid = len(x2)
    x1 = S.x1_of_boundary(b, x2)
    R, LfR, a, D, G = S.drift_terms(x1, x2)
    sb_over_ell = b*(1 - b)/S.ELL

    # the nominal law on this boundary, with the set-point beyond the wall (worst case)
    un = S.u_nominal(x1, x2, np.full_like(x2, 50.0))
    nom = LfR + a*un

    # pointwise slacks of the two conditions, in physical units
    slack16 = -(LfR + np.minimum(a*(-S.U_BAR), a*S.U_BAR)) - D
    slackA5 = -(nom) - D

    i_sing = int(np.argmin(np.abs(a)))
    a_min = float(np.abs(a).min())
    # exact singular point: x2 = V_W
    j = int(np.argmin(np.abs(x2 - S.V_W)))

    # near-singular band: the contiguous run around V_W on which slackA5 >= 0
    ok = slackA5 >= 0
    lo = hi = j
    if ok[j]:
        while lo > 0 and ok[lo-1]:
            lo -= 1
        while hi < ngrid-1 and ok[hi+1]:
            hi += 1
    a_edges = (float(np.abs(a[lo])), float(np.abs(a[hi])))
    a_thr_max = min(a_edges) if ok[j] else 0.0
    a_thr = 0.9*a_thr_max                       # any threshold below the edge works

    off = np.abs(a) > a_thr                     # boundary minus the band
    eta_global = float(slack16.min())
    eta_off = float(slack16[off].min())
    du2 = (2*S.U_BAR)**2
    c_nom, D_max = float(nom.max()), float(D.max())
    k0_i_whole = None if a_min <= 1e-15 else float((c_nom + D_max)/a_min**2)
    k0_i_off = float((c_nom + D_max)/a_thr**2) if a_thr > 0 else None
    out = dict(
        level=float(b), grid_points=ngrid, domain_x2max=S.X2MAX,
        max_abs_R_minus_b=float(np.max(np.abs(R - b))),
        a_min=a_min, a_min_at_x2=float(x2[i_sing]),
        singular_point=dict(x2=float(x2[j]), x1=float(x1[j]), a=float(a[j]),
                            LfR=float(LfR[j]), D=float(D[j]), grad_norm=float(G[j]),
                            slack_cond16=float(slack16[j]), slack_assumption5=float(slackA5[j]),
                            conditions_identical=bool(abs(slack16[j]-slackA5[j]) < 1e-14)),
        G_bd=float(G.max()), D_max=D_max, c_nom=c_nom,
        nominal_unsafe_fraction=float(np.mean(nom + D > 0)),
        assumption5_band=dict(x2_lo=float(x2[lo]), x2_hi=float(x2[hi]),
                              width=float(x2[hi]-x2[lo]),
                              a_at_edges=a_edges, a_thr_max=float(a_thr_max),
                              a_thr_used=float(a_thr),
                              holds_at_singular_point=bool(ok[j]),
                              fraction_of_boundary=float(ok.mean())),
        # ---- case (ii): Lemma 1. Its bound has no ||a|| in it.
        case_ii=dict(eta_whole_boundary=eta_global,
                     eta_outside_band=eta_off,
                     k_bar_whole_boundary=float(du2/eta_global) if eta_global > 0 else None,
                     k_bar_outside_band=float(du2/eta_off) if eta_off > 0 else None,
                     cond16_holds_on_whole_boundary=bool(eta_global > 0),
                     band_needed=bool(eta_global <= 0)),
        # ---- case (i): gain condition, which does contain ||a||^2
        case_i=dict(k0_whole_boundary=k0_i_whole, k0_outside_band=k0_i_off,
                    infeasible_without_band=bool(a_min <= 1e-15)),
        sb_over_ell=float(sb_over_ell),
    )
    k0 = math.ceil(out["case_ii"]["k_bar_whole_boundary"])
    u = np.clip(un - k0*a, -S.U_BAR, S.U_BAR)
    out["k0_used"] = k0
    out["closed_loop_worst"] = float((LfR + a*u + D).max())
    out["closed_loop_ok"] = bool(out["closed_loop_worst"] <= 0)
    return out


# --------------------------------------------------------------------- 5-7. Monte Carlo
def montecarlo(b, k0, dt, adversarial, a_thr, mu):
    rng_ic = np.random.default_rng(SEED_MC_IC)
    x2_0 = rng_ic.uniform(-1.0, 2.0, N_MC)
    depth = rng_ic.uniform(0.05, 5.0, N_MC)
    x1_0 = S.x1_of_boundary(b, x2_0) - depth
    r = rng_ic.uniform(9.0, 15.0, N_MC)
    n = int(T_MC/dt)

    res = {}
    for name in ("nominal", "proposed", "cbf", "cbf_robust"):
        rng_w = np.random.default_rng(SEED_MC_W)      # same realization for every arm
        x1, x2 = x1_0.copy(), x2_0.copy()
        worst = np.full(N_MC, -np.inf)
        # Band entries are counted only in the boundary layer R >= B - MU (deep inside S,
        # a(x) is small only because sigma' is small).
        in_band = np.zeros(N_MC, bool)
        band_ticks = layer_ticks = 0
        near_singular_dist = np.full(N_MC, np.inf)
        crossed = np.zeros(N_MC, bool)
        max_implied = 0.0
        max_abs_x2 = 0.0
        sat_ticks = 0
        for i in range(n):
            if name == "nominal":
                u = S.u_nominal(x1, x2, r); R = S.risk(x1, x2); a = S.risk_grad(x1, x2)[3]
                sat = np.zeros(N_MC, bool)
            elif name == "proposed":
                u, R, a, sat = S.u_supervisor(x1, x2, r, b, k0, K1, mu)
            else:
                u, R, a = S.u_cbf(x1, x2, r, b, GAMMA, robust=(name == "cbf_robust"))
                sat = np.zeros(N_MC, bool)
                with np.errstate(divide="ignore", invalid="ignore"):
                    imp = np.where(a > 1e-12, np.abs(u - S.u_nominal(x1, x2, r))/(a*a), 0.0)
                max_implied = max(max_implied, float(np.nanmax(imp)))
            worst = np.maximum(worst, R - b)
            layer = R >= b - mu                       # the boundary layer
            band = layer & (np.abs(a) <= a_thr)       # near-singular AND near the boundary
            in_band |= band
            band_ticks += int(band.sum())
            layer_ticks += int(layer.sum())
            near_singular_dist = np.where(layer,
                                          np.minimum(near_singular_dist,
                                                     np.abs(x2 - S.V_W)),
                                          near_singular_dist)
            sat_ticks += int(sat.sum())
            crossed |= (x1 >= S.X_BAR)
            max_abs_x2 = max(max_abs_x2, float(np.abs(x2).max()))
            w = (np.full(N_MC, S.DELTA) if adversarial
                 else rng_w.uniform(-S.DELTA, S.DELTA, N_MC))
            x1, x2 = S.step(x1, x2, u, w, dt)
        reached = np.isfinite(near_singular_dist)
        res[name] = dict(runs=N_MC, violations=int((worst > VIOL_TOL).sum()),
                         viol_tol=VIOL_TOL,
                         worst_max_R_minus_B=float(worst.max()),
                         wall_crossings=int(crossed.sum()),
                         runs_reaching_boundary_layer=int(reached.sum()),
                         runs_entering_band_in_layer=int(in_band.sum()),
                         band_ticks=band_ticks, layer_ticks=layer_ticks,
                         band_fraction_of_layer=float(band_ticks/layer_ticks)
                         if layer_ticks else 0.0,
                         median_closest_approach_to_V_W=float(
                             np.median(near_singular_dist[reached])) if reached.any() else None,
                         min_closest_approach_to_V_W=float(
                             near_singular_dist[reached].min()) if reached.any() else None,
                         projection_active_fraction=float(sat_ticks/(N_MC*n)),
                         max_abs_x2=max_abs_x2,
                         max_implied_gain=(max_implied if name.startswith("cbf") else float(k0)))
    return res


def montecarlo_boundary(b, k0, dt, adversarial, a_thr, mu, n_x2=2001):
    """Start on the boundary {R = b}, sweeping the initial speed through the singular
    speed V_W (x2(0) = V_W included)."""
    x2_0 = np.unique(np.concatenate([np.linspace(-2.0, 2.0, n_x2), [S.V_W]]))
    N = len(x2_0)
    per_start = {"x2_0": x2_0}
    x1_0 = S.x1_of_boundary(b, x2_0)
    r = np.full(N, 50.0)                 # set-point far beyond the wall: worst-case nominal
    n = int(T_MC/dt)

    res = {}
    for name in ("nominal", "proposed", "cbf", "cbf_robust"):
        rng_w = np.random.default_rng(SEED_MC_W)
        x1, x2 = x1_0.copy(), x2_0.copy()
        worst = np.full(N, -np.inf)
        for i in range(n):
            if name == "nominal":
                u = S.u_nominal(x1, x2, r); R = S.risk(x1, x2)
            elif name == "proposed":
                u, R, a, sat = S.u_supervisor(x1, x2, r, b, k0, K1, mu)
            else:
                u, R, a = S.u_cbf(x1, x2, r, b, GAMMA, robust=(name == "cbf_robust"))
            worst = np.maximum(worst, R - b)
            w = (np.full(N, S.DELTA) if adversarial
                 else rng_w.uniform(-S.DELTA, S.DELTA, N))
            x1, x2 = S.step(x1, x2, u, w, dt)
        per_start[name] = worst.copy()
        j = int(np.argmin(np.abs(x2_0 - S.V_W)))        # the singular start
        band = np.abs(S.risk_grad(x1_0, x2_0)[3]) <= a_thr
        res[name] = dict(starts=N, violations=int((worst > VIOL_TOL).sum()),
                         viol_tol=VIOL_TOL,
                         worst_max_R_minus_B=float(worst.max()),
                         worst_at_x2_0=float(x2_0[int(np.argmax(worst))]),
                         at_singular_start=dict(x2_0=float(x2_0[j]),
                                                max_R_minus_B=float(worst[j]),
                                                violated=bool(worst[j] > VIOL_TOL)),
                         starts_in_band=int(band.sum()),
                         violations_in_band=int((worst[band] > VIOL_TOL).sum()),
                         worst_in_band=float(worst[band].max()) if band.any() else None)
    res["_per_start"] = per_start
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=S.DT)
    args = ap.parse_args()
    dt = args.dt
    suffix = "" if abs(dt - 0.01) < 1e-12 else "_dt%g" % dt

    print("=" * 78)
    print("SINGULAR-AUTHORITY EXAMPLE  (dt = %g s)" % dt)
    print("=" * 78)
    print("plant: x1' = x2,  x2' = beta(x2) u + w,  beta(x2) = (x2-V_W)^2/((x2-V_W)^2+c)")
    print("       V_W = %.2f m/s, c = %.3f, U_BAR = %.1f, DELTA = %.1f, |x2| <= %.1f"
          % (S.V_W, S.C_BETA, S.U_BAR, S.DELTA, S.X2MAX))

    print("\n1. CALIBRATION")
    cal = calibrate(dt)
    print("   N = %d independent episodes of %.0f s" % (cal["N"], cal["episode_length_s"]))
    print("   B        = %.6f   (barrier margin h = %+.3f m)"
          % (cal["B"], S.h_of_level(cal["B"])))
    print("   eps_N    = %.6f" % cal["eps_N"])
    print("   B_hat+   = %.6f   (barrier margin h = %+.3f m)"
          % (cal["B_hat_plus"], S.h_of_level(cal["B_hat_plus"])))
    b = cal["B_hat_plus"]

    print("\n2. SINGULAR GEOMETRY on the deployed boundary {R = B_hat+}")
    c = certify(b, dt)
    sp = c["singular_point"]
    print("   a_min on the boundary          = %.3e   at x2 = %+.4f m/s"
          % (c["a_min"], c["a_min_at_x2"]))
    print("   singular point x = (%.4f, %+.4f): a = %.3e  (Sigma ^ dS is NON-EMPTY)"
          % (sp["x1"], sp["x2"], sp["a"]))
    bd = c["assumption5_band"]
    print("   near-singular band around it   : x2 in [%+.4f, %+.4f], width %.4f m/s"
          % (bd["x2_lo"], bd["x2_hi"], bd["width"]))
    print("   |a| at the band edges          = %.5f, %.5f  =>  a_thr < %.5f"
          % (bd["a_at_edges"][0], bd["a_at_edges"][1], bd["a_thr_max"]))
    print("   drift-only condition holds on %.1f%% of the boundary (elsewhere the nominal"
          % (100*bd["fraction_of_boundary"]))
    print("   law accelerates toward the wall there)")

    print("\n3. CONDITIONS AT a = 0")
    print("   At the singular point:")
    print("     slack of condition (15) = %+.10f" % sp["slack_cond16"])
    print("     drift-only slack        = %+.10f" % sp["slack_assumption5"])
    print("   identical: %s" % sp["conditions_identical"])
    print("   at a = 0 both reduce to L_f R + D <= 0")

    print("\n4. CERTIFICATE")
    cii, ci = c["case_ii"], c["case_i"]
    print("   case (ii)  [Lemma 1; bound = sum (Delta u_i)^2 / k, no ||a|| in it]")
    print("     condition (15) on the whole boundary: eta = %+.6f  -> holds: %s"
          % (cii["eta_whole_boundary"], cii["cond16_holds_on_whole_boundary"]))
    print("     k_bar whole boundary               = %.1f" % cii["k_bar_whole_boundary"])
    print("     k_bar outside the band             = %.1f  (eta = %+.6f)"
          % (cii["k_bar_outside_band"], cii["eta_outside_band"]))
    print("     band needed for case (ii)?         %s" % cii["band_needed"])
    print("   case (i)   [gain condition k0 ||a||^2 >= c_nom + D]")
    print("     c_nom = %.6f   D_max = %.6f   G_bd = %.6f"
          % (c["c_nom"], c["D_max"], c["G_bd"]))
    print("     k0 on the whole boundary           = %s"
          % ("infinite (a_min = 0)" if ci["infeasible_without_band"] else "%.1f" % ci["k0_whole_boundary"]))
    print("     k0 outside the band (a_thr = %.5f) = %.1f"
          % (bd["a_thr_used"], ci["k0_outside_band"]))
    print("   deployed gain k0 = ceil(k_bar) = %d ; boundary check D^+R <= 0: worst %+.3e, ok %s"
          % (c["k0_used"], c["closed_loop_worst"], c["closed_loop_ok"]))

    k0 = c["k0_used"]
    a_thr = bd["a_thr_used"]
    mu = MU_FRAC*b
    print("\n   gain schedule: k1 = %.1f, mu = %.5f = %.2f x B_hat+"
          % (K1, mu, MU_FRAC))
    print("\n5. MONTE CARLO  (%d runs, %.0f s, set-point beyond the wall)" % (N_MC, T_MC))
    mc = {}
    for scen, adv in (("random", False), ("adversarial", True)):
        mc[scen] = montecarlo(b, k0, dt, adv, a_thr, mu)
        print("   %s disturbance" % scen)
        print("     %-12s %-11s %-13s %-7s %-9s %s"
              % ("controller", "violations", "worst R-B", "wall", "in layer", "max |x2|"))
        for name in ("nominal", "proposed", "cbf", "cbf_robust"):
            r = mc[scen][name]
            print("     %-12s %-11d %+13.3e %-7d %-9d %.3f"
                  % (name, r["violations"], r["worst_max_R_minus_B"], r["wall_crossings"],
                     r["runs_reaching_boundary_layer"], r["max_abs_x2"]))

    print("\n6. BAND EXERCISE  (a_thr = %.5f, boundary layer R >= B - %.4f)" % (a_thr, mu))
    print("   counted only in the boundary layer")
    print("   %-12s %-11s %-13s %-14s %s"
          % ("controller", "runs->band", "band/layer", "median |x2-V_W|", "min |x2-V_W|"))
    for name in ("proposed", "cbf_robust"):
        r = mc["random"][name]
        print("   %-12s %-11d %-13.4f %-14s %s"
              % (name, r["runs_entering_band_in_layer"], r["band_fraction_of_layer"],
                 ("%.5f" % r["median_closest_approach_to_V_W"])
                 if r["median_closest_approach_to_V_W"] is not None else "--",
                 ("%.2e" % r["min_closest_approach_to_V_W"])
                 if r["min_closest_approach_to_V_W"] is not None else "--"))
    pr = mc["random"]["proposed"]
    frac = pr["runs_entering_band_in_layer"]/N_MC
    if frac >= 0.05:
        print("   near-singular band entered in %.1f%% of the"
              % (100*frac))
        print("      interior-start runs.")
    else:
        print("   band entered by %d of %d interior-start runs (median closest"
              % (pr["runs_entering_band_in_layer"], N_MC))
        print("      approach to V_W on the boundary %.3f m/s)"
              % pr["median_closest_approach_to_V_W"])

    print("\n6b. BOUNDARY-START SWEEP  (x2(0) swept through V_W, x1(0) on {R = B_hat+})")
    mcb = {}
    for scen, adv in (("random", False), ("adversarial", True)):
        mcb[scen] = montecarlo_boundary(b, k0, dt, adv, a_thr, mu)
        ps_ = mcb[scen].pop("_per_start")
        import csv as _csv
        fn_ = "sg_boundary_%s%s.csv" % (scen, suffix)
        with open(fn_, "w", newline="") as fh:
            w_ = _csv.writer(fh)
            cols = ["nominal", "proposed", "cbf", "cbf_robust"]
            w_.writerow(["x2_0"] + ["max_R_minus_B_" + c for c in cols])
            for i_ in range(len(ps_["x2_0"])):
                w_.writerow(["%.10g" % ps_["x2_0"][i_]]
                            + ["%.10e" % ps_[c][i_] for c in cols])
        print("    %s disturbance  (%d boundary starts, %d of them inside Sigma_a_thr)"
              % (scen, mcb[scen]["proposed"]["starts"],
                 mcb[scen]["proposed"]["starts_in_band"]))
        print("      per-start maxima written to %s" % fn_)
        print("      %-12s %-11s %-13s %-13s %s"
              % ("controller", "violations", "worst R-B", "at x2(0)=V_W", "worst in band"))
        for name in ("nominal", "proposed", "cbf", "cbf_robust"):
            r = mcb[scen][name]
            print("      %-12s %-11d %+13.3e %+13.3e %s"
                  % (name, r["violations"], r["worst_max_R_minus_B"],
                     r["at_singular_start"]["max_R_minus_B"],
                     ("%+.3e" % r["worst_in_band"]) if r["worst_in_band"] is not None else "--"))
    ps = mcb["adversarial"]["proposed"]
    print("    start at the singular point, worst-case disturbance: supervisor %s the envelope (max R-B = %+.3e)."
          % ("leaves" if ps["at_singular_start"]["violated"] else "does not leave",
             ps["at_singular_start"]["max_R_minus_B"]))

    print("\n7. CONDITIONING")
    print("   supervisor implied gain           = %d (constant by construction)" % k0)
    for scen in ("random", "adversarial"):
        print("   CBF-QP max implied gain |u-u_nom|/a^2, %-11s = %.3e"
              % (scen, mc[scen]["cbf_robust"]["max_implied_gain"]))

    dom_sup = max(mc[s][n]["max_abs_x2"] for s in mc
                  for n in ("proposed", "cbf", "cbf_robust"))
    dom_all = max(mc[s][n]["max_abs_x2"] for s in mc for n in mc[s])
    print("\n   DOMAIN CHECK (certificate is checked on |x2| <= %.1f m/s)" % S.X2MAX)
    print("     max |x2| over the supervised and filtered runs = %.4f m/s: %s"
          % (dom_sup, "inside" if dom_sup <= S.X2MAX else "outside the checked domain"))
    print("     max |x2| including the unsupervised nominal runs = %.4f m/s %s"
          % (dom_all, "(those runs cross the wall, so they leave the region of interest)"
             if dom_all > dom_sup else ""))

    out = dict(dt=dt, mu=mu, mu_frac=MU_FRAC, k1=K1, gamma_cbf=GAMMA,
           calibration=cal, certificate=c, montecarlo=mc, montecarlo_boundary=mcb,
               max_abs_x2_supervised=dom_sup, max_abs_x2_all_runs=dom_all,
               domain_ok=bool(dom_sup <= S.X2MAX))
    fn = "sg_results%s.json" % suffix
    json.dump(out, open(fn, "w"), indent=1)
    print("\nwrote %s" % fn)


if __name__ == "__main__":
    main()

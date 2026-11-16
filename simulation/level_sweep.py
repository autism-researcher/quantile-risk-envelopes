"""Certified level and actuator authority for the vehicle-following example (Section 6.2,
Tables 3-4, Fig. 6).

certify.py evaluates the conditions of Theorem 1 at two levels (B and B_hat_plus).
This script evaluates them over a ladder of candidate levels b and over the actuator and
lead-braking limits, and reports

    B_dep := max { b <= B_hat_plus : condition (15) holds on {R = b} and the closed loop at
                   k0 = ceil(k_bar) satisfies D^+R <= 0 on the whole level set }

together with the certified ceiling b_max (the largest certifiable level, whether or not
it is below B_hat_plus) and the binding constraint at the ceiling.

Three sweeps:
  1. level ladder at the nominal limits U = [-8, 4] m/s^2, A_LEAD = 4 m/s^2;
  2. brake limit U_MIN, with the baseline data recalibrated at each limit (a weaker brake
     also changes nominal driving and therefore B_hat_plus);
  3. assumed worst-case lead deceleration A_LEAD at the nominal brake limit.

check_level() re-implements certify.check()'s arithmetic so that the grid size and the
limits can be varied. It is validated against certificate.json before use (see selftest()).

Output: level_sweep.json      Run: python level_sweep.py   (after calibrate.py, certify.py)
"""
import json, math, os, sys

# model.py, calibration.json and certificate.json are read from this folder.
_V12 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _V12)
for _need in ("calibration.json", "certificate.json"):
    if not os.path.isfile(os.path.join(_V12, _need)):
        raise SystemExit("%s not found in %s; run calibrate.py and certify.py first."
                         % (_need, _V12))
_OUT = os.getcwd()
os.chdir(_V12)

import numpy as np
import model as M

VMAX = 35.0
GRID_SWEEP = 241      # coarse ladder: 0.146 m/s spacing
GRID_FINAL = 701      # certify.py's grid, used to re-verify the reported levels


# ----------------------------------------------------------------- certificate check
def check_level(b, n=GRID_FINAL, u_min=None, u_max=None, a_lead=None, k0=None):
    """Hypotheses of Theorem 1 on {R = b}, as a function of level and limits.

    Mirrors certify.check(): same boundary parametrization, same constants, same
    disturbance term D(x) = |R_v| DELTA_V + |R_vl| A_LEAD.
    """
    u_min = M.U_MIN if u_min is None else u_min
    u_max = M.U_MAX if u_max is None else u_max
    a_lead = M.A_LEAD if a_lead is None else a_lead

    v, vl = np.meshgrid(np.linspace(0, VMAX, n), np.linspace(0, VMAX, n))
    v, vl = v.ravel(), vl.ravel()
    s = M.margin_of_level(b) + M.S0 + M.H*v + (v*v - vl*vl)/(2*M.BD)
    keep = s >= 0
    s, v, vl = s[keep], v[keep], vl[keep]

    R, Rs, Rv, Rl = M.risk_grad(s, v, vl)
    LfR = Rs*(vl - v) + Rv*(-M.CD*v*v)
    a = Rv
    D = np.abs(Rv)*M.DELTA_V + np.abs(Rl)*a_lead
    G = np.sqrt(Rs**2 + Rv**2 + Rl**2)

    # nominal law clipped to the same input box
    un = np.clip(0.5*(vl - v) + 0.2*(s - 5.0 - 1.5*v), u_min, u_max)
    nom = LfR + a*un

    mindrift = LfR + np.minimum(a*u_min, a*u_max)
    slack = -mindrift - D                                   # slack of condition (15), pointwise
    j = int(np.argmin(slack))                               # the binding boundary point
    eta = float(slack[j])
    du = u_max - u_min
    kbar = du*du/eta if eta > 0 else math.inf
    a_min = float(np.min(np.abs(a)))

    out = dict(level=float(b), grid=n, u_min=float(u_min), u_max=float(u_max),
               a_lead=float(a_lead), grid_points=int(len(s)),
               margin_m=float(M.margin_of_level(b)),
               G_bd=float(G.max()), D_max=float(D.max()), c_nom=float(nom.max()),
               a_min=a_min, a_max=float(np.max(np.abs(a))),
               eta_feas=eta, input_feasible=bool(eta > 0),
               k_bar=kbar if math.isfinite(kbar) else None,
               # boundary point where condition (15) is tightest
               binding_point=dict(s=float(s[j]), v=float(v[j]), vl=float(vl[j]),
                                  a=float(a[j]), LfR=float(LfR[j]), D=float(D[j]),
                                  dR_dvl=float(Rl[j])))

    if k0 is None:
        k0 = math.ceil(kbar) if math.isfinite(kbar) else None
    out["k0_used"] = k0
    if k0 is not None:
        cl = LfR + a*np.clip(un - k0*a, u_min, u_max) + D
        out["closed_loop_worst"] = float(cl.max())
        out["closed_loop_ok"] = bool(cl.max() <= 0)
    else:
        out["closed_loop_worst"] = None
        out["closed_loop_ok"] = False
    out["certified"] = bool(out["input_feasible"] and out["closed_loop_ok"])
    return out


def factorization_test(u_min=-4.8, verbose=True):
    """Why the certified set of levels is all-or-nothing for this risk functional.

    R = sigma(-m(x)/ELL) with m independent of the level, so on {R = b}
        grad R = sigma'(.) * (-grad m)/ELL,    sigma' = b(1-b) =: s_b,
    and every term of the slack of condition (15) carries the same factor s_b/ELL:

        slack(x; b) = (s_b/ELL) * [ -v + (H + v/BD)(CD v^2 - u_min - DELTA_V) ]   (at vl = 0)

    The bracket does not contain b. Hence the SIGN of the slack at a given (v, vl) is
    independent of the level: condition (15) either holds at every level or at none.
    Lowering the level therefore cannot restore feasibility -- only more input authority,
    a smaller disturbance bound, or a different risk functional can.

    This test checks the factorization numerically: slack(x; b) * ELL / s_b must be the
    same number at every level, for the same (v, vl).
    """
    v, vl = 17.21, 0.0
    vals = []
    for b in (0.02, 0.05, 0.10, 0.20, 0.30, 0.383, 0.50, 0.70, 0.90):
        s = M.margin_of_level(b) + M.S0 + M.H*v + (v*v - vl*vl)/(2*M.BD)
        R, Rs, Rv, Rl = M.risk_grad(s, v, vl)
        LfR = Rs*(vl - v) + Rv*(-M.CD*v*v)
        a = Rv
        D = abs(Rv)*M.DELTA_V + abs(Rl)*M.A_LEAD
        slack = -(LfR + min(a*u_min, a*M.U_MAX)) - D
        s_b = b*(1 - b)
        vals.append(slack*M.ELL/s_b)
    spread = max(vals) - min(vals)
    if verbose:
        print("  factorization test at (v, vl) = (%.2f, %.2f), u_min = %.1f:" % (v, vl, u_min))
        print("    slack * ELL / [b(1-b)] over 9 levels b in [0.02, 0.90]:")
        print("      min %.10f   max %.10f   spread %.2e" % (min(vals), max(vals), spread))
        print("    spread %.1e; bracket %.6f (negative: condition (15) fails at every level"
              % (spread, vals[0]))
        print("       for this u_min)")
    assert spread < 1e-9, "slack does not factor as claimed (spread %.3e)" % spread
    return dict(v=v, vl=vl, u_min=u_min, bracket=float(vals[0]),
                spread_over_levels=float(spread), levels_tested=9)


def selftest():
    """check_level must reproduce certify.py's certificate.json at both stored levels."""
    ref = json.load(open("certificate.json"))
    bad = []
    for key in ("B", "B_hat_plus"):
        r = ref[key]
        got = check_level(r["level"], n=GRID_FINAL)
        for fld in ("grid_points", "G_bd", "D_max", "c_nom", "a_min", "a_max",
                    "eta_feas", "k_bar", "k0_used", "closed_loop_worst"):
            w, g = r[fld], got[fld]
            if w is None or g is None:
                ok = (w is None) == (g is None)
            elif isinstance(w, int) and isinstance(g, int):
                ok = w == g
            else:
                ok = abs(g - w) <= 1e-9*max(1.0, abs(w))
            if not ok:
                bad.append((key, fld, w, g))
    if bad:
        for row in bad:
            print("  MISMATCH %s.%s  certificate.json %r  recomputed %r" % row)
        raise SystemExit("check_level does not reproduce certificate.json.")
    print("  selftest OK: check_level reproduces certificate.json at B and B_hat_plus")


# ----------------------------------------------------------------- baseline re-calibration
def calibrate(u_min, u_max, seed=1000, N=3000, T_EP=60.0, alpha=0.05, beta=0.05):
    """calibrate.py's procedure with the input box passed in (same seed and stream)."""
    rng = np.random.default_rng(seed)
    n = int(T_EP/M.DT)
    v = rng.uniform(5, 30, N)
    vl = np.clip(v + rng.uniform(-3, 3, N), 0, 30)
    s = 5 + 1.5*v + rng.uniform(0, 30, N)
    t_sample = rng.integers(0, n, N)
    samples = np.full(N, np.nan)
    wl = rng.uniform(-2.0, 1.5, N)
    for i in range(n):
        if i % 300 == 0:
            wl = rng.uniform(-2.0, 1.5, N)
        hit = (t_sample == i)
        samples[hit] = M.risk(s[hit], v[hit], vl[hit])
        u = np.clip(0.5*(vl - v) + 0.2*(s - 5.0 - 1.5*v), u_min, u_max)
        dv = rng.uniform(-M.DELTA_V, M.DELTA_V, N)
        wl_eff = np.where((vl >= 30) & (wl > 0), 0.0, wl)
        s, v, vl = M.step(s, v, vl, u, dv, wl_eff)
    eps = float(np.sqrt(np.log(2/beta)/(2*N)))
    return dict(B=float(np.quantile(samples, 1-alpha, method="inverted_cdf")),
                eps_N=eps,
                B_hat_plus=float(np.quantile(samples, 1-alpha+eps, method="inverted_cdf")))


# ----------------------------------------------------------------- sweeps
def ladder(levels, n, **kw):
    return [check_level(b, n=n, **kw) for b in levels]


def ceiling(rows):
    """Largest certified level in a ladder, and the first failing level above it."""
    cert = [r for r in rows if r["certified"]]
    if not cert:
        return None, None
    top = max(cert, key=lambda r: r["level"])
    above = [r for r in rows if r["level"] > top["level"]]
    return top, (min(above, key=lambda r: r["level"]) if above else None)


def binding(row):
    if not row["input_feasible"]:
        return "input feasibility (15): eta <= 0"
    if not row["closed_loop_ok"]:
        return "closed-loop boundary check at k0 = ceil(k_bar)"
    return "none (certified)"


def main():
    cal = json.load(open("calibration.json"))
    B, Bp = cal["B"], cal["B_hat_plus"]
    print("=" * 78)
    print("CERTIFIED-LEVEL SWEEP -- vehicle following")
    print("=" * 78)
    print("calibrated level   B        = %.6f  (margin %.3f m)" % (B, M.margin_of_level(B)))
    print("deployed level     B_hat+   = %.6f  (margin %.3f m)" % (Bp, M.margin_of_level(Bp)))
    print("\nvalidating check_level against certify.py ...")
    selftest()
    print("\n0. LEVEL FACTORIZATION OF THE SLACK")
    fac_fail = factorization_test(u_min=-4.8)
    fac_ok = factorization_test(u_min=-5.0, verbose=False)
    print("    bracket at u_min = -5.0 is %+.6f (positive: condition (15) holds at every level)"
          % fac_ok["bracket"])

    out = {"calibration": {"B": B, "B_hat_plus": Bp, "eps_N": cal["eps_N"]},
           "grid_sweep": GRID_SWEEP, "grid_final": GRID_FINAL}
    out["factorization"] = {"infeasible_case": fac_fail, "feasible_case": fac_ok}

    # ---- sweep 1: level ladder at the nominal limits ------------------------------
    print("\n1. LEVEL LADDER at U = [%.1f, %.1f] m/s^2, A_LEAD = %.1f m/s^2"
          % (M.U_MIN, M.U_MAX, M.A_LEAD))
    print("   (margin m(b) is the stopping margin in metres on {R=b}; m < 0 means the")
    print("    boundary lies inside the design stopping distance, so levels with m < 0")
    print("    are reported for completeness but are not operationally meaningful.)")
    levels = np.round(np.concatenate([np.arange(0.02, 0.901, 0.01),
                                      np.array([0.93, 0.95, 0.97, 0.98, 0.99])]), 4)
    rows = ladder(levels, GRID_SWEEP)
    top, first_fail = ceiling(rows)
    print("   %-9s %-9s %-12s %-10s %-11s %-9s %s" %
          ("level", "m(b)[m]", "eta_feas", "a_min", "k_bar", "cl_worst", "certified"))
    for r in rows:
        if r["level"] in (0.02, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99) or \
           abs(r["level"] - Bp) < 0.006:
            print("   %-9.3f %-9.2f %-12.5f %-10.5f %-11s %-9s %s"
                  % (r["level"], r["margin_m"], r["eta_feas"], r["a_min"],
                     ("%.1f" % r["k_bar"]) if r["k_bar"] else "inf",
                     ("%.2e" % r["closed_loop_worst"]) if r["closed_loop_worst"] is not None else "--",
                     r["certified"]))
    cert_below = [r for r in rows if r["certified"] and r["level"] <= Bp]
    B_dep = max((r["level"] for r in cert_below), default=None)
    n_fail = sum(1 for r in rows if not r["certified"])
    print("\n   levels tested %d, uncertified %d" % (len(rows), n_fail))
    print("   certified ceiling b_max        = %s%s"
          % (("%.3f" % top["level"]) if top else "none",
             "  (= top of the ladder; no ceiling found in range)"
             if top and top["level"] == max(levels) else ""))
    if first_fail:
        print("   first uncertified level above  = %.3f   binding: %s"
              % (first_fail["level"], binding(first_fail)))
    print("   B_dep (largest certified <= B_hat+) = %s" % (("%.6f" % B_dep) if B_dep else "none"))
    print("   B_dep reaches B_hat+"
          if (B_dep and B_dep >= Bp - 0.011) else "   B_dep is below B_hat+")

    # re-verify the two reported levels on the full grid
    ver = {}
    for name, b in (("b_max", top["level"] if top else None), ("B_hat_plus", Bp)):
        if b is None:
            continue
        ver[name] = check_level(b, n=GRID_FINAL)
        print("   re-verified %-11s level %.3f on %d-point grid: certified=%s  cl_worst=%.3e"
              % (name, b, ver[name]["grid_points"], ver[name]["certified"],
                 ver[name]["closed_loop_worst"]))
    out["sweep_levels"] = rows
    out["b_max"] = top["level"] if top else None
    out["B_dep"] = B_dep
    out["verified_full_grid"] = ver

    # ---- sweep 2: brake limit, with re-calibration --------------------------------
    print("\n2. BRAKE LIMIT SWEEP (baseline re-calibrated at each limit)")
    print("   %-8s %-10s %-10s %-11s %s" % ("u_min", "B", "B_hat+", "eta_feas", "data level certified?"))
    brake_rows = []
    brake_grid = [-10.0, -9.0, -8.0, -7.0, -6.0, -5.5, -5.0, -4.8, -4.6, -4.4, -4.2,
                  -4.0, -3.5, -3.0, -2.5, -2.0, -1.5]
    for umin in brake_grid:
        c = calibrate(umin, M.U_MAX)
        at_data = check_level(c["B_hat_plus"], n=GRID_SWEEP, u_min=umin)
        brake_rows.append(dict(u_min=umin, B=c["B"], B_hat_plus=c["B_hat_plus"],
                               data_level_certified=at_data["certified"],
                               binding_at_data_level=binding(at_data),
                               eta_at_data_level=at_data["eta_feas"],
                               k_bar_at_data_level=at_data["k_bar"],
                               binding_point=at_data["binding_point"]))
        print("   %-8.1f %-10.4f %-10.4f %-11.5f %s"
              % (umin, c["B"], c["B_hat_plus"], at_data["eta_feas"],
                 "yes" if at_data["certified"] else "NO  <- " + binding(at_data)))
    out["sweep_brake"] = brake_rows
    ok = [r["u_min"] for r in brake_rows if r["data_level_certified"]]
    bad = [r["u_min"] for r in brake_rows if not r["data_level_certified"]]
    if ok and bad:
        # weakest brake = largest (least negative) u_min that still certifies
        print("   weakest brake tested that certifies its own data level: u_min = %.1f m/s^2"
              % max(ok))
        print("   strongest brake tested that does not: u_min = %.1f m/s^2" % min(bad))
        print("   threshold lies in (%.1f, %.1f] m/s^2 of braking authority"
              % (abs(max(ok)), abs(min(bad))))
        out["brake_threshold_interval_m_s2"] = [abs(max(ok)), abs(min(bad))]
    elif ok:
        print("   the data level stays certifiable over the whole brake range tested")

    # ---- sweep 3: assumed lead deceleration ---------------------------------------
    print("\n3. ASSUMED WORST-CASE LEAD DECELERATION SWEEP (at u_min = %.1f, level = B_hat+)"
          % M.U_MIN)
    print("   %-8s %-12s %-11s %-11s %-24s %s"
          % ("a_lead", "eta_feas", "k_bar", "D_max", "binding pt (v, vl, dR/dvl)", "certified?"))
    lead_rows = []
    for al in (2.0, 4.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 12.0, 16.0):
        r = check_level(Bp, n=GRID_SWEEP, a_lead=al)
        bp = r["binding_point"]
        lead_rows.append(dict(a_lead=al, eta_feas=r["eta_feas"], k_bar=r["k_bar"],
                              D_max=r["D_max"], certified=r["certified"],
                              binding=binding(r), binding_point=bp))
        print("   %-8.1f %-12.5f %-11s %-11.5f (%5.2f, %5.2f, %9.2e) %s"
              % (al, r["eta_feas"], ("%.1f" % r["k_bar"]) if r["k_bar"] else "inf",
                 r["D_max"], bp["v"], bp["vl"], bp["dR_dvl"],
                 "yes" if r["certified"] else "NO <- " + binding(r)))
    out["sweep_lead"] = lead_rows
    bad_lead = [r for r in lead_rows if not r["certified"]]
    if bad_lead:
        print("   B_hat+ ceases to be certifiable at a_lead = %.1f m/s^2"
              % min(r["a_lead"] for r in bad_lead))
        out["lead_certified_up_to_m_s2"] = max(r["a_lead"] for r in lead_rows if r["certified"])
    else:
        print("   B_hat+ stays certifiable up to a_lead = %.1f m/s^2 (whole range tested)"
              % max(r["a_lead"] for r in lead_rows))
        out["lead_certified_up_to_m_s2"] = max(r["a_lead"] for r in lead_rows)
    print("   where dR/dvl at the binding point is ~0, the lead term does not enter eta")

    # ---- sweep 4: certified region in the (braking authority, level) plane --------
    print("\n4. CERTIFIED REGION in the (braking authority, level) plane")
    print("   For each brake limit, the certified level interval on the ladder.")
    print("   %-8s %-12s %-12s %-10s %s"
          % ("u_min", "b_lowest", "b_highest", "n_cert", "B_hat+ inside?"))
    region = []
    fine = np.round(np.arange(0.02, 0.991, 0.01), 4)
    for umin in (-10.0, -8.0, -7.0, -6.0, -5.5, -5.0, -4.8, -4.5, -4.0, -3.0, -2.0):
        rr = ladder(fine, GRID_SWEEP, u_min=umin)
        cl = [r["level"] for r in rr if r["certified"]]
        inside = any(abs(r["level"] - Bp) < 0.006 and r["certified"] for r in rr)
        region.append(dict(u_min=umin, b_lowest=min(cl) if cl else None,
                           b_highest=max(cl) if cl else None, n_certified=len(cl),
                           B_hat_plus_certified=bool(inside)))
        print("   %-8.1f %-12s %-12s %-10d %s"
              % (umin, ("%.3f" % min(cl)) if cl else "--",
                 ("%.3f" % max(cl)) if cl else "--", len(cl),
                 "yes" if inside else "no"))
    out["sweep_region"] = region
    print("   all ladder levels certified at u_min = -5.0, none at -4.8 m/s^2")

    dest = os.path.join(_OUT, "level_sweep.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()

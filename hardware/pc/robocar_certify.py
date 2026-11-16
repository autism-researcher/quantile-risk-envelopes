#!/usr/bin/env python3
"""Conditions of Theorem 1 on the RoboCar boundary, evaluated on a grid (Section 6.3).

Grid over (v, vl) in [0, VMAX]^2 with s placed on {R = b}. Reported per level:
  eta_feas  = min[-(L_fR + min_U a u) - D]   margin of condition (15)
  k_bar     = (Delta u)^2 / eta_feas         gain bound for the projected law
  closed-loop worst at k0 = ceil(k_bar)
Results are given for the stated constants and for scaled constants
(|U_MIN| * --umin-scale, U_MAX * --umax-scale, DELTA_V * --delta-scale).

  python robocar_certify.py --params params.json                (levels 0.05..0.60)
  python robocar_certify.py --params params.json --level 0.31   (one level)
Output: certificate_results.json
"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "car"))
import robocar_model as TM

GRID = 201


def check(car, b, k0=None):
    v, vl = np.meshgrid(np.linspace(0, car.VMAX, GRID), np.linspace(0, car.VMAX, GRID))
    v, vl = v.ravel(), vl.ravel()
    s = car.margin_of_level(b) + car.S0 + car.H * v + (v * v - vl * vl) / (2 * car.BD)
    keep = s >= 0
    s, v, vl = s[keep], v[keep], vl[keep]
    z = np.clip(-(s - car.S0 - car.H * v - (v * v - vl * vl) / (2 * car.BD)) / car.ELL, -60, 60)
    R = 1 / (1 + np.exp(-z)); sp = R * (1 - R)
    Rs, Rv, Rl = -sp / car.ELL, sp * (car.H + v / car.BD) / car.ELL, -sp * vl / (car.BD * car.ELL)
    LfR = Rs * (vl - v)
    D = np.abs(Rv) * car.DELTA_V + np.abs(Rl) * car.A_LEAD
    un = np.clip(car.KV * (vl - v) + car.KS * (s - car.G0 - car.TH * v), car.U_MIN, car.U_MAX)
    nom = LfR + Rv * un
    mind = LfR + np.minimum(Rv * car.U_MIN, Rv * car.U_MAX)
    eta = float(np.min(-mind - D))
    du = car.U_MAX - car.U_MIN
    kbar = du ** 2 / eta if eta > 0 else math.inf
    if k0 is None:
        k0 = math.ceil(kbar) if math.isfinite(kbar) else None
    out = dict(level=b, margin_m=car.margin_of_level(b), grid_points=int(len(s)),
               max_abs_R_minus_b=float(np.max(np.abs(R - b))),
               G_bd=float(np.max(np.sqrt(Rs ** 2 + Rv ** 2 + Rl ** 2))), D_max=float(D.max()),
               c_nom=float(nom.max()), a_min=float(np.min(np.abs(Rv))),
               eta_feas=eta, input_feasible=bool(eta > 0), k_bar=kbar, k0_used=k0)
    if k0 is not None:
        u = np.clip(un - k0 * Rv, car.U_MIN, car.U_MAX)
        cl = LfR + Rv * u + D
        out["closed_loop_worst"] = float(cl.max()); out["closed_loop_ok"] = bool(cl.max() <= 0)
        i = int(np.argmin(-mind - D))
        out["binding_point"] = dict(s=float(s[i]), v=float(v[i]), vl=float(vl[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--level", type=float, default=None)
    ap.add_argument("--umin-scale", type=float, default=0.9)
    ap.add_argument("--umax-scale", type=float, default=0.9)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--out", default="certificate_results.json")
    a = ap.parse_args()
    car = TM.Car(a.params)
    cons = TM.Car(a.params, U_MIN=car.U_MIN * a.umin_scale, U_MAX=car.U_MAX * a.umax_scale,
                  DELTA_V=car.DELTA_V * a.delta_scale, BD=car.BD)
    levels = [a.level] if a.level else [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    res = dict(params=a.params, conservative=dict(umin_scale=a.umin_scale, umax_scale=a.umax_scale,
                                                  delta_scale=a.delta_scale),
               measured=[check(car, b) for b in levels], conservative_results=[check(cons, b) for b in levels])
    res["go"] = all(r["input_feasible"] for r in res["conservative_results"])
    json.dump(res, open(a.out, "w"), indent=1)
    for lab, rr in (("measured", res["measured"]), ("conservative", res["conservative_results"])):
        print(lab)
        for r in rr:
            print("  level %.2f  eta %+.4f  k_bar %s  closed-loop worst %s" %
                  (r["level"], r["eta_feas"], "inf" if not math.isfinite(r["k_bar"]) else "%.1f" % r["k_bar"],
                   "%.4f" % r["closed_loop_worst"] if "closed_loop_worst" in r else "-"))
    print("condition (15) holds at all levels (scaled constants): %s" % ("yes" if res["go"] else "no"))


if __name__ == "__main__":
    main()

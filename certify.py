"""Step 2 -- check the hypotheses of the invariance theorem on the deployed boundary.

Boundary {R = b} is parametrized exactly by (v, vl) in the operating domain [0,30]^2:
s = m_b + S0 + H v + (v^2 - vl^2)/(2 BD), where m_b = margin_of_level(b).
Computed on a dense grid (GRID x GRID points) -- a numerical grid check, not an
interval-arithmetic proof.

Reported constants (all maxima/minima over the boundary grid):
  G_bd     = max ||grad R||
  D_max    = max sup_d grad R . d = |R_v| DELTA_V + |R_vl| A_LEAD   (worst-case disturbance)
  c_nom    = max (L_f R + a u_nom)            nominal drift (positive => nominal alone unsafe)
  a_min    = min |a(x)|                        first-order authority
  eta_feas = min [ -(L_f R + min_U a u) - D ]  input-feasibility margin incl. disturbance
  k_bar    = (Delta u)^2 / eta_feas            explicit gain bound for the projected law:
             a u(x) <= min_U a u + (Delta u)^2/k  (coordinatewise clamping, 1-D input here)
  k0_unproj= (c_nom + D_max)/a_min^2           gain condition if projection were inactive
  closed-loop worst = max (L_f R + a u_sup + D) at gain k0 = ceil(k_bar)  (must be <= 0)

Output: certificate.json.   Run: python certify.py   (after calibrate.py)
"""
import json, math, numpy as np, model as M

GRID = 601

def boundary(b, n=GRID):
    v, vl = np.meshgrid(np.linspace(0, 30, n), np.linspace(0, 30, n))
    v, vl = v.ravel(), vl.ravel()
    s = M.margin_of_level(b) + M.S0 + M.H*v + (v*v - vl*vl)/(2*M.BD)
    keep = s >= 0
    return s[keep], v[keep], vl[keep]

def check(b, k0=None):
    s, v, vl = boundary(b)
    R, Rs, Rv, Rl = M.risk_grad(s, v, vl)
    _, LfR, a, D = M.drift_terms(s, v, vl)
    G = np.sqrt(Rs**2 + Rv**2 + Rl**2)
    un = M.u_nominal(s, v, vl)
    nom = LfR + a*un
    mindrift = LfR + np.minimum(a*M.U_MIN, a*M.U_MAX)
    eta = float(np.min(-mindrift - D))
    du = M.U_MAX - M.U_MIN
    kbar = du**2/eta if eta > 0 else math.inf
    a_min = float(np.min(np.abs(a)))
    if k0 is None:
        k0 = math.ceil(kbar) if math.isfinite(kbar) else None
    out = dict(level=b, margin_m=float(M.margin_of_level(b)), grid_points=int(len(s)),
               max_abs_R_minus_b=float(np.max(np.abs(R - b))),
               G_bd=float(G.max()), D_max=float(D.max()), c_nom=float(nom.max()),
               nominal_unsafe_fraction=float(np.mean(nom + D > 0)),
               a_min=a_min, a_max=float(np.max(np.abs(a))),
               near_singular_boundary_points=0 if a_min > 0 else int(np.sum(a == 0)),
               eta_feas=eta, input_feasible=bool(eta > 0),
               k_bar=kbar, k0_unprojected=float((nom.max() + D.max())/a_min**2) if a_min > 0 else math.inf,
               k0_used=k0)
    if k0 is not None:
        u = np.clip(un - k0*a, M.U_MIN, M.U_MAX)
        cl = LfR + a*u + D
        out["closed_loop_worst"] = float(cl.max())
        out["closed_loop_ok"] = bool(cl.max() <= 0)
        # smallest gain on a coarse ladder that passes the direct closed-loop check
        for kk in (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000):
            if (LfR + a*np.clip(un - kk*a, M.U_MIN, M.U_MAX) + D).max() <= 0:
                out["smallest_ladder_gain_passing"] = kk; break
    return out

def main():
    cal = json.load(open("calibration.json"))
    res = {"deployed_level": "B_hat_plus",
           "B_hat_plus": check(cal["B_hat_plus"]),
           "B": check(cal["B"])}
    json.dump(res, open("certificate.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()

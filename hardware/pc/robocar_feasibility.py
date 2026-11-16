"""RoboCar feasibility study (Section 6.3): numbers for Table 6 and Fig. 8.

Usage: python robocar_feasibility.py --logs ../data/ident_logs --out ../results
Reads the braking steps (BRK-*, BRKI-*) and random-input runs (RND-*; INV-11..13 fitting,
INV-14..18 held out), estimates delta = |v_dot - u| over 0.24 s windows (input shifted by
0.24 s), and evaluates the margin of condition (15) with robocar_certify.py for a range of
assumed delta and braking limits.
"""
import argparse, csv, json, os, re, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "car"))
import robocar_model as TM

def load(f):
    r = list(csv.reader(open(f))); h = r[1]; rows = [dict(zip(h, x)) for x in r[2:]]
    g = lambda k: np.array([float(x[k]) for x in rows])
    return g("t"), g("u_cmd"), g("v_meas"), [x["phase"] for x in rows]

def brake_step(f):
    t, u, v, ph = load(f); i0 = ph.index("brake")
    stop = next((i for i in range(i0, len(v)) if v[i] < 0.02), len(v) - 1)
    return dict(v0=float(v[i0]), t_stop=float(t[stop] - t[i0]), mean=float(v[i0] / (t[stop] - t[i0])),
                t=(t[i0 - 10:stop + 3] - t[i0]).tolist(), v=v[i0 - 10:stop + 3].tolist())

def resid(f, lag=6, win=6, dt=0.04):
    t, u, v, ph = load(f); out = []
    for k in range(len(v) - win):
        if v[k] > 0.05 and ph[k + win] == "random" and k + 1 - lag >= 0:
            out.append(((v[k + win] - v[k]) - u[k + 1 - lag:k + win + 1 - lag].sum() * dt) / (win * dt))
    return np.abs(np.array(out))

def margin(brake, delta, tmp):
    """Normalised feasibility margin (m/s^2) at level 0.5, where b(1-b)/ell = 1 for ell = 0.25."""
    p = dict(TM.DEFAULTS); p.update(U_MIN=-brake, U_MAX=1.0, DELTA_V=delta, A_LEAD=0.5, H=1.0, VMAX=0.6, BD=0.8 * brake)
    json.dump(p, open(tmp, "w"))
    o = subprocess.run([sys.executable, os.path.join(HERE, "robocar_certify.py"), "--params", tmp,
                        "--umin-scale", "1.0", "--umax-scale", "1.0", "--out", tmp + ".out"],
                       capture_output=True, text=True).stdout
    m = re.findall(r"level 0\.50\s+eta ([+-][0-9.]+)", o)
    return float(m[0])

ap = argparse.ArgumentParser(); ap.add_argument("--logs", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); L = lambda n: os.path.join(a.logs, "ident_%s.csv" % n)
res = {}
res["brake_inverse"] = {n: brake_step(L(n)) for n in ["BRKI-06-1", "BRKI-06-2", "BRKI-06-3", "BRKI-06-4"]}
res["brake_ramp"] = {n: brake_step(L(n)) for n in ["BRK-04-1", "BRK-06-1", "BRK-08-1"]}
fit = np.concatenate([resid(L("INV-%d" % i)) for i in (11, 12, 13)])
ho = np.concatenate([resid(L("INV-%d" % i)) for i in range(14, 19)])
rmp = np.concatenate([resid(L("RND-0%d" % i)) for i in (1, 3, 4, 5)])  # RND-02 excluded (wall contact)
res["delta_ramp"] = dict(n=len(rmp), max=float(rmp.max()), p95=float(np.percentile(rmp, 95)))
res["delta_fit"] = dict(n=len(fit), max=float(fit.max()), p95=float(np.percentile(fit, 95)))
res["delta_holdout"] = dict(n=len(ho), max=float(ho.max()), p95=float(np.percentile(ho, 95)))
bi = [x["mean"] for x in res["brake_inverse"].values()]
res["brake_inverse_summary"] = dict(min=min(bi), mean=float(np.mean(bi)), max=max(bi))
tmp = os.path.join(a.out, "_cert_tmp.json"); deltas = np.round(np.arange(0.0, 1.41, 0.05), 2)
res["margin_curve"] = {"%.2f" % b: [margin(b, d, tmp) for d in deltas] for b in (min(bi), float(np.mean(bi)), max(bi))}
res["deltas"] = deltas.tolist()
res["margin_points"] = {"brake %.2f delta %.2f" % (b, d): margin(b, d, tmp)
                        for b in (min(bi), float(np.mean(bi)), max(bi)) for d in (res["delta_holdout"]["max"], res["delta_fit"]["max"])}
# largest delta with a non-negative margin (linear interpolation on the curve), per braking value
res["largest_feasible_delta"] = {b: float(np.interp(0.0, -np.array(c), deltas)) for b, c in res["margin_curve"].items()}
# ramp mode: best measured ramp braking with the ramp-mode delta, over a range of risk designs
def margin_design(brake, delta, H, vmax):
    p = dict(TM.DEFAULTS); p.update(U_MIN=-brake, U_MAX=1.0, DELTA_V=delta, A_LEAD=0.5, H=H, VMAX=vmax, BD=0.8 * brake)
    json.dump(p, open(tmp, "w"))
    o = subprocess.run([sys.executable, os.path.join(HERE, "robocar_certify.py"), "--params", tmp,
                        "--umin-scale", "1.0", "--umax-scale", "1.0", "--out", tmp + ".out"],
                       capture_output=True, text=True).stdout
    return float(re.findall(r"level 0\.50\s+eta ([+-][0-9.]+)", o)[0])
rb = max(x["mean"] for x in res["brake_ramp"].values())
res["ramp_design_sweep"] = {"braking": rb, "delta": res["delta_ramp"]["max"],
    "margins": {"H=%.1f,vmax=%.1f" % (H, vm): margin_design(rb, res["delta_ramp"]["max"], H, vm)
                for H in (0.4, 1.0, 1.5, 2.0) for vm in (0.5, 0.6, 1.0)}}
for f in (tmp, tmp + ".out"):
    if os.path.exists(f): os.remove(f)
json.dump(res, open(os.path.join(a.out, "robocar_feasibility.json"), "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # embed TrueType fonts
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.75), constrained_layout=True)
for i, (n, s) in enumerate(res["brake_inverse"].items()):
    ax[0].plot(s["t"], s["v"], drawstyle="steps-post", lw=0.9, label="run %d" % (i + 1))
ax[0].axvline(0, color="k", lw=0.5, ls=":"); ax[0].set_xlabel("time from brake command (s)")
ax[0].set_ylabel("measured speed (m/s)"); ax[0].set_title("(a)", fontsize=8); ax[0].set_xlim(-0.4, 0.9)
sty = ["--", "-", ":"]
for (b, c), s in zip(res["margin_curve"].items(), sty):
    ax[1].plot(deltas, c, "k" + s, lw=0.9, label=r"$|u_{\min}|=%s$" % b)
ax[1].axhline(0, color="k", lw=0.5)
ax[1].axvspan(res["delta_holdout"]["max"], res["delta_fit"]["max"], color="0.85", lw=0)
ax[1].set_xlabel(r"assumed $\delta$ (m/s$^2$)"); ax[1].set_ylabel("margin (m/s)"); ax[1].set_title("(b)", fontsize=8)
ax[1].legend(fontsize=6, frameon=True, framealpha=1, edgecolor="none", loc="upper right", handlelength=1.5, borderpad=0.2)
fig.savefig(os.path.join(a.out, "fig_robocar.pdf")); fig.savefig(os.path.join(a.out, "fig_robocar.png"), dpi=200)
print(json.dumps({k: v for k, v in res.items() if k not in ("margin_curve", "brake_inverse", "brake_ramp", "deltas")}, indent=1))
print({n: round(s["mean"], 2) for n, s in {**res["brake_inverse"], **res["brake_ramp"]}.items()})

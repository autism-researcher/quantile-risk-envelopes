"""Figures for the double-integrator example. Run after di_run.py."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import di_model as D
plt.rcParams.update({"font.size": 9, "figure.figsize": (3.5, 2.4), "axes.grid": True, "grid.alpha": 0.3})
LAB = {"nominal": "Nominal PD", "proposed": "Proposed supervisor", "cbf_robust": "Robust CBF-QP"}
STY = {"nominal": dict(color="#888888", ls=":"), "proposed": dict(color="#1f5fa8", ls="-"),
       "cbf_robust": dict(color="#d68910", ls="-.")}
res = json.load(open("di_results.json"))
B = res["tau_pos"]["calibration"]["B_hat_plus"]; k0 = res["tau_pos"]["k0_used"]
fig, ax = plt.subplots()
for k in LAB:
    d = pd.read_csv(f"di_traj_{k}.csv"); ax.plot(d.time, d.x1, lw=1.1, label=LAB[k], **STY[k])
ax.axhline(D.X_BAR, color="k", lw=0.8); ax.text(0.3, D.X_BAR + 0.15, "wall $\\bar x$", fontsize=8)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Position $x_1$"); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig("fig_di_position.pdf"); plt.close(fig)
# boundary drift vs x2 for tau>0 (certified) and tau=0 (not certifiable)
fig, ax = plt.subplots()
for tag, tau, kk, col in (("tau_pos", 0.5, k0, "#1f5fa8"), ("tau_zero", 0.0, res["tau_zero"]["k0_used"], "#c0392b")):
    b = res[tag]["calibration"]["B_hat_plus"]
    x2 = np.linspace(-3, 3, 6001)
    x1 = D.X_BAR - tau*x2 - x2*np.abs(x2)/(2*D.BRK) - D.h_of_level(b)
    R, R1, a = D.risk_grad(x1, x2, tau)
    u = np.clip(D.u_nominal(x1, x2, 50.0) - kk*a, -D.U_BAR, D.U_BAR)   # nominal pushing toward the wall
    ax.plot(x2, (R1*x2 + a*u + np.abs(a)*D.DELTA)/(b*(1-b)), color=col, lw=1.1,
            label=(f"$\\tau={tau}$, $k_0={kk}$" + (" (certified)" if tau > 0 else " (not certifiable)")))
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Velocity $x_2$ on $\\partial\\mathcal{S}$"); ax.set_ylabel("worst-case $D^+R\\,/\\,\\sigma'$")
ax.legend(fontsize=6.5); fig.tight_layout(); fig.savefig("fig_di_certificate.pdf"); plt.close(fig)

"""Vehicle-following figures from the CSV/JSON outputs. The boundary map (fig_certificate)
re-evaluates the closed-loop boundary drift on the same grid as certify.py.
Run: python plot.py"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M, certify as C

LAB = {"nominal": "Nominal ACC", "proposed": "Proposed supervisor",
       "cbf": "CBF-QP", "cbf_robust": "Robust CBF-QP"}
STY = {"nominal": dict(color="#888888", ls=":"), "proposed": dict(color="#1f5fa8", ls="-"),
       "cbf": dict(color="#c0392b", ls="--"), "cbf_robust": dict(color="#d68910", ls="-.")}
plt.rcParams.update({"font.size": 9, "figure.figsize": (3.5, 2.4), "axes.grid": True,
                     "grid.alpha": 0.3})

def save(fig, name):
    fig.tight_layout(); fig.savefig(name); plt.close(fig)

def main():
    cal = json.load(open("calibration.json")); B = cal["B_hat_plus"]
    T = {k: pd.read_csv(f"traj_{k}.csv") for k in LAB}
    for col, ylabel, fname in (("risk", "Risk $R(x)$", "fig_risk.pdf"),
                               ("spacing", "Spacing $s$ [m]", "fig_spacing.pdf"),
                               ("velocity", "Ego speed $v$ [m/s]", "fig_velocity.pdf")):
        fig, ax = plt.subplots()
        for k, d in T.items():
            ax.plot(d.time, d[col], lw=1.1, label=LAB[k], **STY[k])
        if col == "risk":
            ax.axhline(B, color="k", lw=0.8); ax.text(1, B + 0.01, r"$\widehat B_N^+$", fontsize=8)
        if col == "velocity":
            ax.plot(T["proposed"].time, T["proposed"].lead_velocity, color="k", lw=0.7,
                    alpha=0.6, label="Lead speed")
        ax.set_xlabel("Time [s]"); ax.set_ylabel(ylabel); ax.legend(fontsize=6.5)
        save(fig, fname)
    # control input: one panel per controller; nominal input in gray for reference
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(3.5, 4.6))
    for ax, k in zip(axs, LAB):
        if k != "nominal":
            ax.plot(T["nominal"].time, T["nominal"].control, color="#aaaaaa", lw=0.9)
        ax.plot(T[k].time, T[k].control, lw=1.0, color=STY[k]["color"])
        ax.set_ylim(-7, 2.8); ax.set_yticks([2, 0, -3, -6])
        ax.text(0.98, 0.08, LAB[k], transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, fontweight="bold", color=STY[k]["color"])
    axs[-1].set_xlabel("Time [s]"); fig.supylabel("Input $u$ [m/s$^2$]", fontsize=9)
    fig.tight_layout(h_pad=0.2); fig.savefig("fig_control.pdf"); plt.close(fig)
    fig, ax = plt.subplots()
    bins = np.linspace(-0.4, 0.3, 57)
    for k in ("nominal", "cbf", "proposed"):
        d = pd.read_csv(f"mc_{k}.csv").max_R_minus_B
        ax.hist(d, bins=bins, alpha=0.55, label=LAB[k], color=STY[k]["color"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\max_t\,[R(x(t))-\widehat B_N^+]$"); ax.set_ylabel("Runs"); ax.legend(fontsize=6.5)
    save(fig, "fig_montecarlo_histogram.pdf")
    # boundary certificate map
    cert = json.load(open("certificate.json"))["B_hat_plus"]
    n = C.GRID
    v, vl = np.meshgrid(np.linspace(0, C.VMAX, n), np.linspace(0, C.VMAX, n))
    s = M.margin_of_level(B) + M.S0 + M.H*v + (v*v - vl*vl)/(2*M.BD)
    _, LfR, a, D = M.drift_terms(s, v, vl)
    u = np.clip(M.u_nominal(s, v, vl) - cert["k0_used"]*a, M.U_MIN, M.U_MAX)
    cl = np.where(s >= 0, LfR + a*u + D, np.nan)
    fig, ax = plt.subplots()
    im = ax.contourf(v, vl, cl, 20, cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"worst-case $D^+R$ on $\partial\mathcal{S}$")
    ax.set_xlabel("Ego speed $v$ [m/s]"); ax.set_ylabel(r"Lead speed $v_\ell$ [m/s]")
    save(fig, "fig_certificate.pdf")

if __name__ == "__main__":
    main()

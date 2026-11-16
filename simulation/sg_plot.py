"""Figures for Supporting Information S-II and Section 6.2. Run after sg_run.py and level_sweep.py.

  fig_sg_conditions.pdf   (a) feasibility condition and drift alone along the boundary, with |a|
                          (b) per-start max(R - B) for boundary starts, all four controllers
  fig_level_sweep.pdf     (a) margin of condition (15) vs available braking authority
                          (b) margin of condition (15) vs assumed lead deceleration
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42     # embed TrueType fonts
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt
import sg_model as S

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# same colours as plot.py / di_plot.py
C_NOM, C_PROP, C_CBF, C_ROB = "#888888", "#1f5fa8", "#c0392b", "#d68910"
LAB = {"nominal": "Nominal PD", "proposed": "Proposed supervisor",
       "cbf": "CBF-QP", "cbf_robust": "Robust CBF-QP"}
COL = {"nominal": C_NOM, "proposed": C_PROP, "cbf": C_CBF, "cbf_robust": C_ROB}
STY = {"nominal": ":", "proposed": "-", "cbf": "--", "cbf_robust": "-."}


# ---------------------------------------------------------------- Figure: sg conditions
res = json.load(open("sg_results.json"))
b = res["calibration"]["B_hat_plus"]
cert = res["certificate"]
a_thr = cert["assumption5_band"]["a_thr_used"]
sb_over_ell = b*(1 - b)/S.ELL

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.6))

# ---- panel (a): the two conditions along the boundary
x2 = np.linspace(-2.0, 2.0, 40001)
x1 = S.x1_of_boundary(b, x2)
_, LfR, a, D, G = S.drift_terms(x1, x2)
un = S.u_nominal(x1, x2, np.full_like(x2, 50.0))
slack16 = -(LfR + np.minimum(a*(-S.U_BAR), a*S.U_BAR)) - D
slackDR = -LfR - D          # drift alone: the input has no effect

ax.axhline(0.0, color="k", lw=0.8)
ax.plot(x2, slack16, color=C_PROP, lw=1.3, label=r"feasibility condition")
ax.plot(x2, slackDR, color=C_ROB, lw=1.3, ls="-.", label=r"drift alone")
ax.axvline(S.V_W, color=C_CBF, lw=0.9, ls=":")
ax.annotate(r"$a(x)=0$" "\n" r"$x_2=V_w$", (S.V_W, 0.0), xytext=(6, 30),
            textcoords="offset points", fontsize=7, color=C_CBF,
            arrowprops=dict(arrowstyle="-", color=C_CBF, lw=0.7))
ax.plot([S.V_W], [cert["singular_point"]["slack_cond16"]], "o", ms=4.5,
        mfc="none", mec="k", mew=1.0, zorder=5)
ax.set_xlabel(r"speed $x_2$ on $\partial\mathcal{S}$ [m/s]")
ax.set_ylabel(r"condition slack [s$^{-1}$]")
ax.set_xlim(-2.0, 2.0)
ax.legend(fontsize=6.5, loc="lower right")
ax.set_title("(a) the input adds nothing where $a=0$", fontsize=8.5, loc="left")

ax2 = ax.twinx()
ax2.plot(x2, np.abs(a), color="#555555", lw=0.9, alpha=0.55)
ax2.set_ylabel(r"$|a(x)|$  (grey)", fontsize=8, color="#555555")
ax2.tick_params(labelsize=7, colors="#555555")
ax2.grid(False)
ax2.set_ylim(0, None)

# ---- panel (b): boundary-start sweep, adversarial disturbance
d = pd.read_csv("sg_boundary_adversarial.csv")
# values below FLOOR are floating-point zero (boundary starts begin at R = B exactly);
# clamping keeps them on one flat line instead of a band of rounding noise
FLOOR = 1e-16
for name in ("nominal", "cbf", "cbf_robust", "proposed"):
    y = np.maximum(d["max_R_minus_B_" + name].values, FLOOR)
    bx.semilogy(d.x2_0, y, color=COL[name], ls=STY[name], lw=1.2, label=LAB[name])
bx.axvline(S.V_W, color=C_CBF, lw=0.9, ls=":")
bx.axhline(res["montecarlo_boundary"]["adversarial"]["proposed"]["viol_tol"],
           color="k", lw=0.7, ls="--")
bx.annotate("violation tolerance", (-2.0, 1e-12), xytext=(4, 4),
            textcoords="offset points", fontsize=6.5, ha="left", color="k")
bx.annotate("floating-point zero", (2.0, FLOOR), xytext=(-4, 7),
            textcoords="offset points", fontsize=6.5, ha="right", color=C_PROP)
bx.set_xlabel(r"initial speed $x_2(0)$ on $\partial\mathcal{S}$ [m/s]")
bx.set_ylabel(r"$\max_t\,[R(x(t))-\widehat B_N^+]$")
bx.set_xlim(-2.0, 2.0)
bx.set_ylim(FLOOR/3, 30)
bx.legend(fontsize=6.5, loc="center left")
bx.set_title("(b) boundary starts, worst-case disturbance", fontsize=8.5, loc="left")

fig.tight_layout()
fig.savefig("fig_sg_conditions.pdf")
plt.close(fig)
print("wrote fig_sg_conditions.pdf")


# ---------------------------------------------------------------- Figure: level sweep
ls = json.load(open("level_sweep.json"))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.5))

br = pd.DataFrame(ls["sweep_brake"])
ax.axhline(0.0, color="k", lw=0.8)
ax.plot(-br.u_min, br.eta_at_data_level, "o-", color=C_PROP, lw=1.3, ms=3.5)
lo, hi = ls["brake_threshold_interval_m_s2"]
ax.axvspan(min(lo, hi), max(lo, hi), color=C_CBF, alpha=0.16, lw=0)
ax.annotate("certified at %.1f,\nfails at %.1f m/s$^2$" % (max(lo, hi), min(lo, hi)),
            (min(lo, hi), 0.0), xytext=(20, -38), textcoords="offset points",
            fontsize=7, color=C_CBF,
            arrowprops=dict(arrowstyle="->", color=C_CBF, lw=0.7))
ax.axvline(-ls["sweep_brake"][2]["u_min"], color="#555555", lw=0.8, ls="--")
ax.annotate("brake limit\nof the study", (8.0, br.eta_at_data_level.min()),
            xytext=(-6, 6), textcoords="offset points", fontsize=7,
            ha="right", va="bottom", color="#555555")
ax.set_xlabel(r"available braking authority $|u_{\min}|$ [m/s$^2$]")
ax.set_ylabel(r"feasibility margin $\eta$")
ax.set_title(r"(a) $\eta$ at the calibrated level $\widehat B_N^+$", fontsize=8.5, loc="left")

ld = pd.DataFrame(ls["sweep_lead"])
bx.axhline(0.0, color="k", lw=0.8)
bx.plot(ld.a_lead, ld.eta_feas, "s-", color=C_PROP, lw=1.3, ms=3.5)
bx.axvline(4.0, color="#555555", lw=0.8, ls="--")
bx.annotate("assumed in\nthe study", (4.0, ld.eta_feas.min()), xytext=(5, 6),
            textcoords="offset points", fontsize=7, va="bottom", color="#555555")
cert_to = ls["lead_certified_up_to_m_s2"]
bx.axvline(cert_to, color=C_CBF, lw=0.9, ls=":")
bx.annotate("certified up to\n%.1f m/s$^2$" % cert_to, (cert_to, 0.0),
            xytext=(-14, -40), textcoords="offset points", fontsize=7, ha="right",
            color=C_CBF, arrowprops=dict(arrowstyle="->", color=C_CBF, lw=0.7))
bx.set_xlabel(r"assumed worst-case lead deceleration [m/s$^2$]")
bx.set_ylabel(r"feasibility margin $\eta$")
bx.set_title(r"(b) sensitivity to the lead-braking bound", fontsize=8.5, loc="left")

fig.tight_layout()
fig.savefig("fig_level_sweep.pdf")
plt.close(fig)
print("wrote fig_level_sweep.pdf")

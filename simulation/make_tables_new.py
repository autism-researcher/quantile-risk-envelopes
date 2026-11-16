"""Writes tables_sweep.tex (Tables 3-4) and tables_sg.tex (Supporting Information S-II)
from level_sweep.json and sg_results*.json.

Run from the folder holding level_sweep.json, sg_results.json and
sg_results_dt0.001.json:
    python make_tables_new.py [output_dir]
Default output_dir is ./tables
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)


def j(p):
    return json.load(open(os.path.join(HERE, p)))


def sci(x, d=2):
    if x == 0:
        return "0"
    s = f"{x:.{d}e}"
    m, e = s.split("e")
    e = int(e)
    return f"{m}" if e == 0 else f"{m}\\times10^{{{e}}}"


def fx(x, d=3):
    return f"{x:.{d}f}"


ls = j("level_sweep.json")
sg = j("sg_results.json")
sg1 = j("sg_results_dt0.001.json")

# =============================================================== certified-level sweep
t = []
t.append(r"""\begin{table}[!t]
\caption{Vehicle following: condition~\eqref{eq:case_ii} margin $\eta$ at the calibrated
level $\widehat B_N^+$ as the braking authority is varied. The baseline is
re-calibrated at each limit, so $\widehat B_N^+$ is the level implied by nominal
driving under that same limit.}
\label{tab:brake_sweep}
\centering\footnotesize
\resizebox{\columnwidth}{!}{
\begin{tabular}{rccc}
\hline
$|u_{\min}|$ [m/s$^2$] & $\widehat B_N^+$ & $\eta$ at $\widehat B_N^+$ & Certified \\ \hline""")
for r in ls["sweep_brake"]:
    t.append("$%.1f$ & $%s$ & $%s$ & %s \\\\"
             % (abs(r["u_min"]), fx(r["B_hat_plus"], 4), fx(r["eta_feas_at_data_level"]
                if "eta_feas_at_data_level" in r else r["eta_at_data_level"], 4),
                "yes" if r["data_level_certified"] else "no"))
lo, hi = ls["brake_threshold_interval_m_s2"]
t.append(r"\hline")
t.append("\\multicolumn{4}{l}{Threshold between $%.1f$ and $%.1f$~m/s$^2$; "
         "every failure is condition~\\eqref{eq:case_ii},} \\\\" % (min(lo, hi), max(lo, hi)))
t.append(r"\multicolumn{4}{l}{never the closed-loop boundary check.} \\")
t.append(r"\hline" "\n" r"\end{tabular}}" "\n" r"\end{table}")

t.append(r"""
\begin{table}[!t]
\caption{Vehicle following: sensitivity of the certificate at $\widehat B_N^+$ to the
assumed worst-case lead deceleration. Below $6$~m/s$^2$ the binding boundary point has
$\partial R/\partial v_\ell=0$, so the lead term drops out of $\eta$ there.}
\label{tab:lead_sweep}
\centering\footnotesize
\resizebox{\columnwidth}{!}{
\begin{tabular}{rcccc}
\hline
Lead decel.\ [m/s$^2$] & $D_{\max}$ & $\eta$ & $\bar k$ & Certified \\ \hline""")
for r in ls["sweep_lead"]:
    t.append("$%.1f$ & $%s$ & $%s$ & %s & %s \\\\"
             % (r["a_lead"], fx(r["D_max"], 3), fx(r["eta_feas"], 4),
                ("$%.1f$" % r["k_bar"]) if r["k_bar"] else r"$\infty$",
                "yes" if r["certified"] else "no"))
t.append(r"\hline")
t.append("\\multicolumn{5}{l}{Certified up to $%.1f$~m/s$^2$.} \\\\"
         % ls["lead_certified_up_to_m_s2"])
t.append(r"\hline" "\n" r"\end{tabular}}" "\n" r"\end{table}")
open(os.path.join(OUT, "tables_sweep.tex"), "w").write("\n".join(t) + "\n")

# ============================================================== singular-authority example
c = sg["certificate"]
sp = c["singular_point"]
bd = c["assumption5_band"]
cii, ci = c["case_ii"], c["case_i"]
v = []
v.append(r"""\begin{table}[!t]
\caption{Singular-authority example: boundary certificate at $\widehat B_N^+$ over the
operating domain $|x_2|\le%.0f$~m/s (%d grid points).}
\label{tab:sg_certificate}
\centering\footnotesize
\resizebox{\columnwidth}{!}{
\begin{tabular}{lc}
\hline
Quantity & Value \\ \hline""" % (c["domain_x2max"], c["grid_points"]))
rows = [
    ("Baseline episodes $N$", f"{sg['calibration']['N']}"),
    ("$B_N$ / $\\widehat B_N^+$",
     f"${fx(sg['calibration']['B'],4)}$ / ${fx(sg['calibration']['B_hat_plus'],4)}$"),
    ("$a_{\\min}$ on $\\partial\\mathcal{S}$", f"${sci(c['a_min'])}$ (at $x_2=%s$~m/s)" % fx(sp["x2"], 2)),
    ("$\\max\\|\\nabla R\\|$ on $\\partial\\mathcal{S}$", f"${fx(c['G_bd'],3)}$"),
    ("Worst disturbance term $D_{\\max}$", f"${fx(c['D_max'],3)}$"),
    ("Nominal drift bound $c_{\\mathrm{nom}}$", f"${fx(c['c_nom'],3)}$"),
    ("\\multicolumn{2}{l}{\\emph{At the singular point} $a(x)=0$:}", None),
    ("\\quad slack of \\eqref{eq:case_ii}", f"${fx(sp['slack_cond16'],8)}$"),
    ("\\quad drift-only slack $-(L_fR+D)$", f"${fx(-(sp['LfR']+sp['D']),8)}$"),
    ("\\multicolumn{2}{l}{\\emph{Case (ii), Lemma~\\ref{lem:proj_feasibility}} "
     "(no bound on $\\|a\\|$ needed):}", None),
    ("\\quad $\\eta$ on the whole boundary", f"${fx(cii['eta_whole_boundary'],5)}$"),
    ("\\quad $\\bar k$ on the whole boundary", f"${cii['k_bar_whole_boundary']:.1f}$"),
    ("\\multicolumn{2}{l}{\\emph{Case (i), gain condition} \\eqref{eq:k_condition}:}", None),
    ("\\quad $k_0$ on the whole boundary", r"$\infty$" if ci["infeasible_without_band"]
     else f"${ci['k0_whole_boundary']:.1f}$"),
    ("Gain used $k_0=\\lceil\\bar k\\rceil$", f"${c['k0_used']}$"),
    ("Worst closed-loop $D^+R$ on $\\partial\\mathcal{S}$", f"${sci(c['closed_loop_worst'])}$"),
]
for a, b in rows:
    v.append(f"{a} \\\\" if b is None else f"{a} & {b} \\\\")
v.append(r"\hline" "\n" r"\end{tabular}}" "\n" r"\end{table}")

v.append(r"""
\begin{table*}[!t]
\caption{Singular-authority example: envelope violations over %d starts placed
\emph{on} $\partial\mathcal{S}$, with $x_2(0)$ swept through the singular speed
$V_w$. Worst $\max_t[R-\widehat B_N^+]$ in parentheses; $k_0=%d$. Values at
$10^{-16}$ and below are floating-point zero, since the runs start at $R=\widehat
B_N^+$ exactly.}
\label{tab:sg_boundary}
\centering\footnotesize
\begin{tabular}{lcccc}
\hline
& \multicolumn{2}{c}{Random $w$} & \multicolumn{2}{c}{Adversarial $w$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Controller & $\Delta t=10^{-2}$ & $10^{-3}$ & $\Delta t=10^{-2}$ & $10^{-3}$ \\ \hline"""
         % (sg["montecarlo_boundary"]["random"]["proposed"]["starts"], c["k0_used"]))
for k, lab in (("nominal", "Nominal PD"), ("cbf", "CBF-QP"),
               ("cbf_robust", "Robust CBF-QP"), ("proposed", "Proposed supervisor")):
    cells = []
    for src in (sg, sg1):
        for scen in ("random", "adversarial"):
            e = src["montecarlo_boundary"][scen][k]
            cells.append((scen, e))
    r0 = sg["montecarlo_boundary"]["random"][k]
    r1 = sg1["montecarlo_boundary"]["random"][k]
    a0 = sg["montecarlo_boundary"]["adversarial"][k]
    a1 = sg1["montecarlo_boundary"]["adversarial"][k]
    v.append("%s & %d (${%s}$) & %d (${%s}$) & %d (${%s}$) & %d (${%s}$) \\\\"
             % (lab,
                r0["violations"], sci(r0["worst_max_R_minus_B"]),
                r1["violations"], sci(r1["worst_max_R_minus_B"]),
                a0["violations"], sci(a0["worst_max_R_minus_B"]),
                a1["violations"], sci(a1["worst_max_R_minus_B"])))
v.append(r"\hline")
v.append("\\multicolumn{5}{l}{Violation count uses the tolerance $%s$.} \\\\"
         % sci(sg["montecarlo_boundary"]["adversarial"]["proposed"]["viol_tol"]))
v.append(r"\hline" "\n" r"\end{tabular}" "\n" r"\end{table*}")
open(os.path.join(OUT, "tables_sg.tex"), "w").write("\n".join(v) + "\n")

print("wrote %s/tables_sweep.tex and %s/tables_sg.tex" % (OUT, OUT))

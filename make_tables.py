"""Writes latex/tables_*.tex directly from the JSON results, so every number in the
tables is machine-copied (no hand typing). Run after the simulations:
   python make_tables.py      (from the IEEE_Access_submission folder)"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "code"); L = os.path.join(HERE, "latex")
def j(p): return json.load(open(p))
def sci(x, d=2):
    if x == 0: return "0"
    s = f"{x:.{d}e}"; m, e = s.split("e"); e = int(e)
    return f"{m}" if e == 0 else f"{m}\\times10^{{{e}}}"
def fx(x, d=3): return f"{x:.{d}f}"
def vi(e):  return f"{e['violations']}"

di = j(os.path.join(C, "double_integrator", "di_results.json"))
p, A, Z = di["parameters"], di["tau_pos"], di["tau_zero"]
t = []
t.append(r"""\begin{table}[!t]
\caption{Double integrator: calibration and closed-form certificate on $\partial\mathcal{S}$ (grid cross-check in parentheses).}
\label{tab:di_certificate}
\centering\footnotesize
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcc}
\hline
Quantity & $\tau=%s$ & $\tau=0$ \\ \hline""" % p["TAU"])
t.append(f"$B_N$ & ${fx(A['calibration']['B'],4)}$ & ${fx(Z['calibration']['B'],4)}$ \\\\")
t.append(f"$\\widehat B_N^+$ (deployed) & ${fx(A['calibration']['B_hat_plus'],4)}$ & ${fx(Z['calibration']['B_hat_plus'],4)}$ \\\\")
t.append(f"Margin $h$ on $\\partial\\mathcal{{S}}$ & ${fx(A['analytic']['margin_h'],2)}$ & ${fx(Z['analytic']['margin_h'],2)}$ \\\\")
t.append(f"$a_{{\\min}}$ & ${sci(A['analytic']['a_min'])}$ (${sci(A['grid']['a_min_grid'])}$) & $0$ ($0$) \\\\")
t.append(f"Feasibility margin $\\eta$ & ${sci(A['analytic']['eta'])}$ (${sci(A['grid']['eta_grid'])}$) & $0$ ($0$) \\\\")
t.append(f"Gain bound $\\bar k=(\\Delta u)^2/\\eta$ & ${A['analytic']['k_bar']:.1f}$ & $\\infty$ \\\\")
t.append(f"Gain used $k_0$ & ${A['k0_used']}$ & ${Z['k0_used']}$ (uncertified) \\\\")
t.append(f"Worst closed-loop $D^+R$ on $\\partial\\mathcal{{S}}$ & ${sci(A['grid']['closed_loop_worst'])}$ & ${sci(Z['grid']['closed_loop_worst'])}$ \\\\ \\hline")
t.append(r"\end{tabular}}" "\n" r"\end{table}")
t.append(r"""
\begin{table}[!t]
\caption{Double integrator: envelope violations over $1000$ runs (worst $\max_t[R-\widehat B_N^+]$ in parentheses).}
\label{tab:di_mc}
\centering\footnotesize
\resizebox{\columnwidth}{!}{
\begin{tabular}{llcc}
\hline
$\tau$ & Controller & Random $w$ & Adversarial $w$ \\ \hline""")
names = (("nominal", "Nominal PD"), ("proposed", "Proposed supervisor"), ("cbf_robust", "Robust CBF-QP"))
for tag, tau in (("tau_pos", p["TAU"]), ("tau_zero", 0.0)):
    E = di[tag]
    for i, (k, lab) in enumerate(names):
        t.append(f"{('$'+str(tau)+'$') if i==0 else ''} & {lab} & {vi(E['random_'+k])} (${sci(E['random_'+k]['worst'])}$) & {vi(E['adversarial_'+k])} (${sci(E['adversarial_'+k]['worst'])}$) \\\\")
    t.append(r"\hline")
t.append(r"\end{tabular}}" "\n" r"\end{table}")
open(os.path.join(L, "tables_di.tex"), "w").write("\n".join(t) + "\n")

cal = j(os.path.join(C, "calibration.json")); cert = j(os.path.join(C, "certificate.json"))["B_hat_plus"]
sm = j(os.path.join(C, "simulation_summary.json"))
v = []
v.append(r"""\begin{table}[!t]
\caption{Vehicle following: calibration and boundary certificate at $\widehat B_N^+$ (grid of %d boundary points).}
\label{tab:veh_certificate}
\centering\footnotesize
\resizebox{\columnwidth}{!}{
\begin{tabular}{lc}
\hline
Quantity & Value \\ \hline""" % cert["grid_points"])
rows = [("Baseline episodes $N$", f"{cal['N']}"),
        ("$B_N$ / stopping margin", f"${fx(cal['B'],4)}$ / ${fx(cal['margin_at_B_m'],2)}$~m"),
        ("$\\epsilon_N$", f"${fx(cal['eps_N'],4)}$"),
        ("$\\widehat B_N^+$ / stopping margin", f"${fx(cal['B_hat_plus'],4)}$ / ${fx(cal['margin_at_B_hat_plus_m'],2)}$~m"),
        ("$\\max\\|\\nabla R\\|$ on $\\partial\\mathcal{S}$", f"${fx(cert['G_bd'],3)}$"),
        ("Worst disturbance term $D_{\\max}$", f"${fx(cert['D_max'],3)}$"),
        ("Nominal drift bound $c_{\\mathrm{nom}}$", f"${fx(cert['c_nom'],3)}$"),
        ("Boundary points where nominal law fails", f"${100*cert['nominal_unsafe_fraction']:.0f}\\%$"),
        ("Authority $|a(x)|$ range", f"$[{fx(cert['a_min'],4)},\\,{fx(cert['a_max'],4)}]$"),
        ("Feasibility margin $\\eta$", f"${fx(cert['eta_feas'],4)}$"),
        ("Gain bound $\\bar k$ / $k_0$ used", f"${cert['k_bar']:.1f}$ / ${cert['k0_used']}$"),
        ("Worst closed-loop $D^+R$ on $\\partial\\mathcal{S}$", f"${fx(cert['closed_loop_worst'],4)}$")]
for a, b in rows: v.append(f"{a} & {b} \\\\")
v.append(r"\hline" "\n" r"\end{tabular}}" "\n" r"\end{table}")
v.append(r"""
\begin{table}[!t]
\caption{Vehicle following: envelope violations over $1000$ runs (worst $\max_t[R-\widehat B_N^+]$ in parentheses); $k_0=%s$.}
\label{tab:veh_mc}
\centering\footnotesize
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcc}
\hline
Controller & Random braking & Adversarial \\ \hline""" % sm["k0"])
for k, lab in (("nominal", "Nominal ACC"), ("proposed", "Proposed supervisor"), ("cbf", "CBF-QP"), ("cbf_robust", "Robust CBF-QP")):
    v.append(f"{lab} & {vi(sm['mc_'+k])} (${sci(sm['mc_'+k]['worst_max_R_minus_B'])}$) & {vi(sm['adv_'+k])} (${sci(sm['adv_'+k]['worst_max_R_minus_B'])}$) \\\\")
v.append(r"\hline" "\n" f"\\multicolumn{{3}}{{l}}{{Collisions (adversarial): nominal {sm['adv_nominal']['collisions']}, others 0.}} \\\\")
v.append(r"\hline" "\n" r"\end{tabular}}" "\n" r"\end{table}")
open(os.path.join(L, "tables_vehicle.tex"), "w").write("\n".join(v) + "\n")
print("tables written")

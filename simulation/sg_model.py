"""Vanishing control authority example (Supporting Information S-II).

The input gain vanishes at one boundary point, where the drift is strictly inward, so the
singular set on the boundary is non-empty and certifiable.

Plant. A body moving along x1 toward a wall at x1 = X_BAR, actuated by a control surface
whose effectiveness scales with dynamic pressure relative to a steady flow of speed V_W:

    x1' = x2
    x2' = beta(x2) u + w,      |u| <= U_BAR,  |w| <= DELTA
    beta(x2) = (x2 - V_W)^2 / ((x2 - V_W)^2 + C_BETA)

so f(x) = [x2, 0],  g(x) = [0, beta(x2)]^T,  d = [0, w].
beta(V_W) = 0: at the flow speed the surface has zero relative airspeed and no authority.
V_W < 0, so the singular speed is directed away from the wall and the drift there is
strictly inward. The disturbance is not attenuated by beta, so D(x) stays bounded away
from zero where a(x) -> 0.

Risk. Same braking barrier as the double integrator (Supporting Information S-I):
    h(x) = X_BAR - x1 - TAU*x2 - x2|x2|/(2*BRK),     R(x) = sigma(-h(x)/ELL).
On {R = b}, with s_b := b(1-b) and q(x2) := TAU + |x2|/BRK:
    grad R  = (s_b/ELL) [1, q],            ||grad R|| = (s_b/ELL) sqrt(1 + q^2)
    a(x)    = (s_b/ELL) q beta(x2)         -> 0 as x2 -> V_W
    L_f R   = (s_b/ELL) x2
    D(x)    = sup_w grad R . d = (s_b/ELL) q DELTA

Every term carries the same factor s_b/ELL, so the bracketed quantities below are
level-independent (the same factorization as in level_sweep.py).

Operating domain |x2| <= X2MAX, so that the boundary is compact and c_nom, G_bd are
finite; sg_run.py checks that the simulated speeds stay inside it.
"""
import numpy as np

X_BAR, BRK, ELL = 10.0, 1.0, 0.5
U_BAR, DELTA = 2.0, 0.5
TAU = 0.5
V_W, C_BETA = -0.9, 0.05
X2MAX = 7.0        # operating domain; sg_run.py checks the simulated speeds against it
DT = 0.01


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def beta(x2):
    """Control-surface effectiveness; zero at the flow speed x2 = V_W."""
    r = (x2 - V_W)**2
    return r/(r + C_BETA)


def h(x1, x2):
    return X_BAR - x1 - TAU*x2 - x2*np.abs(x2)/(2*BRK)


def h_of_level(b):
    return -ELL*np.log(b/(1 - b))


def risk(x1, x2):
    return sigmoid(-h(x1, x2)/ELL)


def risk_grad(x1, x2):
    """R, dR/dx1, dR/dx2, a(x) = dR/dx2 * beta(x2)."""
    R = sigmoid(-h(x1, x2)/ELL)
    sp = R*(1 - R)
    q = TAU + np.abs(x2)/BRK
    R1 = sp/ELL
    R2 = sp*q/ELL
    return R, R1, R2, R2*beta(x2)


def drift_terms(x1, x2):
    """R, L_f R, a(x), D(x) = sup_w grad R . d, ||grad R||."""
    R, R1, R2, a = risk_grad(x1, x2)
    LfR = R1*x2
    D = np.abs(R2)*DELTA
    G = np.sqrt(R1**2 + R2**2)
    return R, LfR, a, D, G


def x1_of_boundary(b, x2):
    """x1 on the level set {R = b} at speed x2."""
    return X_BAR - h_of_level(b) - TAU*x2 - x2*np.abs(x2)/(2*BRK)


# --------------------------------------------------------------- level-free brackets
# Dividing by s_b/ELL removes the level from every condition (see module docstring).
def bracket_suff(x2):
    """Slack of condition (15) / (s_b/ELL):  -[x2 - q beta U_BAR] - q DELTA."""
    q = TAU + np.abs(x2)/BRK
    return -x2 + q*beta(x2)*U_BAR - q*DELTA


def bracket_a5(x2, u_nom_over_ubar=1.0):
    """Drift-only slack with the nominal input / (s_b/ELL):  -[x2 + q beta u_nom] - q DELTA.

    u_nom_over_ubar = +1 is the worst case (nominal law accelerating toward the wall).
    """
    q = TAU + np.abs(x2)/BRK
    return -x2 - q*beta(x2)*U_BAR*u_nom_over_ubar - q*DELTA


def a_over_sb(x2):
    """a(x) / (s_b/ELL) = q beta."""
    return (TAU + np.abs(x2)/BRK)*beta(x2)


# --------------------------------------------------------------- controllers
def u_nominal(x1, x2, r, kp=1.0, kd=1.5):
    return np.clip(-kp*(x1 - r) - kd*x2, -U_BAR, U_BAR)


def gain(M, k0, k1, mu):
    """Same scheduled gain as model.py / di_model.py."""
    return k0*np.clip(1.0 - M/mu, 0.0, 1.0) + k1*np.maximum(0.0, -M)


def u_supervisor(x1, x2, r, B, k0, k1=10.0, mu=0.05):
    R, R1, R2, a = risk_grad(x1, x2)
    uu = u_nominal(x1, x2, r) - gain(B - R, k0, k1, mu)*a
    u = np.clip(uu, -U_BAR, U_BAR)
    return u, R, a, (u != uu)


def u_cbf(x1, x2, r, B, gamma=5.0, robust=False):
    """Exact closed-form 1-D CBF-QP on the same plant:
         min (u - u_nom)^2  s.t.  L_f R + a u [+ D] <= gamma (B - R),  u in U.
    The solution divides by a(x), so the implied gain is unbounded as a -> 0."""
    R, LfR, a, D, G = drift_terms(x1, x2)
    lhs = LfR + D if robust else LfR
    un = u_nominal(x1, x2, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        bnd = (gamma*(B - R) - lhs)/a
    u = np.where(a > 1e-12, np.minimum(un, bnd), un)
    return np.clip(u, -U_BAR, U_BAR), R, a


def step(x1, x2, u, w, dt=DT):
    """Explicit Euler on the control-affine plant."""
    return x1 + x2*dt, x2 + (beta(x2)*u + w)*dt

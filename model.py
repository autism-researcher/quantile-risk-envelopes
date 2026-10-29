"""Vehicle-following example for the IEEE Access revision (consistent with the theory).

State x = [s, v, vl]  (spacing [m], ego speed [m/s], lead speed [m/s]).
Plant (control-affine, relative degree 1 in the risk):
    s'  = vl - v
    v'  = u - CD*v^2 + d_v,        |d_v| <= DELTA_V
    vl' = w_l,                     |w_l| <= A_LEAD   (lead acceleration, exogenous)
so f(x) = [vl - v, -CD v^2, 0],  g = [0, 1, 0]^T,  d = [0, d_v, w_l].
Input box U = [U_MIN, U_MAX].

Risk functional (single normalized feature, logistic saturation):
    m(x) = s - S0 - H*v - (v^2 - vl^2)/(2*BD)   headway + stopping-distance margin [m]
    R(x) = sigma(-m(x)/ELL)  in (0,1)
"""
import numpy as np

CD, S0, BD, ELL, H = 0.002, 2.0, 6.0, 5.0, 1.0
U_MIN, U_MAX = -8.0, 4.0
DELTA_V, A_LEAD = 1.0, 4.0
DT = 0.01

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def margin(s, v, vl):
    return s - S0 - H*v - (v*v - vl*vl)/(2*BD)

def risk_grad(s, v, vl):
    """R, dR/ds, dR/dv, dR/dvl (analytic)."""
    z = -margin(s, v, vl)/ELL
    R = sigmoid(z); sp = R*(1-R)
    return R, -sp/ELL, sp*(H + v/BD)/ELL, -sp*vl/(BD*ELL)

def risk(s, v, vl):
    return sigmoid(-margin(s, v, vl)/ELL)

def margin_of_level(b):
    """Stopping margin m (metres) on the level set {R=b}."""
    return -ELL*np.log(b/(1-b))

def u_nominal(s, v, vl):
    """Nominal ACC law (constant-time-headway gap 5 m + 1.5 s * v), clipped to U."""
    return np.clip(0.5*(vl - v) + 0.2*(s - 5.0 - 1.5*v), U_MIN, U_MAX)

def drift_terms(s, v, vl):
    """L_f R, a(x)=dR/dv, worst-case disturbance term sup_d grad R . d."""
    R, Rs, Rv, Rl = risk_grad(s, v, vl)
    LfR = Rs*(vl - v) + Rv*(-CD*v*v)
    dist = np.abs(Rv)*DELTA_V + np.abs(Rl)*A_LEAD
    return R, LfR, Rv, dist

def gain(M, k0, k1, mu):
    """k(M) = k0*clip(1 - M/mu, 0, 1) + k1*max(0, -M): locally Lipschitz, k >= k0 for M <= 0,
    zero deep inside S (M >= mu), so the supervisor acts only in a boundary layer of width mu."""
    return k0*np.clip(1.0 - M/mu, 0.0, 1.0) + k1*np.maximum(0.0, -M)

def u_supervisor(s, v, vl, B, k0, k1, mu):
    R, Rs, Rv, Rl = risk_grad(s, v, vl)
    M = B - R
    k = gain(M, k0, k1, mu)
    uu = u_nominal(s, v, vl) - k*Rv
    u = np.clip(uu, U_MIN, U_MAX)
    return u, R, Rv, (u != uu)

def u_cbf(s, v, vl, B, gamma, robust=False):
    """Exact closed-form 1-D CBF-QP on the same plant:
       min (u-u_nom)^2 s.t. L_fR + a u [+ D(x) if robust] <= gamma*(B-R), u in U,
       where D(x) = sup_d grad R . d is the worst-case disturbance term."""
    R, LfR, Rv, D = drift_terms(s, v, vl)
    if robust:
        LfR = LfR + D
    un = u_nominal(s, v, vl)
    with np.errstate(divide='ignore', invalid='ignore'):
        bnd = (gamma*(B - R) - LfR)/Rv
    u = np.where(Rv > 1e-12, np.minimum(un, bnd), un)
    return np.clip(u, U_MIN, U_MAX), R

def step(s, v, vl, u, d_v, w_l, dt=DT):
    """Explicit Euler; speeds kept nonnegative (vehicles do not reverse)."""
    s2 = s + (vl - v)*dt
    v2 = np.maximum(v + (u - CD*v*v + d_v)*dt, 0.0)
    vl2 = np.maximum(vl + w_l*dt, 0.0)
    return s2, v2, vl2
